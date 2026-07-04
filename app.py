import asyncio
import logging
import time
from io import BytesIO

import cv2
import numpy as np
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    Response,
    UploadFile,
    status,
)
from PIL import Image, UnidentifiedImageError
from paddleocr import PaddleOCR

app = FastAPI(title="PaddleOCR API")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

print("Loading OCR model...")

t0 = time.time()

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    enable_mkldnn=True,
    cpu_threads=4,
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)

print(f"Loaded in {time.time() - t0:.2f}s")

# Prevent OpenCV from creating its own thread pool
cv2.setNumThreads(1)

# -------------------------------------------------------------------
# Limits
# -------------------------------------------------------------------

MAX_UPLOAD_SIZE = 10 * 1024 * 1024      # 10 MB
MAX_PIXELS = 25_000_000                 # 25 MP
MAX_DIMENSION = 1600
OCR_TIMEOUT = 10                        # seconds
MAX_CONCURRENT_OCR = 2

OCR_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_OCR)

ALLOWED_FORMATS = {
    "JPEG",
    "PNG",
    "WEBP",
    "BMP",
}

Image.MAX_IMAGE_PIXELS = MAX_PIXELS

def invalid_image():
    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Invalid image upload.",
    )


@app.post("/ocr")
async def run_ocr(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
):
    start = time.perf_counter()

    response.headers["Cache-Control"] = "no-store"

    # Reject obviously oversized requests
    content_length = request.headers.get("content-length")
    if (
        content_length is not None
        and int(content_length) > MAX_UPLOAD_SIZE
    ):
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large.",
        )

    image_bytes = await file.read(MAX_UPLOAD_SIZE + 1)

    if not image_bytes:
        invalid_image()

    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large.",
        )

    # -----------------------------------------------------------------
    # Verify image with Pillow before decoding
    # -----------------------------------------------------------------

    try:
        img = Image.open(BytesIO(image_bytes))

        if img.format not in ALLOWED_FORMATS:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Unsupported image type.",
            )

        width, height = img.size

        if width * height > MAX_PIXELS:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail="Image dimensions too large.",
            )

        img.verify()

    except HTTPException:
        raise

    except (UnidentifiedImageError, OSError):
        invalid_image()

    except Exception:
        logger.exception("Image verification failed")
        invalid_image()

    # -----------------------------------------------------------------
    # Decode with OpenCV
    # -----------------------------------------------------------------

    try:
        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR,
        )
    except Exception:
        logger.exception("OpenCV decode failed")
        invalid_image()

    del image_bytes

    if image is None:
        invalid_image()

    if image.dtype != np.uint8:
        invalid_image()

    if image.ndim != 3:
        invalid_image()

    h, w = image.shape[:2]

    scale = min(1.0, MAX_DIMENSION / max(h, w))

    if scale < 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    # -----------------------------------------------------------------
    # OCR
    # -----------------------------------------------------------------

    try:
        async with OCR_SEMAPHORE:
            result = await asyncio.wait_for(
                asyncio.to_thread(ocr.predict, image),
                timeout=OCR_TIMEOUT,
            )

    except asyncio.TimeoutError:
        logger.warning("OCR timeout")
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="OCR processing timed out.",
        )

    except Exception:
        logger.exception("OCR failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OCR processing failed.",
        )

    finally:
        del image

    predictions = result[0].json if result else []

    return {
        "processing_time": round(time.perf_counter() - start, 3),
        "predictions": predictions,
    }
