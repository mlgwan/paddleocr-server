import time
import cv2
import numpy as np

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from paddleocr import PaddleOCR

app = FastAPI(title="PaddleOCR API")

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

print(f"Loaded in {time.time()-t0:.2f}s")

# Limits

MAX_UPLOAD_SIZE = 10 * 1024 * 1024      # 10 MB
MAX_PIXELS = 25_000_000                 # 25 MP
MAX_DIMENSION = 1600

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/bmp",
}

@app.post("/ocr")
async def run_ocr(request: Request, file: UploadFile = File(...)):
    start = time.perf_counter()

    # Validate MIME type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image type.",
        )

    # Check size limit
    image_bytes = await file.read(MAX_UPLOAD_SIZE + 1)

    if len(image_bytes) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File too large.",
        )

    if not image_bytes:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty upload.",
        )

    # Decode safely
    try:
        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR,
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image.",
        )

    if image is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not decode image.",
        )

    h, w = image.shape[:2]

    # Prevent decompression bombs
    if h * w > MAX_PIXELS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image dimensions too large.",
        )

    # Resize
    scale = min(1.0, MAX_DIMENSION / max(h, w))

    if scale < 1.0:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    # OCR
    try:
        result = ocr.predict(image)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="OCR processing failed.",
        )

    if not result:
        predictions = []
    else:
        predictions = result[0].json

    return {
        "processing_time": round(time.perf_counter() - start, 3),
        "predictions": predictions,
    }
