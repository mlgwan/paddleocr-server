import time
import cv2
import numpy as np

from fastapi import FastAPI, UploadFile, File
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

@app.post("/ocr")
async def run_ocr(file: UploadFile = File(...)):
    start = time.time()

    image_bytes = await file.read()

    image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        return {"error": "Invalid image"}

    h, w = image.shape[:2]

    scale = 1600 / max(h, w)

    if scale < 1:
        image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
        )

    result = ocr.predict(image)

    json_data = result[0].json

    return {
        "processing_time": round(time.time() - start, 3),
        "predictions": json_data
    }