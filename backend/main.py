from __future__ import annotations

import os
from contextlib import asynccontextmanager

import cv2
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from ultralytics import YOLO

from stream_handler import (
    decode_upload_image,
    draw_top_prediction,
    generate_camera_stream,
    predict_top1,
    prediction_to_dict,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolo_driver_detection.pt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found: {MODEL_PATH}")

    app.state.model = YOLO(MODEL_PATH)
    print("Model loaded.")
    yield


app = FastAPI(title="Driver Detection API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Driver Detection API is running"}


@app.post("/predict/image")
async def predict_image(
    file: UploadFile = File(...),
    conf: float = Query(0.45, ge=0.01, le=0.99),
    iou: float = Query(0.5, ge=0.01, le=0.99),
):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    image = decode_upload_image(raw_bytes)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    prediction = predict_top1(model=app.state.model, frame=image, conf=conf, iou=iou)
    return JSONResponse(content=prediction_to_dict(prediction))


@app.post("/predict/image/annotated")
async def predict_annotated_image(
    file: UploadFile = File(...),
    conf: float = Query(0.45, ge=0.01, le=0.99),
    iou: float = Query(0.5, ge=0.01, le=0.99),
):
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    image = decode_upload_image(raw_bytes)
    if image is None:
        raise HTTPException(status_code=400, detail="Invalid image format.")

    prediction = predict_top1(model=app.state.model, frame=image, conf=conf, iou=iou)
    annotated = draw_top_prediction(image, prediction)

    ok, buffer = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Cannot encode output image.")

    return StreamingResponse(iter([buffer.tobytes()]), media_type="image/jpeg")


@app.get("/camera/stream")
def camera_stream(
    conf: float = Query(0.45, ge=0.01, le=0.99),
    iou: float = Query(0.5, ge=0.01, le=0.99),
    alarm_threshold: int = Query(5, ge=1, le=60),
):
    def frame_generator():
        try:
            yield from generate_camera_stream(
                model=app.state.model,
                conf=conf,
                iou=iou,
                alarm_threshold=alarm_threshold,
            )
        except RuntimeError as exc:
            error_frame = f"Camera error: {exc}".encode("utf-8")
            yield (
                b"--frame\r\n"
                b"Content-Type: text/plain\r\n\r\n" + error_frame + b"\r\n"
            )

    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# Run with: uvicorn main:app --reload --host 0.0.0.0 --port 8000
