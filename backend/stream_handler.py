from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Generator, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class TopPrediction:
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: Tuple[int, int, int, int]


def open_working_camera(max_index: int = 4) -> Optional[cv2.VideoCapture]:
    backends = [
        ("DSHOW", cv2.CAP_DSHOW),
        ("MSMF", cv2.CAP_MSMF),
        ("ANY", cv2.CAP_ANY),
    ]

    for index in range(max_index + 1):
        for backend_name, backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            ok, _ = cap.read()
            if ok:
                print(f"Using camera index={index}, backend={backend_name}")
                return cap

            cap.release()

    return None


def decode_upload_image(raw_bytes: bytes) -> Optional[np.ndarray]:
    np_buffer = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)
    return image


def extract_top_prediction(result) -> Optional[TopPrediction]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None

    top_box = max(boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, top_box.xyxy[0].tolist())
    confidence = float(top_box.conf[0])
    class_id = int(top_box.cls[0])
    class_name = str(result.names[class_id])

    return TopPrediction(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=(x1, y1, x2, y2),
    )


def draw_top_prediction(frame: np.ndarray, prediction: Optional[TopPrediction]) -> np.ndarray:
    annotated = frame.copy()
    if prediction is None:
        return annotated

    x1, y1, x2, y2 = prediction.bbox_xyxy
    color = (0, 0, 255) if prediction.class_name == "using_phone" else (255, 0, 0)
    label = f"{prediction.class_name} {prediction.confidence:.2f}"

    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        annotated,
        label,
        (x1, max(20, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    return annotated


def predict_top1(model: YOLO, frame: np.ndarray, conf: float = 0.45, iou: float = 0.5) -> Optional[TopPrediction]:
    results = model(frame, conf=conf, iou=iou, agnostic_nms=True, verbose=False)
    return extract_top_prediction(results[0])


def generate_camera_stream(
    model: YOLO,
    conf: float = 0.45,
    iou: float = 0.5,
    alarm_threshold: int = 5,
) -> Generator[bytes, None, None]:
    cap = open_working_camera()
    if cap is None:
        raise RuntimeError("No working camera found.")

    phone_frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            prediction = predict_top1(model=model, frame=frame, conf=conf, iou=iou)
            annotated = draw_top_prediction(frame, prediction)

            is_using_phone = prediction is not None and prediction.class_name == "using_phone"
            if is_using_phone:
                phone_frame_count += 1
            else:
                phone_frame_count = max(0, phone_frame_count - 1)

            if phone_frame_count >= alarm_threshold:
                cv2.putText(
                    annotated,
                    "WARNING: DISTRACTED DRIVING!",
                    (30, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )

            ok, buffer = cv2.imencode(".jpg", annotated)
            if not ok:
                continue

            frame_bytes = buffer.tobytes()
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )
    finally:
        cap.release()


def prediction_to_dict(prediction: Optional[TopPrediction]) -> Dict:
    if prediction is None:
        return {
            "detected": False,
            "class_id": None,
            "class_name": None,
            "confidence": None,
            "bbox_xyxy": None,
        }

    return {
        "detected": True,
        "class_id": prediction.class_id,
        "class_name": prediction.class_name,
        "confidence": round(prediction.confidence, 4),
        "bbox_xyxy": list(prediction.bbox_xyxy),
    }
