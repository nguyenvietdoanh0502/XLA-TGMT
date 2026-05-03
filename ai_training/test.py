import cv2
import os
from ultralytics import YOLO


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(current_dir, '../backend/models/yolo_driver_detection.pt'))

model = YOLO(model_path)


def open_working_camera(max_index=4):
    backends = [
        ('DSHOW', cv2.CAP_DSHOW),
        ('MSMF', cv2.CAP_MSMF),
        ('ANY', cv2.CAP_ANY),
    ]

    for index in range(max_index + 1):
        for backend_name, backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            ok, _ = cap.read()
            if ok:
                print(f'Using camera index={index}, backend={backend_name}')
                return cap

            cap.release()

    return None


def draw_top1_prediction(image, result):
    annotated = image.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated, False

    top_box = max(boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, top_box.xyxy[0].tolist())
    conf = float(top_box.conf[0])
    class_id = int(top_box.cls[0])
    class_name = result.names[class_id]
    label = f'{class_name} {conf:.2f}'

    color = (0, 0, 255) if class_id == 1 else (255, 0, 0)
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

    return annotated, class_id == 1


cap = open_working_camera()
if cap is None:
    raise RuntimeError('No working camera found. Check camera permission or webcam availability.')

phone_frame_count = 0
ALARM_THRESHOLD = 5

while True:
    ret, frame = cap.read()
    if not ret:
        print('Cannot read frame from camera. Stopping...')
        break

    results = model(frame, conf=0.4, verbose=False)
    annotated_frame, is_using_phone_now = draw_top1_prediction(frame, results[0])

    if is_using_phone_now:
        phone_frame_count += 1
    else:
        phone_frame_count = max(0, phone_frame_count - 1)

    if phone_frame_count >= ALARM_THRESHOLD:
        cv2.putText(
            annotated_frame,
            'WARNING: DISTRACTED DRIVING!',
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            3,
        )

    cv2.imshow('Phone Usage Detection Test', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
