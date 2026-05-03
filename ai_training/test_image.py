import cv2
import os
import glob
from ultralytics import YOLO


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(current_dir, '../backend/models/yolo_driver_detection.pt'))

input_folder = os.path.join(current_dir, 'test_images')
output_folder = os.path.join(current_dir, 'test_results_images')

os.makedirs(input_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

print('Loading model...')
model = YOLO(model_path)


def draw_top1_prediction(image, result):
    annotated = image.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return annotated

    top_box = max(boxes, key=lambda b: float(b.conf[0]))
    x1, y1, x2, y2 = map(int, top_box.xyxy[0].tolist())
    conf = float(top_box.conf[0])
    class_id = int(top_box.cls[0])
    class_name = result.names[class_id]
    label = f'{class_name} {conf:.2f}'

    color = (0, 0, 255) if class_name == 'using_phone' else (255, 0, 0)
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


image_extensions = ['*.jpg', '*.jpeg', '*.png']
image_files = []
for ext in image_extensions:
    image_files.extend(glob.glob(os.path.join(input_folder, ext)))

if len(image_files) == 0:
    print(f'No images found in: {input_folder}')
    print('Please add images and run again.')
    raise SystemExit(0)

print(f'Found {len(image_files)} images. Running inference...')

count = 0
for img_path in image_files:
    filename = os.path.basename(img_path)

    img = cv2.imread(img_path)
    if img is None:
        print(f'Cannot read file: {filename}')
        continue

    results = model(img, conf=0.45, verbose=False)
    annotated_img = draw_top1_prediction(img, results[0])

    save_path = os.path.join(output_folder, f'result_{filename}')
    cv2.imwrite(save_path, annotated_img)
    print(f'Processed: {filename}')
    count += 1

print(f'Done. Saved {count} image(s).')

