import os
import shutil
from ultralytics import YOLO


current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.abspath(os.path.join(current_dir, '../backend/models/yolo_driver_detection.pt'))
data_path = os.path.abspath(os.path.join(current_dir, '../datasets/phone-using-detection/data.yaml'))


target_runs_dir = os.path.join(current_dir, 'runs', 'test_results')


model = YOLO(model_path)
print("Đang đánh giá...")

metrics = model.val(data=data_path, split='test', name='test_results')


actual_save_dir = str(metrics.save_dir)

if os.path.exists(actual_save_dir) and actual_save_dir != target_runs_dir:
    

    os.makedirs(os.path.dirname(target_runs_dir), exist_ok=True)
    

    if os.path.exists(target_runs_dir):
        shutil.rmtree(target_runs_dir)
        

    shutil.move(actual_save_dir, target_runs_dir)
    print("XONG!")