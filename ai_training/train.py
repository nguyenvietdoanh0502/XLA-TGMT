import os
from ultralytics import YOLO

current_dir = os.path.dirname(os.path.abspath(__file__))


data_path = os.path.abspath(os.path.join(current_dir, '../datasets/phone-using-detection/data.yaml'))

runs_path = os.path.join(current_dir, 'runs')

def main():
    print(f"Đang load dữ liệu từ: {data_path}")
    
    model = YOLO('yolov8n.pt') 

    results = model.train(
        data=data_path,     
        epochs=50,          
        imgsz=640,          
        batch=16,           
        project='runs',     
        name='model_v1',    
        device=''           
    )

if __name__ == '__main__':
    main()