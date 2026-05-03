# XLA-TGMT

Hệ thống phát hiện tài xế dùng điện thoại khi lái xe, gồm:
- `ai_training/`: train/test model YOLO
- `backend/`: FastAPI phục vụ API detect ảnh + realtime camera stream
- `frontend/`: ReactJS UI để upload ảnh và xem stream

## 1) Cấu trúc thư mục
```text
XLA-TGMT/
├── ai_training/
├── backend/
│   ├── main.py
│   ├── stream_handler.py
│   ├── requirements.txt
│   └── models/
│       └── yolo_driver_detection.pt
├── frontend/
└── datasets/
```

## 2) Yêu cầu môi trường
- Windows + Python `3.13` (khuyến nghị ổn định cho `ultralytics/opencv/numpy`)
- Node.js `>=18`
- Model file: `backend/models/yolo_driver_detection.pt`

## 3) Setup Python (khuyến nghị dùng venv)
Chạy tại thư mục gốc project:
```bash
cd e:\XLA-TGMT
py -3.13 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

## 4) Chạy Backend (FastAPI)
```bash
cd backend
py -3.13 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Sau khi chạy:
- API base: `http://127.0.0.1:8000`
- Swagger docs: `http://127.0.0.1:8000/docs`

## 5) Chạy Frontend (React + Vite)
```bash
cd frontend
npm install
npm run dev
```

Mở `http://localhost:5173` và nhập backend URL là `http://127.0.0.1:8000`.

## 6) API chính của Backend
- `GET /`  
  Health check.
- `POST /predict/image`  
  Upload ảnh, trả JSON top-1 prediction:
  - `class_name`, `confidence`, `bbox_xyxy`
- `POST /predict/image/annotated`  
  Upload ảnh, trả ảnh đã vẽ bounding box.
- `GET /camera/stream`  
  MJPEG realtime stream từ webcam.

Ví dụ test nhanh:
```bash
curl -X POST "http://127.0.0.1:8000/predict/image" -F "file=@path/to/test.jpg"
```

## 7) Chạy script trong `ai_training`
Ví dụ:
```bash
py -3.13 ai_training/test.py
py -3.13 ai_training/test_image.py
```

Lưu ý:
- Nên chạy cùng interpreter Python 3.13 để tránh lệch package.




