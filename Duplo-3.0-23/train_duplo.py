from ultralytics import YOLO
import os

# 1. 모델 로드
model = YOLO('yolov8n-seg.pt')

# 2. 현재 폴더에 있는 data.yaml 사용
data_path = 'data.yaml' 

# 3. 학습 시작 (RTX 5060 Ti 활용)
model.train(
    data=data_path,
    epochs=50,
    imgsz=640,
    device=0,  # GPU 사용
    project='duplo_project',
    name='train_v1'
)
