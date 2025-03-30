from ultralytics import YOLO

# Load mô hình YOLOv8 đã được pre-train
#model = YOLO("D:/code/Computer_vision/Supermarket_checkout/models/yolov8n.pt")
model = YOLO("D:/code/Computer_vision/runs/detect/train3/weights/best.pt")

# Huấn luyện với dữ liệu có sẵn
model.train(data="D:/code/Computer_vision/Supermarket_checkout/datasets/dataset.yaml",epochs=100, imgsz=640,device = 'cuda:0', workers=0)
