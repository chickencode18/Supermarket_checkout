from ultralytics import YOLO
import cv2

# Tải mô hình đã huấn luyện
model = YOLO("D:/code/Computer_vision/runs/detect/train4/weights/best.pt")

# Mở camera
cap = cv2.VideoCapture(0)  # 0 là camera mặc định

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Dự đoán trên khung hình từ camera
    results = model(frame,conf = 0.3)  # Không cần `.predict()`, YOLOv8 dùng trực tiếp model(frame)

    # Lấy kết quả dự đoán
    for result in results:
        boxes = result.boxes  # Lấy danh sách bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Lấy tọa độ khung
            conf = box.conf[0].item()  # Độ tin cậy
            cls = int(box.cls[0].item())  # Lớp dự đoán

            # Vẽ bounding box
            label = f"{model.names[cls]} {conf:.2f}"  # Lấy tên nhãn từ model.names
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Hiển thị khung hình
    cv2.imshow("YOLOv8 Detection", frame)

    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
