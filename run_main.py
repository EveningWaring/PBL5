import cv2
from ultralytics import YOLO
import numpy as np


def yolo_predict_simple(image_path, model_path, conf_threshold=0.5, iou_threshold=0.45):
    """
    Dự đoán YOLO với NMS để loại bỏ các boxes trùng lập
    """
    # Load model
    model = YOLO(model_path)

    # Đọc ảnh và kiểm tra
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌ Không thể đọc ảnh từ: {image_path}")
        return

    results = model(image_path, conf=conf_threshold, iou=iou_threshold)

    print("🎯 KẾT QUẢ DỰ ĐOÁN YOLO (ĐÃ ÁP DỤNG NMS)")
    print("=" * 50)
    print(f"Confidence threshold: {conf_threshold}")
    print(f"IOU threshold (NMS): {iou_threshold}")
    print("=" * 50)

    # Xử lý kết quả
    for r in results:
        boxes = r.boxes

        # Nếu có boxes, kiểm tra số lượng trước/sau NMS
        if len(boxes) > 0:
            print(f"✅ Số lượng boxes sau NMS: {len(boxes)}")

        for i, box in enumerate(boxes):
            # Lấy tọa độ
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Tính kích thước
            width = x2 - x1
            height = y2 - y1

            # Thông tin
            conf = box.conf[0]
            cls = int(box.cls[0])
            class_name = model.names[cls]

            # Vẽ box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{class_name} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                image,
                f"{width}x{height}",
                (x1, y1 - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

            # In kết quả
            print(f"Box {i+1}: {class_name} - {width}x{height} px (conf: {conf:.2f})")

    # Hiển thị
    cv2.imshow("YOLO Prediction - Box XANH (Đã áp dụng NMS)", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# SỬ DỤNG
if __name__ == "__main__":
    image_path = r"D:\Xuonggg\BoneFractureYolo8\train\images\image1_1197_png.rf.76fcc01f37eb7297d4b43d4859fde95a.jpg"
    model_path = r"D:\Gayxuong\Train_9_9\train\weights\best.pt"

    yolo_predict_simple(
        image_path,
        model_path,
        conf_threshold=0.4,  # Tăng để loại bỏ predictions yếu
        iou_threshold=0.45,  # Giảm để xóa boxes trùng nhiều hơn
    )
