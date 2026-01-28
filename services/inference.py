import random
import cv2
import numpy as np
from ultralytics import YOLO

# YOLO 모델 로드 (서버 시작 시 1번만)
model = YOLO("yolov5n.pt")

# ============================
# Mock Inference
# ============================

def run_inference(image_path: str) -> dict:
    """
    YOLO 기반 실제 inference
    결과를 프로젝트 표준 포맷으로 변환
    """

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Image not found")

    H, W, _ = img.shape

    results = model(image_path, imgsz=416, conf=0.3)

    objects = []

    r = results[0]
    if r.boxes is not None:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # person class만 사용
            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            objects.append({
                "class": "person",
                "bbox": [x1, y1, x2, y2],
                "confidence": round(conf, 3)
            })

    return {
        "objects": objects,
        "meta": {
            "image_width": W,
            "image_height": H,
            "person_count": len(objects)
        }
    }