import os
import shutil
import cv2

# =========================
# 경로 설정 (네 환경 기준)
# =========================
VISDRONE_ROOT = r"C:\Users\kangn\Downloads\VisDrone2019-DET-val\VisDrone2019-DET-val"
IMG_DIR = os.path.join(VISDRONE_ROOT, "images")
ANN_DIR = os.path.join(VISDRONE_ROOT, "annotations")

OUT_ROOT = "dataset"
OUT_IMG_DIR = os.path.join(OUT_ROOT, "images", "val")
OUT_LBL_DIR = os.path.join(OUT_ROOT, "labels", "val")

os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# =========================
# VisDrone category 정의
# =========================
PEDESTRIAN_CLASS_ID = 1   # VisDrone 기준 pedestrian
YOLO_CLASS_ID = 0         # YOLO에서는 person = 0

# =========================
# 변환 시작
# =========================
ann_files = sorted(os.listdir(ANN_DIR))

print(f"🔄 Converting {len(ann_files)} annotation files...")

for ann_file in ann_files:
    if not ann_file.endswith(".txt"):
        continue

    img_name = ann_file.replace(".txt", ".jpg")
    img_path = os.path.join(IMG_DIR, img_name)

    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    if img is None:
        continue

    H, W, _ = img.shape

    yolo_lines = []

    with open(os.path.join(ANN_DIR, ann_file), "r") as f:
        for line in f:
            parts = line.strip().split(",")

            if len(parts) < 8:
                continue

            x, y, w, h = map(float, parts[:4])
            category = int(parts[5])

            # pedestrian만 사용
            if category != PEDESTRIAN_CLASS_ID:
                continue

            # YOLO 포맷 변환 (정규화)
            x_center = (x + w / 2) / W
            y_center = (y + h / 2) / H
            w_norm = w / W
            h_norm = h / H

            yolo_lines.append(
                f"{YOLO_CLASS_ID} "
                f"{x_center:.6f} {y_center:.6f} "
                f"{w_norm:.6f} {h_norm:.6f}"
            )

    # 사람이 한 명도 없으면 스킵
    if len(yolo_lines) == 0:
        continue

    # 이미지 복사
    shutil.copy(img_path, os.path.join(OUT_IMG_DIR, img_name))

    # 라벨 저장
    lbl_name = ann_file.replace(".txt", ".txt")
    with open(os.path.join(OUT_LBL_DIR, lbl_name), "w") as f:
        f.write("\n".join(yolo_lines))

print("✅ VisDrone → YOLO 변환 완료")
