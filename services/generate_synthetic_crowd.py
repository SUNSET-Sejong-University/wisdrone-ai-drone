import cv2
import numpy as np
import random
import os

# ==========================
# 설정
# ==========================
IMG_W, IMG_H = 1280, 720
NUM_IMAGES = 200

OUTPUT_DIR = "synthetic_data"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
LBL_DIR = os.path.join(OUTPUT_DIR, "labels")

os.makedirs(IMG_DIR, exist_ok=True)
os.makedirs(LBL_DIR, exist_ok=True)

# ==========================
# 유틸
# ==========================
def clamp(v, lo, hi):
    return max(lo, min(v, hi))

# ==========================
# 이미지 생성
# ==========================
for idx in range(NUM_IMAGES):

    # 배경 (아스팔트/콘서트 바닥 느낌)
    bg = np.random.randint(60, 100, (IMG_H, IMG_W, 3), dtype=np.uint8)

    # 사람 수 (위험도 분포 반영)
    person_count = random.choices(
        [5, 15, 30, 50, 80, 120],
        weights=[2, 4, 6, 6, 4, 2],
        k=1
    )[0]

    labels = []

    center_bias = min(person_count / 120, 1.0)

    for _ in range(person_count):
        # 사람 크기
        pw = random.randint(18, 28)
        ph = random.randint(35, 55)

        # 위치
        if random.random() < center_bias:
            cx = int(random.gauss(IMG_W // 2, IMG_W * 0.15))
            cy = int(random.gauss(IMG_H // 2, IMG_H * 0.15))
        else:
            cx = random.randint(0, IMG_W)
            cy = random.randint(0, IMG_H)

        x1 = clamp(cx - pw // 2, 0, IMG_W - 1)
        y1 = clamp(cy - ph // 2, 0, IMG_H - 1)
        x2 = clamp(cx + pw // 2, 0, IMG_W - 1)
        y2 = clamp(cy + ph // 2, 0, IMG_H - 1)

        # 사람 색
        color = (
            random.randint(120, 220),
            random.randint(120, 220),
            random.randint(120, 220)
        )

        # 사람 그리기 (타원)
        cv2.ellipse(
            bg,
            ((cx, cy), (pw, ph), random.randint(-15, 15)),
            color,
            -1
        )

        # YOLO label (class 0 = person)
        xc = ((x1 + x2) / 2) / IMG_W
        yc = ((y1 + y2) / 2) / IMG_H
        w = (x2 - x1) / IMG_W
        h = (y2 - y1) / IMG_H

        labels.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

    # 저장
    img_name = f"img_{idx:04d}.jpg"
    lbl_name = f"img_{idx:04d}.txt"

    cv2.imwrite(os.path.join(IMG_DIR, img_name), bg)

    with open(os.path.join(LBL_DIR, lbl_name), "w") as f:
        f.write("\n".join(labels))

    print(f"Generated {img_name} | persons={person_count}")

print("✅ Synthetic dataset generation complete.")
