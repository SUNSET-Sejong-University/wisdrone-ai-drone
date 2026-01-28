import numpy as np

# ============================
# Risk 계산 파라미터 (튜닝 포인트)
# ============================
N0 = 30          # 위험 시작 인원 기준
D0 = 0.12        # 위험 밀집도 기준
TN = 10          # 인원 급증 기준
TD = 0.02        # 밀집 급증 기준

# EMA (튐 방지)
ALPHA = 0.3

# 이전 상태 저장 (드론 1대 기준 mock)
_prev_state = {
    "person_count": None,
    "density": None,
    "risk_smooth": None
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_density(objects, image_area):
    """ bbox 면적 기반 밀집도 """
    total_area = 0
    for obj in objects:
        x1, y1, x2, y2 = obj["bbox"]
        total_area += max(0, x2 - x1) * max(0, y2 - y1)
    return total_area / image_area if image_area > 0 else 0


def compute_risk(inference_result: dict) -> dict:
    """
    Mock inference 결과 → Crowd Risk 계산
    """

    global _prev_state

    objects = inference_result["objects"]
    meta = inference_result["meta"]

    W = meta["image_width"]
    H = meta["image_height"]
    image_area = W * H

    # ----------------------------
    # 1. 기본 지표
    # ----------------------------
    person_count = len(objects)
    density = compute_density(objects, image_area)

    # ----------------------------
    # 2. 변화량 (temporal)
    # ----------------------------
    prev_N = _prev_state["person_count"]
    prev_D = _prev_state["density"]

    dN = person_count - prev_N if prev_N is not None else 0
    dD = density - prev_D if prev_D is not None else 0

    # ----------------------------
    # 3. 점수화 (0~1)
    # ----------------------------
    count_score = sigmoid((person_count - N0) / 10)
    density_score = min(density / D0, 1.0)

    trend_score = min(
        max(0, dN) / TN + max(0, dD) / TD,
        1.0
    )

    # ----------------------------
    # 4. Risk Score (0~100)
    # ----------------------------
    risk_raw = 100 * (
        0.45 * count_score +
        0.45 * density_score +
        0.10 * trend_score
    )

    # EMA smoothing
    prev_risk = _prev_state["risk_smooth"]
    risk_smooth = (
        risk_raw if prev_risk is None
        else ALPHA * risk_raw + (1 - ALPHA) * prev_risk
    )

    # ----------------------------
    # 5. Risk Level
    # ----------------------------
    if risk_smooth < 30:
        level = "LOW"
    elif risk_smooth < 55:
        level = "MID"
    elif risk_smooth < 75:
        level = "HIGH"
    else:
        level = "CRITICAL"

    # ----------------------------
    # 6. 상태 업데이트
    # ----------------------------
    _prev_state.update({
        "person_count": person_count,
        "density": density,
        "risk_smooth": risk_smooth
    })

    return {
        "risk_score": round(float(risk_smooth), 2),
        "risk_level": level,
        "metrics": {
            "person_count": person_count,
            "density": round(float(density), 4),
            "dN": dN,
            "dDensity": round(float(dD), 4)
        }
    }
