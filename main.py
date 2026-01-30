from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import time
from collections import deque

from core.config import IMAGE_DIR, HOST_IP, PORT
from services.inference import run_inference
from services.mobius import send_to_mobius
from services.risk import compute_risk
from services.flow import compute_cluster_flow

app = FastAPI()

# 정적 파일
app.mount("/static", StaticFiles(directory="static"), name="static")

# 최근 프레임 버퍼
FRAME_BUFFER_SIZE = 5
frame_buffer = deque(maxlen=FRAME_BUFFER_SIZE)


@app.get("/")
def read_root():
    return {"message": "AIoT Drone Analysis Server Running"}


@app.post("/analyze")
async def analyze_snapshot(file: UploadFile = File(...)):

    # 1. 이미지 저장
    filename = f"{int(time.time())}.jpg"
    file_path = os.path.join(IMAGE_DIR, filename)

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    image_url = f"http://{HOST_IP}:{PORT}/static/images/{filename}"
    print(f"📸 Image Saved: {image_url}")

    # 2. YOLO inference
    ai_result = run_inference(file_path)

    # 3. 프레임 버퍼 업데이트
    frame_buffer.append(ai_result)

    # 4. Flow / pressure 계산
    if len(frame_buffer) >= 2:
        flow_result = compute_cluster_flow(list(frame_buffer))
    else:
        flow_result = {"clusters": [], "pressure": 0.0}

    # 5. Risk 계산 (flow 반영)
    risk_result = compute_risk(ai_result, flow_result)

    # 6. 리포트 구성
    report_data = {
        "timestamp": int(time.time()),
        "image_url": image_url,
        "risk": {
            "score": risk_result["risk_score"],
            "level": risk_result["risk_level"]
        },
        "metrics": risk_result["metrics"],   # pressure 포함
        "analysis": {
            "person_count": ai_result["meta"]["person_count"]
        },
        "temporal": {
            "clusters": flow_result.get("clusters", [])
        }
    }

    # 7. Mobius 전송 (옵션)
    if os.getenv("ENABLE_MOBIUS", "0") == "1":
        send_to_mobius(report_data)

    # 8. 로컬 디버그 로그
    print(
        f"[FRAME] N={risk_result['metrics']['person_count']} "
        f"pressure={risk_result['metrics']['pressure']} "
        f"risk={risk_result['risk_score']} ({risk_result['risk_level']})"
    )

    return {"status": "success", "data": report_data}


if __name__ == "__main__":
    print(f"🚀 AI Server starting on {HOST_IP}:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
