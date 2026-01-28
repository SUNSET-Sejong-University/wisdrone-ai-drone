from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
import uvicorn
import shutil
import os
import time

from core.config import IMAGE_DIR, HOST_IP, PORT
from services.inference import run_inference
from services.mobius import send_to_mobius
from services.risk import compute_risk


app = FastAPI()

# 1. 정적 파일 마운트 (이게 있어야 URL로 이미지 접근 가능)
# 예: http://192.168.0.XX:8000/static/images/파일.jpg
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def read_root():
    return {"message": "AIoT Drone Analysis Server Running"}

@app.post("/analyze")
async def analyze_snapshot(file: UploadFile = File(...)):
    """
    [Flow]
    1. Gateway에서 스냅샷 수신
    2. 로컬 /static/images 폴더에 저장
    3. 접속 가능한 URL 생성
    4. AI 추론 실행
    5. 결과 + URL을 Mobius로 전송
    """
    
    # 1. 파일명 생성 (타임스탬프)
    filename = f"{int(time.time())}.jpg"
    file_path = os.path.join(IMAGE_DIR, filename)
    
    # 2. 이미지 로컬 저장
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    # 3. URL 생성
    image_url = f"http://{HOST_IP}:{PORT}/static/images/{filename}"
    print(f"📸 Image Saved: {image_url}")
    
    # 4. AI 추론 (Mock)
    ai_result = run_inference(file_path)

    # 5. Risk 계산
    risk_result = compute_risk(ai_result)

    # 6. Mobius 전송 데이터 구성 (최종 Report)
    report_data = {
        "timestamp": int(time.time()),
        "image_url": image_url,
        "risk": {
            "score": risk_result["risk_score"],
            "level": risk_result["risk_level"]
        },
        "metrics": risk_result["metrics"],
        "analysis": {
            "person_count": ai_result["meta"]["person_count"]
        }
    }

    # 7. Mobius 전송
    if os.getenv("ENABLE_MOBIUS", "0") == "1":
        send_to_mobius(report_data)

    print("[LOCAL TEST REPORT]")
    print(report_data)

    
    return {"status": "success", "data": report_data}

if __name__ == "__main__":
    print(f"🚀 AI Server starting on {HOST_IP}:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)