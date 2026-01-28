import requests
import json
from core.config import MOBIUS_HOST, MOBIUS_PORT, AE_NAME, CNT_NAME

'''
**********
**********
포멧아직 안맞춤
**********
**********
'''


def send_to_mobius(data: dict):
    """
    분석 결과(JSON)를 Mobius에 업로드 (ContentInstance 생성)
    """
    url = f"http://{MOBIUS_HOST}:{MOBIUS_PORT}/Mobius/{AE_NAME}/{CNT_NAME}"
    
    headers = {
        "Accept": "application/json",
        "X-M2M-RI": "12345",
        "X-M2M-Origin": "SOrigin", # AE ID (보통 SOrigin 사용)
        "Content-Type": "application/vnd.onem2m-res+json; ty=4" # ty=4는 cin(데이터)
    }
    
    payload = {
        "m2m:cin": {
            "con": data # data는 딕셔너리나 문자열이어야 함
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # 에러 시 예외 발생
        print(f"📡 [Mobius] Sent Success: {response.status_code}")
        return True
    except Exception as e:
        print(f"⚠️ [Mobius] Send Error: {e}")
        return False