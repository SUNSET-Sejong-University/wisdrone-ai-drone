import os

# 🔴 서버의 IP로 변경하세요! (프론트/게이트웨이가 접속할 주소)
# localhost로 하면 젯슨에서 접속 못 함.
HOST_IP = "192.168.0.XX" 
PORT = 8000

# Mobius 설정
MOBIUS_HOST = "127.0.0.1" # Mobius가 같은 맥북에 있다면 localhost
MOBIUS_PORT = "7579"

#ae,cnt,cin 네임은 추후에 맞춰야함
AE_NAME = "drone1"
CNT_NAME = "report" # 분석 결과 저장할 컨테이너 이름

# 이미지 저장 경로
#이미지 ai서버 돌리는 로컬환경에 저장할거임(Static URL 생성)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(os.path.dirname(BASE_DIR), "static", "images")

# 폴더 없으면 생성
os.makedirs(IMAGE_DIR, exist_ok=True)