# 🛸 NAEON AIr – Crowd Risk Analysis Server

**AIoT-based Crowd Safety Digital Twin (Drone View)**

본 프로젝트는 **드론에서 촬영된 영상/이미지를 기반으로 군중 밀집·이동·압박을 분석하여 사고 위험도를 계산**하는 AI 서버입니다.
YOLO 기반 사람 인식 결과를 바탕으로 **군집 이동(flow)과 압박(pressure)**을 추정하고, 이를 **위험도(risk score)**로 통합합니다.

This project is an **AI inference server for crowd safety**, designed to analyze **crowd density, movement, and pressure** from drone images and estimate potential risk in real time.

---

## 📁 Project Structure

```text
wisdrone-ai-drone/
├── core/
│   └── config.py
├── dataset/
│   ├── images/
│   ├── labels/
│   └── dataset.yaml
├── models/
│   └── best.pt
├── services/
│   ├── flow.py
│   ├── inference.py
│   ├── mobius.py
│   └── risk.py
├── main.py
├── requirements.txt
└── .gitignore
```

---

## 📌 Folder & File Descriptions

### 🔹 `core/`

**Configuration layer**

* `config.py`
  서버 설정 파일
  (이미지 저장 경로, 호스트 IP, 포트 등)

> Server-level configuration (paths, IP, port)

---

### 🔹 `dataset/`

**Training & validation dataset (YOLO format)**

* `images/`
  학습/검증 이미지
* `labels/`
  YOLO format 라벨 파일 (`.txt`)
* `dataset.yaml`
  YOLO 학습용 데이터셋 설정 파일

> Dataset used for fine-tuning YOLO models (VisDrone-based)

---

### 🔹 `models/`

**YOLO model weights**

* `best.pt`
  Fine-tuned YOLO model (used for inference)

> Trained detection models for person detection (top-down drone view)

---

### 🔹 `services/`

**Core AI logic modules**

#### `inference.py`

* YOLO 기반 사람 검출
* 이미지 → bbox → person objects 반환

> Runs YOLO inference and extracts person bounding boxes

---

#### `flow.py`

* 프레임 간 사람 위치 중심점 추출
* 공간 군집화(K-Means)
* 군집별 이동 / 정체 상태 분석
* **압박(pressure) 지표 계산**

> Analyzes crowd movement and pressure based on clustered motion

---

#### `risk.py`

* 인원 수, 밀집도, 변화량, 압박 지표를 통합
* 최종 **Risk Score (0–100)** 및 단계 산출

> Computes final crowd risk score using density, trend, and pressure

---

#### `mobius.py`

* oneM2M Mobius 서버 연동 모듈
* 위험 분석 결과 전송 (옵션)

> Sends analysis results to Mobius (if enabled)

---

### 🔹 `main.py`

**FastAPI entry point**

* 이미지 업로드 API (`POST /analyze`)
* Inference → Flow → Risk 전체 파이프라인 실행
* 결과 JSON 반환 및 Mobius 연동

> Main FastAPI server for real-time analysis

---

### 🔹 `requirements.txt`

* Python dependencies (FastAPI, Ultralytics, NumPy, etc.)

---

## 🚀 Execution

```bash
pip install -r requirements.txt
python main.py
```

API Test (example):

```python
import requests

url = "http://127.0.0.1:8000/analyze"
with open("dataset/images/crowd/02/00001.jpg", "rb") as f:
    r = requests.post(url, files={"file": f})
print(r.json())
```

---

## 🧠 Key Concept

> **Crowd accidents often occur when a stationary crowd is compressed by adjacent moving groups.**
> This system explicitly models that phenomenon using **cluster-level flow and pressure**, rather than relying solely on headcount.

