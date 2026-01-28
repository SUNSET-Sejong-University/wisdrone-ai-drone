from inference import run_inference
from risk import compute_risk

img = "dataset/images/val/0000301_00001_d_0000156.jpg"

for i in range(3):
    infer = run_inference(img)
    risk = compute_risk(infer)
    print(f"[Frame {i}] {risk}")

print(risk)
