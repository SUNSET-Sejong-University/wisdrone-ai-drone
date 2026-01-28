from inference import run_inference
from risk import compute_risk

img = "synthetic_data/images/img_0010.jpg"

infer = run_inference(img)
risk = compute_risk(infer)

print("Inference:", infer["meta"])
print("Risk:", risk)
