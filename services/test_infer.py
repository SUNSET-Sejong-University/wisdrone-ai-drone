from inference import run_inference

img_path = "dataset/images/val/0000364_01765_d_0000782.jpg"

result = run_inference(img_path)

print("Person count:", result["meta"]["person_count"])
print("First 3 objects:")
for obj in result["objects"][:3]:
    print(obj)
