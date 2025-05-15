from ultralytics import YOLO
model = YOLO(model="yolo11n.pt", task="detect")
results = model.train(data="train.yaml", epochs=2000, batch=100, patience=30)

print(results)
