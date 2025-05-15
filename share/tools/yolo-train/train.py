from ultralytics import YOLO
model = YOLO(model="yolo11s.pt", task="detect")

results = model.train(data="train.yaml", project="./", epochs=200, batch=100, patience=50, name="runs/train")

model.export(format="onnx")
