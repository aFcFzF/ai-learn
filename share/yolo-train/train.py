from ultralytics import YOLO
model = YOLO(model="yolo11n.pt", task="detect")

results = model.train(data="train.yaml", project="./", epochs=1, batch=100, patience=3, name="runs/train")

model.export(format="onnx")

