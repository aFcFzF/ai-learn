from ultralytics import YOLO
model = YOLO(model="yolo11s.pt")

results = model.train(
  data="/data/workspace/ai-learn/share/proj/mole/train/datasets/best/train.yaml",
  project="/data/workspace/ai-learn/share/proj/mole/train/datasets/best",
  epochs=10,
  batch=50,
  patience=200,
  name="runs/train"
)

model.export(format="onnx")
