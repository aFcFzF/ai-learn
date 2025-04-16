from ultralytics import YOLO
yolo = YOLO(model="yolo11n.pt", task="detect")
yolo(source="screen", save=True)
