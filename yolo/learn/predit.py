from ultralytics import YOLO
yolo = YOLO(model="yolo11n.pt", task="detect")
yolo(
    source="./beij.webp", 
    save=True, 
    save_txt=True, 
    show=True, 
    show_labels=True,
    show_boxes=True,
    max_dep=300,
    line_width=1
)
