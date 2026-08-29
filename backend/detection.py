from ultralytics import YOLO
model=YOLO('backend/yolo11n.pt')

def detect(frame):
    results = model(frame)
    annotatedframe = results[0].plot()
    return annotatedframe

