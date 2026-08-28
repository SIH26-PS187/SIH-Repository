import cv2 as cv
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

capture = cv.VideoCapture(0)

while True:

    success, frame = capture.read()

    if not success:
        break

    results = model(frame)

    newframe = results[0].plot()

    cv.imshow("YOLO11 Detection", newframe)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()