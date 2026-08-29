import cv2
import numpy as np


def makePolygon(points):
    return np.array(points, dtype=np.int32).reshape((-1, 1, 2))


def isInside(box, points):
    x1, y1, x2, y2 = box
    objectPoint = ((x1 + x2) // 2, y2)
    polygon = makePolygon(points)

    return cv2.pointPolygonTest(polygon, objectPoint, False) >= 0


def getAlerts(detections, points):
    alerts = []
    importantClasses = ["person", "car", "motorcycle", "bus", "truck"]

    for detection in detections:
        if detection["class_name"] not in importantClasses:
            continue

        if isInside(detection["bbox"], points):
            alerts.append(
                f'{detection["class_name"].title()} entered restricted zone'
            )

    return alerts


def drawZone(frame, points):
    polygon = makePolygon(points)
    cv2.polylines(frame, [polygon], True, (0, 0, 255), 2)

    return frame
