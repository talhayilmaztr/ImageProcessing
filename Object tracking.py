import cv2
import numpy as np
cv2

cap = cv2.VideoCapture()

while True:
    _ , frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    sensivitiy = 15

    lower_weight = np.array([0,0,255-sensivitiy])
    upper_weight = np.array([255,sensivitiy,255])

    mask = cv2.inRange(hsv, lower_weight,upper_weight)

