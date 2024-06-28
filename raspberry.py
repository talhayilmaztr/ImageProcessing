import cv2
import numpy as np
from picamera import PiCamera
from picamera.array import PiRGBArray
import time

def colorProcess(frame):
    red_houses = 0
    green_houses = 0
    blue_houses = 0
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Renk aralıkları
    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])

    lower_green = np.array([36, 25, 25])
    upper_green = np.array([90, 255, 255])

    lower_blue = np.array([90, 20, 50])
    upper_blue = np.array([130, 255, 255])

    colors = {"kirmizi": [(lower_red, upper_red), (red_lower2, red_upper2)], "yesil": [(lower_green, upper_green)], "mavi": [(lower_blue, upper_blue)]}

    for color_name, bounds in colors.items():
        mask = None
        for (lower_color, upper_color) in bounds:
            if mask is None:
                mask = cv2.inRange(hsv, lower_color, upper_color)
            else:
                mask |= cv2.inRange(hsv, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 3000:
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    if color_name == "kirmizi":
                        red_houses += 1
                    elif color_name == "yesil":
                        green_houses += 1
                    elif color_name == "mavi":
                        blue_houses += 1

    return red_houses, green_houses, blue_houses, frame

# Raspberry Pi kamera ayarları
camera = PiCamera()
camera.resolution = (640, 480)
camera.framerate = 32
rawCapture = PiRGBArray(camera, size=(640, 480))

# Kamerayı başlatma ve biraz bekletme
camera.start_preview()
time.sleep(2)  # Kameranın ısınması için biraz zaman verin

try:
    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        image = frame.array
        red_houses, green_houses, blue_houses, processed_image = colorProcess(image)
        
        # Ekrana yazdırma işlemleri
        cv2.putText(processed_image, f"Kirmizi Ev Sayisi: {red_houses}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(processed_image, f"Yesil Ev Sayisi: {green_houses}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(processed_image, f"Mavi Ev Sayisi: {blue_houses}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Görüntüyü göster
        cv2.imshow('Processed Frame', processed_image)
        key = cv2.waitKey(1) & 0xFF
        rawCapture.truncate(0)

        # 'q' tuşuna basılırsa çık
        if key == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    camera.close()
