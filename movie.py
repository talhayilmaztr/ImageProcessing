import cv2
import numpy as np

def colorProcess(frame):
    red_houses = 0
    green_houses = 0
    blue_houses = 0
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([200, 255, 255])

    lower_green = np.array([36, 25, 25])
    upper_green = np.array([90, 255, 255])

    lower_blue = np.array([90, 20, 50])
    upper_blue = np.array([130, 255, 255])

    colors = {"kirmizi": (lower_red, upper_red), "yesil": (lower_green, upper_green), "mavi": (lower_blue, upper_blue)}

    for color_name, (lower_color, upper_color) in colors.items():
        mask = cv2.inRange(hsv, lower_color, upper_color)
        _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
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
        else:
            print(f"{color_name.capitalize()} ev bulunamadi!")

    return red_houses, green_houses, blue_houses

# Open the video file
video_path = "/Users/talhayilmaz/Desktop/OpenCv/ColorDrone/3.MP4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        red_houses, green_houses, blue_houses = colorProcess(frame)
        
        cv2.putText(frame, f"Kirmizi Kare Sayisi: {red_houses}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(frame, f"Yesil Kare Sayisi: {green_houses}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
        cv2.putText(frame, f"Mavi Kare Sayisi: {blue_houses}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        
        cv2.imshow('Processed Frame', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
