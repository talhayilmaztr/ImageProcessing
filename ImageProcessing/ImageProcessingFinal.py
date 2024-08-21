import cv2
import numpy as np
from scipy.spatial import distance
import time

def apply_morphology(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def colorProcess(frame):
    red_centers = []
    green_triangles = []
    blue_centers = []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])

    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])

    lower_blue = np.array([90, 100, 100])
    upper_blue = np.array([130, 255, 255])

    colors = {"kirmizi": [(lower_red, upper_red), (red_lower2, red_upper2)], "yesil": [(lower_green, upper_green)],
              "mavi": [(lower_blue, upper_blue)]}

    for color_name, bounds in colors.items():
        mask = None
        for (lower_color, upper_color) in bounds:
            if mask is None:
                mask = cv2.inRange(hsv, lower_color, upper_color)
            else:
                mask |= cv2.inRange(hsv, lower_color, upper_color)

        mask = apply_morphology(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1840:
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.08 * perimeter, True)

                if len(approx) == 4 and solidity > 0.8 and color_name != "yesil":
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    center = (x + w / 2, y + h / 2)
                    if color_name == "kirmizi":
                        red_centers.append((x, y, w, h))
                    elif color_name == "mavi":
                        blue_centers.append((x, y, w, h))
                elif len(approx) == 3 and color_name == "yesil":
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    green_triangles.append((x, y, w, h))

    return red_centers, green_triangles, blue_centers, frame

def group_rectangles_by_proximity(rectangles, distance_threshold):
    groups = []
    used = set()

    for i in range(len(rectangles)):
        if i in used:
            continue
        group = [rectangles[i]]
        used.add(i)
        for j in range(i + 1, len(rectangles)):
            if j in used:
                continue
            for (x1, y1, w1, h1) in group:
                x2, y2, w2, h2 = rectangles[j]
                center1 = (x1 + w1 / 2, y1 + h1 / 2)
                center2 = (x2 + w2 / 2, y2 + h2 / 2)
                if distance.euclidean(center1, center2) < distance_threshold:
                    group.append(rectangles[j])
                    used.add(j)
                    break
        groups.append(group)

    return groups

# Video dosyasını aç
video_path = "/Users/talhayilmaz/pythonProject2/en4.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
else:
    city_start_times = {"A Sehri": None, "B Sehri": None, "C Sehri": None}
    detected_cities = {"A Sehri": False, "B Sehri": False, "C Sehri": False}
    city_display_times = {"A Sehri": [], "B Sehri": [], "C Sehri": []}
    city_rectangles = {"A Sehri": None, "B Sehri": None, "C Sehri": None}
    city_texts = {"A Sehri": "", "B Sehri": "", "C Sehri": ""}
    city_detection_counts = {"A Sehri": 0, "B Sehri": 0, "C Sehri": 0}
    city_status_start_times = {"A Sehri": None, "B Sehri": None, "C Sehri": None}
    display_landing_text = False
    landing_text_counter = 0

    frame_skip = 2
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[:, 10:-10]  # Sağ ve sol kenarlardan 10 piksel kes

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        red_centers, green_triangles, blue_centers, processed_frame = colorProcess(frame)

        all_rectangles = red_centers + green_triangles + blue_centers
        distance_threshold = 400

        groups = group_rectangles_by_proximity(all_rectangles, distance_threshold)

        for group in groups:
            if len(group) > 1:
                x_min = min(x for x, y, w, h in group)
                y_min = min(y for x, y, w, h in group)
                x_max = max(x + w for x, y, w, h in group)
                y_max = max(y + h for x, y, w, h in group)
                city_name = None

                red_count = sum(1 for (x, y, w, h) in group if (x, y, w, h) in red_centers)
                green_count = sum(1 for (x, y, w, h) in group if (x, y, w, h) in green_triangles)
                blue_count = sum(1 for (x, y, w, h) in group if (x, y, w, h) in blue_centers)

                if red_count == 2 and green_count == 2 and blue_count == 1:
                    city_name = "C Sehri"
                elif red_count == 2 and green_count == 1 and blue_count == 1 and city_detection_counts["A Sehri"] < 5:
                    city_name = "A Sehri"
                elif red_count == 1 and green_count == 3 and blue_count == 1:
                    city_name = "B Sehri"

                if city_name:
                    if city_start_times[city_name] is None:
                        city_start_times[city_name] = time.time()
                    elif time.time() - city_start_times[city_name] >= 2:
                        detected_cities[city_name] = True
                        city_display_times[city_name].append((time.time(), (x_min, y_min, x_max, y_max)))
                        city_rectangles[city_name] = (x_min, y_min, x_max, y_max)
                        if city_name == "C Sehri":
                            city_texts[city_name] = "C Sehri Algilandi, Inis Yapiliyor"
                        else:
                            city_texts[city_name] = f"{city_name} Algilandi. Yanlis sehir"
                        city_detection_counts[city_name] += 1
                        city_status_start_times[city_name] = time.time()

        red_houses = len(red_centers)
        green_houses = len(green_triangles)
        blue_houses = len(blue_centers)

        cv2.putText(processed_frame, f"Kirmizi Kare Sayisi: {red_houses}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2)
        cv2.putText(processed_frame, f"Yesil Ucgen Sayisi: {green_houses}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)
        cv2.putText(processed_frame, f"Mavi Kare Sayisi: {blue_houses}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 0, 0), 2)

        current_time = time.time()

        for city_name, display_times in city_display_times.items():
            if city_rectangles[city_name] is not None:
                x_min, y_min, x_max, y_max = city_rectangles[city_name]
                if current_time - display_times[0][0] < 5:
                    cv2.rectangle(processed_frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 3)
                    text_size = cv2.getTextSize(city_texts[city_name], cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                    text_x = x_min + (x_max - x_min - text_size[0]) // 2
                    text_y = y_max - 10  # Metni dikdörtgenin içine yazmak için -10 piksel yukarı kaydır
                    cv2.putText(processed_frame, city_texts[city_name], (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                else:
                    city_rectangles[city_name] = None
                    city_display_times[city_name] = []

        for city_name in detected_cities:
            if detected_cities[city_name]:
                if city_name == "C Sehri":
                    if display_landing_text:
                        cv2.putText(processed_frame, "Inis Yapiliyor", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    if time.time() - city_status_start_times[city_name] > 16:
                        detected_cities[city_name] = False
                    else:
                        if int(time.time() - city_status_start_times[city_name]) % 2 == 0:
                            display_landing_text = not display_landing_text
                        landing_text_counter += 1
                else:
                    if current_time - city_status_start_times[city_name] < 3:
                        cv2.putText(processed_frame, "Devam Ediliyor", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    else:
                        detected_cities[city_name] = False

        cv2.imshow('Processed Frame', processed_frame)

        if cv2.waitKey(16) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
