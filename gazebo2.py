import cv2
import numpy as np
import time
from dronekit import connect, VehicleMode
from pymavlink import mavutil
import argparse
from scipy.spatial import distance

def apply_morphology(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def colorProcess(frame):
    red_centers = []
    green_centers = []
    blue_centers = []
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

        mask = apply_morphology(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 4000:
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
                if len(approx) == 4 and solidity > 0.8:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    center = (x + w / 2, y + h / 2)
                    if color_name == "kirmizi":
                        red_centers.append((x, y, w, h))
                    elif color_name == "yesil":
                        green_centers.append((x, y, w, h))
                    elif color_name == "mavi":
                        blue_centers.append((x, y, w, h))

    return red_centers, green_centers, blue_centers, frame

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

def connectMyCopter(connection_string):
    baud_rate = 57600
    vehicle = connect(connection_string, baud=baud_rate, wait_ready=True)
    return vehicle

def arm_and_takeoff(aTargetAltitude, vehicle):
    print("Basic pre-arm checks")
    while not vehicle.is_armable:
        print(" Waiting for vehicle to initialise...")
        time.sleep(1)
        
    print("Arming motors")
    vehicle.mode = VehicleMode("GUIDED")
    vehicle.armed = True

    while not vehicle.armed:      
        print(" Waiting for arming...")
        time.sleep(1)

    print("Taking off!")
    vehicle.simple_takeoff(aTargetAltitude)

    while True:
        print(" Altitude: ", vehicle.location.global_relative_frame.alt)      
        if vehicle.location.global_relative_frame.alt >= aTargetAltitude * 0.95:
            print("Reached target altitude")
            break
        time.sleep(1)

def goto_position_target_relative_ned(velocity_x, velocity_y, velocity_z, vehicle):
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,       
        0, 0,    
        mavutil.mavlink.MAV_FRAME_BODY_OFFSET_NED, 
        0b0000111111000111,  
        0, 0, 0,  
        velocity_x, velocity_y, velocity_z,  
        0, 0, 0,  
        0, 0)
    vehicle.send_mavlink(msg)

def LAND(vehicle):
    vehicle.mode = VehicleMode("LAND")
    print(" Mode: %s" % vehicle.mode.name) 

    while vehicle.mode.name != "LAND":
        time.sleep(1)
        print("Vehicle mode is: %s" % str(vehicle.mode.name))
        vehicle.mode = VehicleMode("LAND")

    print("Vehicle Mode is : LAND")

def condition_yaw(heading, relative, vehicle):
    if relative:
        is_relative = 1 
    else:
        is_relative = 0 
    msg = vehicle.message_factory.command_long_encode(
        0, 0,    
        mavutil.mavlink.MAV_CMD_CONDITION_YAW, 
        0, 
        heading,    
        0,          
        1,          
        is_relative, 
        0, 0, 0)    
    vehicle.send_mavlink(msg)

def get_altitude(vehicle):
    return vehicle.location.global_relative_frame.alt

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Drone connection string')
    parser.add_argument('--connect', type=str, required=True, help='Connection string for drone')
    args = parser.parse_args()

    if args.connect:
        # Connect to drone
        vehicle = connectMyCopter(args.connect)

        # Take off to 10 meters
        arm_and_takeoff(10, vehicle)
        time.sleep(3)

        # Gerçek zamanlı kameradan görüntü alma
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Could not open video.")
            vehicle.close()
            exit()

        stable_condition_met = False
        c_center = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            red_centers, green_centers, blue_centers, processed_frame = colorProcess(frame)
            
            all_rectangles = red_centers + green_centers + blue_centers
            distance_threshold = 400
            
            groups = group_rectangles_by_proximity(all_rectangles, distance_threshold)
            
            for i, group in enumerate(groups):
                if len(group) > 1:  # Only draw groups with more than one rectangle
                    x_min = min(x for x, y, w, h in group)
                    y_min = min(y for x, y, w, h in group)
                    x_max = max(x + w for x, y, w, h in group)
                    y_max = max(y + h for x, y, w, h in group)
                    
                    # Count the number of each color in the group
                    red_count = sum(1 for (x, y, w, h) in group if (x, y, w, h) in red_centers)
                    green_count = sum(1 for (x, y, w, h) in group if (x, y, w, h) in green_centers)
                    blue_count = sum(1 for (x, y, w, h) in group if (x, y, w, h) in blue_centers)
                    
                    # Determine the city type and set the color accordingly
                    if blue_count == 1 and red_count == 3:
                        city_name = "C Sehri"
                        rectangle_color = (0, 0, 255)  # Red for C
                        c_center = (x_min + (x_max - x_min) / 2, y_min + (y_max - y_min) / 2)
                        stable_condition_met = True
                    elif blue_count == 2 and red_count == 1:
                        city_name = "B Sehri"
                        rectangle_color = (255, 0, 0)  # Blue for A and B
                    elif blue_count == 2 and red_count == 2:
                        city_name = "A Sehri"
                        rectangle_color = (255, 0, 0)  # Blue for A and B
                    else:
                        city_name = f"Sehir {i+1}"
                        rectangle_color = (255, 0, 0)  # Default to blue for unclassified cities
                    
                    cv2.rectangle(processed_frame, (x_min, y_min), (x_max, y_max), rectangle_color, 3)
                    cv2.putText(processed_frame, f"{city_name} algilandi", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, rectangle_color, 2)

                    # Check if it is A or B city and wait for 5 seconds
                    if city_name in ["A Sehri", "B Sehri"]:
                        print(f"{city_name} iniş yapılacak şehir değil")
                        time.sleep(5)
            
            if stable_condition_met and c_center is not None:
                print("C Sehri algilandi, inis yapiliyor")
                center_x, center_y = c_center
                frame_center_x = frame.shape[1] / 2
                frame_center_y = frame.shape[0] / 2

                error_x = center_x - frame_center_x
                error_y = center_y - frame_center_y

                if abs(error_x) > 20 or abs(error_y) > 20:
                    if error_x > 20:
                        condition_yaw(vehicle.heading + 1, True, vehicle)
                    elif error_x < -20:
                        condition_yaw(vehicle.heading - 1, True, vehicle)
                    if error_y > 20:
                        goto_position_target_relative_ned(0, 0, 0.1, vehicle)
                    elif error_y < -20:
                        goto_position_target_relative_ned(0, 0, -0.1, vehicle)
                else:
                    # Land if the drone is correctly positioned over the green rectangle
                    LAND(vehicle)
                    break

            cv2.imshow('Processed Frame', processed_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        vehicle.close()
        cv2.destroyAllWindows()
