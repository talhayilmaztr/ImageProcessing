
import cv2
import numpy as np
import time
from dronekit import connect, VehicleMode
from pymavlink import mavutil
import argparse

def apply_morphology(mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    return mask

def colorProcess(frame):
    red_houses = 0
    green_houses = 0
    blue_houses = 0
    
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
    green_center = None

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
                    if color_name == "kirmizi":
                        red_houses += 1
                    elif color_name == "yesil":
                        green_houses += 1
                        green_center = (x + w / 2, y + h / 2)
                    elif color_name == "mavi":
                        blue_houses += 1

    return red_houses, green_houses, blue_houses, green_center, frame

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

        start_time = None
        condition_met = False
        stable_condition_met = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            red_houses, green_houses, blue_houses, green_center, processed_frame = colorProcess(frame)
            
            cv2.putText(processed_frame, f"Kirmizi Kare Sayisi: {red_houses}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            cv2.putText(processed_frame, f"Yeşil Kare Sayisi: {green_houses}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.putText(processed_frame, f"Mavi Kare Sayisi: {blue_houses}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            if green_center:
                cv2.putText(processed_frame, f"kordinat: {green_center}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (25, 0, 0), 3)

            if red_houses == 2 and green_houses == 2 and blue_houses == 1:
                cv2.putText(processed_frame, "C Sehri", (1500, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 255, 255), 10)
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= 2:
                    condition_met = True
                    stable_condition_met = True
            else:
                start_time = None
                condition_met = False
            
            if stable_condition_met:
                cv2.putText(processed_frame, "C Sehri Algilandi, Inis Yapiliyor", (1000, 2000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 10)
                if green_center is not None:
                    center_x, center_y = green_center
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
