import time
import numpy as np
import cv2
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
from ultralytics import YOLO

beamng = BeamNGpy("localhost", 25252, home=r"C:\Project\BeamNG.tech.v0.34.2.0")
bng = beamng.open()

scenario = Scenario("west_coast_usa", "sensor_fusion", description="ACC using Camera + Depth")
ego_vehicle = Vehicle("ego_vehicle", model="etk800", license="ACC", color="Red")

scenario.add_vehicle(ego_vehicle, pos=(-720, 95, 118.675), rot_quat=(0, 0, 0.3826834, 0.9238795))

scenario.make(beamng)
bng.scenario.load(scenario)
bng.scenario.start()
bng.traffic.spawn()

camera = Camera("FrontCamera", bng, ego_vehicle, pos=(0, -1, 1.5),
                resolution=(640, 480), field_of_view_y=100,
                is_render_colours=True, is_render_depth=True)

#ego_vehicle.attach_sensor("FrontCamera", camera)

yolo_model = YOLO("yolov8n.pt")

TARGET_SPEED = 50 / 3.6
FOLLOW_DISTANCE = 70
ACCELERATION_FACTOR = 0.1
BRAKE_FACTOR = 0.1

time.sleep(2)

def process_camera_data(camera_data):
    """Detect vehicles and estimates their distance using the depth camera."""
    if not camera_data or "colour" not in camera_data or "depth" not in camera_data:
        print("Warning: Camera or depth data not available yet.")
        return None, float("inf")

    try:
        img_pil = camera_data["colour"]
        img = np.array(img_pil)[:, :, :3]

        depth_pil = camera_data["depth"]
        depth_map = np.array(depth_pil).astype(np.float32)

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        min_distance = float("inf")
        car_in_front = False
        for r in yolo_model(img):
            for box in r.boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())

                if cls in [2, 7] and conf > 0.3:
                    label = f"Car {conf:.2f}" if cls == 2 else f"Truck {conf:.2f}"
                    cv2.rectangle(img_bgr, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
                    depth_value = depth_map[center_y, center_x]

                    real_distance = float(depth_value) if depth_value > 0 else float("inf")

                    if 240 < center_x < 400:
                        car_in_front = True
                        if real_distance < min_distance:
                            min_distance = real_distance

                    cv2.putText(img_bgr, f"{real_distance:.2f}m", (x_min, y_max + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("BeamNG Camera Feed", img_bgr)
        cv2.waitKey(1)

        return img_bgr, min_distance if car_in_front else float("inf")

    except Exception as e:
        print(f"Camera processing error: {e}")
        return None, float("inf")

def adaptive_cruise_control(current_speed, target_distance):
    """Smoothly adjust vehicle speed using throttle and brake controls, only if a car is in front and distance is lower than targer distance."""
    speed_error = target_distance - FOLLOW_DISTANCE

    if target_distance == float("inf"):
        new_speed = TARGET_SPEED
        throttle = 0.5
        brake = 0
    elif speed_error < 0:
        new_speed = max(current_speed - BRAKE_FACTOR, 0)
        brake = float(min(abs(speed_error) / 50, 1.0)) 
        throttle = 0
    elif speed_error > 10:
        new_speed = min(current_speed + ACCELERATION_FACTOR, TARGET_SPEED)
        throttle = float(min(speed_error / 50, 1.0))
        brake = 0
    else:
        new_speed = current_speed
        throttle = 0.3
        brake = 0

    ego_vehicle.control(throttle=float(throttle), brake=float(brake))
    return new_speed

input("Press ENTER to start Adaptive Cruise Control (ACC)...")

try:
    ego_speed = TARGET_SPEED
    while True:
        camera_data = camera.poll()
        _, nearest_vehicle_distance = process_camera_data(camera_data)

        new_speed = adaptive_cruise_control(ego_speed, nearest_vehicle_distance)
        ego_vehicle.ai_set_speed(new_speed, mode="limit")
        ego_speed = new_speed

        print(f"Nearest vehicle: {nearest_vehicle_distance:.2f}m, Adjusted Speed: {ego_speed * 3.6:.2f} km/h")

        time.sleep(0.5)

finally:
    bng.close()
    cv2.destroyAllWindows()
