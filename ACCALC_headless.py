import sys
import numpy as np
import msvcrt
import json
import socket
from time import sleep

from beamngpy import BeamNGpy, Scenario, Vehicle, angle_to_quat, set_up_simple_logging
from beamngpy.sensors import IdealRadar, AdvancedIMU
from beamngpy.api.beamng import CameraApi


def send_to_visualizer(data, port=9999):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.connect(('localhost', port))
        client_socket.send(json.dumps(data).encode('utf-8'))
        client_socket.close()
    except:
        pass  # Visualization server might not be running


def get_car_annotation_color(bng):
    cam_api = CameraApi(bng)
    annotations = cam_api.get_annotations()
    for class_name, color in annotations.items():
        if "car" in class_name.lower() or "vehicle" in class_name.lower() or "etk" in class_name.lower():
            return tuple(color)
    return (255, 0, 0)


def get_lane_annotation_color(bng):
    cam_api = CameraApi(bng)
    annotations = cam_api.get_annotations()
    for class_name, color in annotations.items():
        if any(keyword in class_name.lower() for keyword in ["lane", "roadline", "road", "marking", "line"]):
            return tuple(color)
    return (255, 255, 255)


def smooth_acc_control(vehicle, target_distance, radar_distance, imu_speed, min_distance=10.0, max_distance=50.0, max_speed=20.0):
    throttle = 0.0
    brake = 0.0
    if radar_distance is None or radar_distance == float("inf"):
        if imu_speed < max_speed:
            throttle = min(1.0, (max_speed - imu_speed) / max_speed)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(1.0, (imu_speed - max_speed) / max_speed)
    else:
        dist_error = radar_distance - target_distance
        throttle = np.clip(0.5 + 0.03 * dist_error, 0.0, 1.0)
        brake = np.clip(-0.03 * dist_error, 0.0, 1.0)
        if imu_speed > max_speed:
            throttle = 0.0
            brake = min(1.0, (imu_speed - max_speed) / max_speed)
    return throttle, brake


def main():
    set_up_simple_logging()
    beamng_home = r"F:\BeamNG.tech.v0.35.5.0"
    bng = BeamNGpy("localhost", 25252, home=beamng_home)
    bng.open()

    vehicle1 = Vehicle("ego_vehicle", model="etk800", licence="EGO", color="Red")
    vehicle2 = Vehicle("other_vehicle", model="etk800", licence="OTHER", color="Blue")

    scenario = Scenario("italy", "ACC_ALC_Test", description="ACC with ALC Demo")
    scenario.add_vehicle(vehicle1, pos=(-350.76, 1169.09, 168.69), rot_quat=angle_to_quat((0, 0, 90)))
    scenario.add_vehicle(vehicle2, pos=(-365.56, 1169.09, 168.69), rot_quat=angle_to_quat((0, 0, 90)))
    scenario.make(bng)

    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    bng.ui.hide_hud()
    bng.scenario.start()

    radar = IdealRadar("idealRADAR1", bng, vehicle1, physics_update_time=0.01)
    imu = AdvancedIMU("imu1", bng, vehicle1, physics_update_time=0.01, pos=(0, 0, 1.7), dir=(0, -1, 0), up=(0, 0, 1))

    print("Setting AI Traffic...")
    vehicle2.ai.set_mode("traffic")
    vehicle1.ai.set_mode("manual")

    sleep(5)
    print("Press ENTER to activate ACC + ALC...")
    input()

    car_color = get_car_annotation_color(bng)
    lane_color = get_lane_annotation_color(bng)
    
    # Send colors to visualization server
    send_to_visualizer({
        'type': 'colors',
        'car_color': car_color,
        'lane_color': lane_color
    })

    target_distance = 25.0
    print("ACC + ALC activated (Headless). Press BACKSPACE to exit.")
    print("Run 'python visualization_server.py' in another terminal for visualization.")

    while True:
        radar_data = radar.poll()
        radar_distance = None
        if radar_data and isinstance(radar_data, list) and radar_data:
            first_vehicle = radar_data[0].get("closestVehicles1", {})
            if first_vehicle and "relDistX" in first_vehicle:
                radar_distance = first_vehicle["relDistX"]

        # Send radar data to visualization
        send_to_visualizer({
            'type': 'radar_update',
            'distance': radar_distance
        })

        imu_data = imu.poll()
        imu_speed = 0.0
        if imu_data and "vel" in imu_data:
            vel = imu_data["vel"]
            imu_speed = np.linalg.norm([vel["x"], vel["y"], vel["z"]])

        throttle, brake = smooth_acc_control(vehicle1, target_distance, radar_distance, imu_speed)
        vehicle1.control(throttle=throttle, brake=brake, steering=0.0)

        sleep(0.05)
        if msvcrt.kbhit() and msvcrt.getch() == b'\x08':
            print("Backspace pressed. Exiting ACC + ALC.")
            break

    bng.ui.show_hud()
    bng.disconnect()


if __name__ == "__main__":
    main()
