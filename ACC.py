from time import sleep
import sys
import numpy as np
import cv2
import msvcrt

from beamngpy import BeamNGpy, Scenario, Vehicle, angle_to_quat, set_up_simple_logging
from beamngpy.sensors import Camera, IdealRadar, AdvancedIMU
from beamngpy.api.beamng import CameraApi

def get_car_annotation_color(bng):
    cam_api = CameraApi(bng)
    annotations = cam_api.get_annotations()
    for class_name, color in annotations.items():
        if "car" in class_name.lower():
            return tuple(color)
    return None

def process_camera_data(camera_data, car_color, radar_distance=None):
    if (
        not camera_data
        or "annotation" not in camera_data
        or "depth" not in camera_data
        or "colour" not in camera_data
        or camera_data["annotation"] is None
        or camera_data["depth"] is None
        or camera_data["colour"] is None
    ):
        print("Warning: Camera, annotation, or depth data not available yet.")
        return None, float("inf")

    try:
        annotation_pil = camera_data["annotation"]
        annotation_img = np.array(annotation_pil)[:, :, :3]
        color_pil = camera_data["colour"]
        color_img = np.array(color_pil)[:, :, :3]
        depth_pil = camera_data["depth"]
        depth_map = np.array(depth_pil).astype(np.float32)

        mask = np.all(annotation_img == car_color, axis=-1)
        min_distance = float("inf")
        label = None

        if np.any(mask):
            car_depths = depth_map[mask]
            min_distance = np.min(car_depths)
            label = "Car"
            overlay = color_img.copy()
            overlay[mask] = (0, 255, 0)
            alpha = 0.4
            blended = cv2.addWeighted(overlay, alpha, color_img, 1 - alpha, 0)
            color_img = blended

            ys, xs = np.where(mask)
            if len(xs) > 0 and len(ys) > 0:
                cx, cy = int(np.mean(xs)), int(np.mean(ys))
                radar_str = f", Radar: {radar_distance:.2f}m" if radar_distance is not None else ""
                cv2.putText(
                    color_img,
                    f"{label}: {min_distance:.2f}m{radar_str}",
                    (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
        try:
            cv2.imshow("BeamNG Camera Overlay", color_img)
        except Exception as e:
            print(f"OpenCV imshow error: {e}")
        sleep(0.001)
        return color_img, min_distance if label else float("inf")
    except Exception as e:
        print(f"Camera processing error: {e}")
        return None, float("inf")
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
    vehicle.control(throttle=throttle, brake=brake)
    return throttle, brake

def main():
    set_up_simple_logging()

    beamng_home = r"F:\BeamNG.tech.v0.35.5.0"
    beamng_user = r"F:\BeamNG_user"

    if " " in beamng_home or " " in beamng_user:
        print(
            "ERROR: Your BeamNG.tech home or user path contains a space. "
            "This will cause connection issues. "
            "Please move BeamNG.tech and user folder to paths without spaces, "
            "or set the userpath manually in startup.ini as a workaround."
        )
        sys.exit(1)

    bng = BeamNGpy("localhost", 25252, home=beamng_home)
    exe_path = rf"{beamng_home}\BeamNG.tech.x64.exe"
    bng.open()

    vehicle1 = Vehicle("ego_vehicle", model="etk800", licence="EGO", color="Red")
    vehicle2 = Vehicle("other_vehicle", model="etk800", licence="OTHER", color="Blue")

    scenario = Scenario("italy", "ACCtest", description="ACC1")

    scenario.add_vehicle(
        vehicle1,
        pos=(-350.76326635601, 1169.0963008935, 168.6981158547),
        rot_quat=angle_to_quat((0, 0, 90)),
    )
    scenario.add_vehicle(
        vehicle2,
        pos=(-365.56326635601, 1169.0963008935, 168.69811585470),
        rot_quat=angle_to_quat((0, 0, 90)),
    )
    scenario.make(bng)

    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    bng.ui.hide_hud()
    bng.scenario.start()

    camera = Camera("FrontCamera", bng, vehicle1, pos=(0, -1, 1.5),
                    resolution=(640, 480), field_of_view_y=100,
                    is_render_colours=True, is_render_annotations=True, is_render_depth=True)

    idealRADAR1 = IdealRadar(
        "idealRADAR1",
        bng,
        vehicle1,
        is_send_immediately=False,
        physics_update_time=0.01,
    )

    imu = AdvancedIMU(
        "imu1",
        bng,
        vehicle1,
        physics_update_time=0.01,
        pos=(0, 0, 1.7),
        dir=(0, -1, 0),
        up=(0, 0, 1),
        is_send_immediately=False,
        is_using_gravity=False,
        is_visualised=False,
    )

    vehicle1.sensors.poll()
    vehicle1.sensors["state"]
    print("Collecting ideal RADAR readings...")
    vehicle2.ai.set_mode("traffic")
    vehicle1.ai.set_mode("manual")
    sleep(5)
    print("Manual driving: Control the car using your input device.")
    print("Press ENTER to enable Adaptive Cruise Control (ACC)...")
    input()

    print("ACC enabled. Software will now control throttle and brake.")
    car_color = get_car_annotation_color(bng)
    if car_color is None:
        print("Could not find car annotation color!")
        sys.exit(1)

    sleep(1.0)
    target_distance = 25.0

    (
        listrelDistX,
        listrelDistY,
        listrelVelX,
        listrelVelY,
        listrelAccX,
        listrelAccY,
        listtime,
        vel_ego,
        vel_egox,
        vel_egoy,
        time_ego,
    ) = ([], [], [], [], [], [], [], [], [], [], [])
    (
        pos_ego,
        posx_ego,
        posy_ego,
        pos_veh2,
        posx_veh2,
        posy_veh2,
        vel_veh2,
        velx_veh2,
        vely_veh2,
    ) = ([], [], [], [], [], [], [], [], [])

    print("Press BACKSPACE to stop ACC and exit.")
    while True:
        data_all = idealRADAR1.poll()
        if not data_all:
            print("Warning: No radar data received.")
            radar_distance = None
            data1stVehicle = []
        else:
            if isinstance(data_all, list) and len(data_all) > 0:
                latest_reading = data_all[0]
            elif isinstance(data_all, dict):
                latest_reading = data_all
            else:
                latest_reading = None

            if latest_reading and "closestVehicles1" in latest_reading:
                data1stVehicle = latest_reading["closestVehicles1"]
                if data1stVehicle != []:
                    raw_dist = data1stVehicle["relDistX"]
                    radar_distance = raw_dist if raw_dist > 1e-2 else float("inf")
                else:
                    radar_distance = None
            else:
                data1stVehicle = []
                radar_distance = None

        camera_data = camera.poll()
        _, nearest_vehicle_distance_camera = process_camera_data(camera_data, car_color, radar_distance)

        imu_data = imu.poll()
        imu_speed = 0.0
        if imu_data and "vel" in imu_data:
            vel = imu_data["vel"]
            imu_speed = np.linalg.norm([vel["x"], vel["y"], vel["z"]])

        smooth_acc_control(vehicle1, target_distance, radar_distance, imu_speed, min_distance=10.0, max_distance=50.0, max_speed=20.0)

        data_all = idealRADAR1.poll()
        if not data_all:
            print("Warning: No radar data received for logging.")
            sleep(0.1)
            if msvcrt.kbhit() and msvcrt.getch() == b'\x08':
                print("Backspace pressed. Exiting ACC loop.")
                break
            continue
        if isinstance(data_all, list) and data_all:
            latest_reading = data_all[0]
        else:
            latest_reading = data_all
        data1stVehicle = latest_reading["closestVehicles1"]
        vehicle1.sensors.poll()
        state_ego = vehicle1.sensors["state"]
        vel_ego = state_ego["vel"]
        pos_ego = state_ego["pos"]
        time_ego.append(state_ego["time"])
        print("sensors poll...")
        if data1stVehicle != []:
            relDistX = data1stVehicle["relDistX"]
            listrelDistX.append(relDistX if relDistX > 1e-2 else float("inf"))
            listrelDistY.append(data1stVehicle["relDistY"])
            listrelVelX.append(data1stVehicle["relVelX"])
            listrelVelY.append(data1stVehicle["relVelY"])
            listrelAccX.append(data1stVehicle["relAccX"])
            listrelAccY.append(data1stVehicle["relAccY"])

            vel_veh2 = data1stVehicle["velBB"]
            pos_veh2 = data1stVehicle["positionB"]
            listtime.append(latest_reading["time"])
            vel_egox.append(vel_ego[0])
            vel_egoy.append(vel_ego[1])
            posx_ego.append(pos_ego[0])
            posy_ego.append(pos_ego[1])

            px = pos_veh2["x"]
            posx_veh2.append(px)
            py = pos_veh2["y"]
            posy_veh2.append(py)

            velx_veh2.append(vel_veh2["x"])
            vely_veh2.append(vel_veh2["y"])
        sleep(1.0)
        if msvcrt.kbhit() and msvcrt.getch() == b'\x08':
            print("Backspace pressed. Exiting ACC loop.")
            break

    bng.ui.show_hud()
    bng.disconnect()
if __name__ == "__main__":
    main()