import random
import time
import keyboard
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Ultrasonic

def process_ultrasonic_data(ultrasonic_data, vehicle):
    vehicle.sensors.poll()
    
    # ✅ Use `.get()` to safely access 'electrics' without KeyError
    #current_gear = vehicle.state.get("electrics", {}).get("gear", "N/A")
    #print(f"Current Gear: {current_gear}")


    current_speed = vehicle.state["vel"][1]
    """ Process Ultrasonic data for close-range detection (bumper-to-bumper traffic). """
    if ultrasonic_data and "distance" in ultrasonic_data:
        distance = ultrasonic_data["distance"]
        if distance < 20 and current_speed != 0:
            breakpressure = distance/current_speed
            vehicle.control(throttle=0, brake=float(breakpressure))
    return None
def main():
    random.seed(1703)

    beamng = BeamNGpy("localhost", 25252, home=r"F:\BeamNG.tech.v0.34.2.0")
    bng = beamng.open()

    scenario = Scenario("west_coast_usa", "sensor_demo", description="ACC with LiDAR + Ultrasonic + Camera Object Detection")
    ego_vehicle = Vehicle("ego_vehicle", model="etk800", license="RED", color="Red")
    scenario.add_vehicle(ego_vehicle, pos=(-717.121, 101, 118.675), rot_quat=(0, 0, 0.3826834, 0.9238795))
    scenario.make(bng)
    bng.settings.set_deterministic(60)
    bng.scenario.load(scenario)
    bng.ui.hide_hud()
    bng.scenario.start()
    ultrasonic = Ultrasonic("ultrasonic1", bng, ego_vehicle, pos=(0, -2.25, 0.5), requested_update_time=0.01, is_snapping_desired=True)

    print("🚗 AI Traffic Started! Press 'Enter' to activate ACC.")
    acc_active = False
    while not acc_active:
        time.sleep(0.001)
        if keyboard.is_pressed("enter"):
            acc_active = True
            print("🚗 ACC Activated!")

    while True:
        time.sleep(0.1)
        ego_vehicle.sensors.poll()
        close_distance = process_ultrasonic_data(ultrasonic.poll(), ego_vehicle)

    bng.close()

if __name__ == "__main__":
    main()
