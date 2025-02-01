import time
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Electrics, Radar

# Configure connection parameters (adjust as needed for your system)
beamng = BeamNGpy('localhost', 25252, home='F:\BeamNG.tech.v0.34.2.0')  # Update the BeamNG.tech path
bng = beamng.open()
# Create a scenario on the "smallgrid" map
scenario = Scenario('smallgrid', 'acc_demo')

# Add a vehicle (using, for example, the etk800 model)
vehicle = Vehicle('ego_vehicle', model='etk800', licence='ACC')
# Place the vehicle at a starting position and orientation
scenario.add_vehicle(vehicle, pos=(0, 0, 0), rot_quat=(0, 0, 0))
# Attach an electrics sensor to get speed information
vehicle.attach_sensor('electrics', Electrics())

# Attach a radar sensor to detect objects in front of the vehicle.
# Adjust the position, rotation, field-of-view (fov), and max_distance as needed.
radar_sensor = Radar("radar",beamng,  pos=(0, 2, 1.2), field_of_view_y=90, is_visualised=True)
vehicle.attach_sensor('radar', radar_sensor)
# Build (or "make") the scenario and open a BeamNG connection.
scenario.make(beamng)
b = beamng.open(launch=True)
b.load_scenario(scenario)
b.start_scenario()
b.hide_hud()  # Optional: hide the HUD for a cleaner view
# ACC configuration parameters
target_speed = 40      # Target speed in km/h
safe_distance = 15     # Minimum safe distance to an object in meters

while True:
    # Poll sensor data from the vehicle
    sensors = vehicle.poll_sensors()
    
    # Retrieve current speed (usually provided by the 'electrics' sensor)
    current_speed = sensors.get('electrics', {}).get('wheelspeed', 0)
    
    # Get radar detections (the data structure can vary; here we assume it returns a list)
    radar_data = sensors.get('radar', [])
    distance_ahead = None
    if radar_data:
        # For simplicity, assume each detection is a dictionary with a 'distance' key.
        # We compute the minimum (closest) detected distance.
        distances = [d.get('distance') for d in radar_data if d.get('distance') is not None]
        if distances:
            distance_ahead = min(distances)
    
    # Initialize control values
    throttle = 0.0
    brake = 0.0

    # ACC Logic:
    # - If an object is detected and is closer than the safe distance, decelerate.
    # - Otherwise, if below the target speed, accelerate.
    if distance_ahead is not None and distance_ahead < safe_distance:
        throttle = 0.0
        brake = 0.7  # Apply moderate braking if too close
    else:
        if current_speed < target_speed:
            throttle = 0.8  # Apply throttle to reach target speed
            brake = 0.0
        else:
            # Optionally, you could maintain speed by applying minimal throttle or coasting.
            throttle = 0.2
            brake = 0.0

    # Send control commands to the vehicle.
    # (steering is set to zero here for straight-line driving)
    vehicle.control(throttle=throttle, brake=brake, steering=0)

    # Sleep briefly to maintain a control loop frequency (e.g., 10 Hz)
    time.sleep(0.1)
