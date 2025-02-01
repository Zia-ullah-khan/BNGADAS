import gym
import time
import numpy as np
import torch
from stable_baselines3 import PPO
import beamngpy as bng


class BeamNGEnv(gym.Env):
    def __init__(self):
        super(BeamNGEnv, self).__init__()
        
        self.bng = bng.BeamNGpy("localhost", 25252, home=r"F:\BeamNG.tech.v0.34.2.0")
        self.bng.open()
        
        self.scenario = bng.Scenario('west_coast_usa', 'rl_training')
        
        self.vehicle = bng.Vehicle('rl_car', model='etk800', license='RL')
        self.scenario.add_vehicle(self.vehicle, pos=(-717.121, 101, 118.675), rot_quat=(0, 0, 0.3826834, 0.9238795))

        self.scenario.make(self.bng)
        self.bng.load_scenario(self.scenario)
        self.bng.start_scenario()

        self.attach_sensors()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(6,))
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(10,))

    def attach_sensors(self):
        """ Attach 360° LiDAR, Cameras, Radar, and Ultrasonic Sensors to the vehicle """
    
        from beamngpy.sensors import Camera, Lidar, Radar, Ultrasonic

        self.lidar = (
            'lidar_top',
            Lidar(
                name='lidar_top',
                bng=self.bng,
                vehicle=self.vehicle,
                pos=(0, 0, 1.7),
                dir=(0, -1, 0),
                up=(0, 0, 1),
                vertical_resolution=64, 
                vertical_angle=26.9,
                frequency=20,
                horizontal_angle=360, 
                max_distance=500, 
                is_rotate_mode=False,
                is_360_mode=True,
                is_using_shared_memory=True,
                is_visualised=True,
                is_streaming=False,
                is_annotated=False,
                is_static=False
            )
        )

        camera_specs = {
            "front": ((0.5, 1.2, 2.0), (0, -1, 0)),
            "back": ((-0.5, 1.2, -2.0), (0, 1, 0)),
            "left": ((0.5, 1.2, 1.5), (-1, -1, 0)),
            "right": ((0.5, 1.2, 1.5), (1, -1, 0))
        }
    
        for name, (pos, direction) in camera_specs.items():
            self.vehicle.attach_sensor(f'cam_{name}_color', 
                Camera(name=f'cam_{name}_color', bng=self.bng, vehicle=self.vehicle, pos=pos, dir=direction, resolution=(640, 480), 
                    field_of_view_y=110, is_render_colours=True, is_render_annotations=False, is_render_depth=False))
        
            self.vehicle.attach_sensor(f'cam_{name}_depth', 
                Camera(name=f'cam_{name}_depth', bng=self.bng, vehicle=self.vehicle, pos=pos, dir=direction, resolution=(640, 480), 
                   field_of_view_y=110, is_render_colours=False, is_render_annotations=False, is_render_depth=True))
        
            self.vehicle.attach_sensor(f'cam_{name}_annot', 
                Camera(name=f'cam_{name}_annot', bng=self.bng, vehicle=self.vehicle, pos=pos, dir=direction, resolution=(640, 480), 
                   field_of_view_y=110, is_render_colours=False, is_render_annotations=True, is_render_depth=False))

        self.vehicle.attach_sensor('radar_front', 
            Radar(name='radar_front', bng=self.bng, vehicle=self.vehicle, pos=(1.5, 0, 0), dir=(1, 0, 0), field_of_view_y=90))
    
        self.vehicle.attach_sensor('radar_rear', 
            Radar(name='radar_rear', bng=self.bng, vehicle=self.vehicle, pos=(-1.5, 0, 0), dir=(-1, 0, 0), field_of_view_y=90))

        ultrasonic_positions = {
            "front_left": (1.2, -2, -0.2),
            "front_center": (0, 2, 1.2),
            "front_right": (1.2, 2, 0.2),
            "rear_left": (-1.2, -2, -0.2),
            "rear_center": (-1.2, 2, 1.2),
            "rear_right": (-1.2, 2, 0.2),
        }
        for name, pos in ultrasonic_positions.items():
            self.vehicle.attach_sensor(f'ultra_{name}', 
                Ultrasonic(name=f'ultra_{name}', bng=self.bng, vehicle=self.vehicle, pos=pos))



    def get_state(self):
        """ Get sensor data from BeamNG """
        sensor_data = self.vehicle.sensors.poll()

        speed = self.vehicle.state['vel'][0]
        lane_distance = np.abs(self.vehicle.state['pos'][1])

        radar_front = sensor_data['radar_front']['distance']
        radar_rear = sensor_data['radar_rear']['distance']

        ultra_front = min(sensor_data[f'ultra_front_{pos}']['distance'] for pos in ['left', 'center', 'right'])
        ultra_rear = min(sensor_data[f'ultra_rear_{pos}']['distance'] for pos in ['left', 'center', 'right'])

        lidar_data = np.mean(sensor_data['lidar_top']['points'])

        return np.array([speed, lane_distance, radar_front, radar_rear, ultra_front, ultra_rear, lidar_data])

    def step(self, action):
        """ Apply AI action to BeamNG """
        throttle, steering, brake, gear, lights, indicator = action
        
        throttle = max(0, min(1, throttle))
        steering = max(-1, min(1, steering))
        brake = max(0, min(1, brake))

        self.vehicle.control(
            throttle=throttle, 
            steering=steering, 
            brake=brake,
            gear=int(gear * 6),
            lights=bool(lights > 0.5),
            turn_signal=int(indicator * 2)  
        )

        state = self.get_state()

        reward = 10 - (state[1] * 5)  
        if self.vehicle.state['damage'] > 0:
            reward -= 50  
        if lights > 0.5:
            reward += 5  
        if indicator > 0.5:
            reward += 5  
        if state[2] < 10:  
            reward -= 10  
        if state[3] < 10:  
            reward -= 10  

        done = self.vehicle.state['damage'] > 0

        return state, reward, done, {}

    def reset(self):
        """ Restart the scenario for a new episode """
        self.bng.restart_scenario()
        return self.get_state()


env = BeamNGEnv()
model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=100000)

model.save("beamng_self_driving")
