import sys
import time
import numpy as np
import cv2
import keyboard
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QSpinBox, QFileDialog
)
from PyQt6.QtCore import QThread, pyqtSignal
from beamngpy import BeamNGpy, Scenario, Vehicle
from beamngpy.sensors import Camera
from ultralytics import YOLO

class ACCThread(QThread):
    update_signal = pyqtSignal(float, float)

    def __init__(self, ui):
        super().__init__()
        self.ui = ui
        self.running = False

    def run(self):
        self.running = True
        TARGET_SPEED = self.ui.speed_input.value() / 3.6
        FOLLOW_DISTANCE = self.ui.distance_input.value()
        ego_speed = TARGET_SPEED

        try:
            while self.running:
                if keyboard.is_pressed('w') or keyboard.is_pressed('s'):
                    self.ui.user_override = True
                    print("User override detected!")
                    break
                
                nearest_vehicle_distance = self.ui.process_camera_data()
                speed_error = nearest_vehicle_distance - FOLLOW_DISTANCE
                throttle = 0.5 if nearest_vehicle_distance == float("inf") else max(0, min(speed_error / 50, 1.0))
                brake = 0 if speed_error > 0 else min(abs(speed_error) / 50, 1.0)
                
                self.ui.ego_vehicle.control(throttle=float(throttle), brake=float(brake))
                self.ui.ego_vehicle.ai_set_speed(ego_speed, mode="limit")
                
                self.update_signal.emit(ego_speed * 3.6, throttle)
                time.sleep(0.5)
        except Exception as e:
            print(f"Error in ACC thread: {e}")
            self.running = False

    def stop(self):
        self.running = False

class BeamNGACCUI(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.beamng = None
        self.bng = None
        self.ego_vehicle = None
        self.camera = None
        self.yolo_model = YOLO("yolov8n.pt")
        self.user_override = False
        self.acc_thread = None
    
    def init_ui(self):
        layout = QVBoxLayout()
        
        self.location_label = QLabel("BeamNG Location:")
        self.location_input = QLineEdit()
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_location)
        
        self.speed_label = QLabel("Set Speed (km/h):")
        self.speed_input = QSpinBox()
        self.speed_input.setRange(0, 300)
        
        self.distance_label = QLabel("Set Distance (m):")
        self.distance_input = QSpinBox()
        self.distance_input.setRange(1, 100)
        
        self.current_speed_label = QLabel("Current Speed: 0 km/h")
        self.throttle_label = QLabel("Throttle: 0")
        
        self.start_beamng_button = QPushButton("Start BeamNG")
        self.start_beamng_button.clicked.connect(self.start_beamng)
        
        self.load_scenario_button = QPushButton("Load Scenario")
        self.load_scenario_button.clicked.connect(self.load_scenario)
        
        self.start_acc_button = QPushButton("Start ACC")
        self.start_acc_button.clicked.connect(self.start_acc)
        
        self.stop_button = QPushButton("Stop ACC")
        self.stop_button.clicked.connect(self.stop_acc)
        
        layout.addWidget(self.location_label)
        layout.addWidget(self.location_input)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.speed_label)
        layout.addWidget(self.speed_input)
        layout.addWidget(self.distance_label)
        layout.addWidget(self.distance_input)
        layout.addWidget(self.current_speed_label)
        layout.addWidget(self.throttle_label)
        layout.addWidget(self.start_beamng_button)
        layout.addWidget(self.load_scenario_button)
        layout.addWidget(self.start_acc_button)
        layout.addWidget(self.stop_button)
        
        self.setLayout(layout)
        self.setWindowTitle("BeamNG ACC Controller")
        self.setGeometry(200, 200, 400, 350)
        
    def browse_location(self):
        folder = QFileDialog.getExistingDirectory(self, "Select BeamNG Location")
        if folder:
            self.location_input.setText(folder)
    
    def start_beamng(self):
        beamng_home = self.location_input.text()
        if not beamng_home:
            print("Please select BeamNG location.")
            return
        
        self.beamng = BeamNGpy("localhost", 25252, home=beamng_home)
        self.bng = self.beamng.open()
        print("BeamNG started.")
    
    def load_scenario(self):
        if not self.bng:
            print("Start BeamNG first.")
            return
        
        scenario = Scenario("west_coast_usa", "sensor_fusion", description="ACC using Camera + Depth")
        self.ego_vehicle = Vehicle("ego_vehicle", model="etk800", license="ACC", color="Red")
        scenario.add_vehicle(self.ego_vehicle, pos=(-720, 95, 118.675), rot_quat=(0, 0, 0.3826834, 0.9238795))
        
        scenario.make(self.beamng)
        self.bng.scenario.load(scenario)
        self.bng.scenario.start()
        self.bng.traffic.spawn()
        
        self.camera = Camera("FrontCamera", self.bng, self.ego_vehicle, pos=(0, -1, 1.5),
                             resolution=(640, 480), field_of_view_y=100,
                             is_render_colours=True, is_render_depth=True)
        print("Scenario loaded.")
    
    def start_acc(self):
        if self.acc_thread and self.acc_thread.isRunning():
            print("ACC is already running.")
            return
        
        self.acc_thread = ACCThread(self)
        self.acc_thread.update_signal.connect(self.update_labels)
        self.acc_thread.start()
        print("ACC Started.")

    def start_acc(self):
        if self.acc_thread and self.acc_thread.isRunning():
            print("ACC is already running.")
            return
        
        self.acc_thread = ACCThread(self)
        self.acc_thread.update_signal.connect(self.update_labels)
        self.acc_thread.start()
        print("ACC Started.")

    def update_labels(self, speed, throttle):
        self.current_speed_label.setText(f"Current Speed: {speed:.1f} km/h")
        self.throttle_label.setText(f"Throttle: {throttle:.2f}")
        
    def process_camera_data(self):
        camera_data = self.camera.poll()
        if not camera_data or "colour" not in camera_data or "depth" not in camera_data:
            return float("inf")
        
        img = np.array(camera_data["colour"])[:, :, :3]
        depth_map = np.array(camera_data["depth"]).astype(np.float32)
        
        min_distance = float("inf")
        for r in self.yolo_model(img):
            for box in r.boxes:
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())
                
                if cls in [2, 7] and conf > 0.3:
                    center_x, center_y = (x_min + x_max) // 2, (y_min + y_max) // 2
                    depth_value = depth_map[center_y, center_x]
                    real_distance = float(depth_value) if depth_value > 0 else float("inf")
                    
                    if 240 < center_x < 400 and real_distance < min_distance:
                        min_distance = real_distance
        
        return min_distance
    
    def stop_acc(self):
        if self.acc_thread and self.acc_thread.isRunning():
            self.acc_thread.stop()
            self.acc_thread.wait()
        print("ACC Stopped.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BeamNGACCUI()
    window.show()
    sys.exit(app.exec())
