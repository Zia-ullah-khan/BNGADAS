import sys
import numpy as np
import cv2
import json
import socket
import threading
from time import sleep

from beamngpy import BeamNGpy, Scenario, Vehicle, angle_to_quat, set_up_simple_logging
from beamngpy.sensors import Camera
from beamngpy.api.beamng import CameraApi


class VisualizationServer:
    def __init__(self, port=9999):
        self.port = port
        self.server_socket = None
        self.running = False
        self.car_color = (255, 0, 0)
        self.lane_color = (255, 255, 255)
        self.radar_distance = None
        
    def start_server(self):
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind(('localhost', self.port))
        self.server_socket.listen(1)
        print(f"Visualization server listening on port {self.port}")
        
        self.running = True
        while self.running:
            try:
                client_socket, addr = self.server_socket.accept()
                data = client_socket.recv(1024).decode('utf-8')
                if data:
                    msg = json.loads(data)
                    if msg['type'] == 'radar_update':
                        self.radar_distance = msg['distance']
                    elif msg['type'] == 'colors':
                        self.car_color = tuple(msg['car_color'])
                        self.lane_color = tuple(msg['lane_color'])
                client_socket.close()
            except:
                break
                
    def stop_server(self):
        self.running = False
        if self.server_socket:
            self.server_socket.close()
            
    def process_camera_data(self, camera_data):
        if (
            not camera_data
            or "annotation" not in camera_data
            or "depth" not in camera_data
            or "colour" not in camera_data
            or camera_data["annotation"] is None
            or camera_data["depth"] is None
            or camera_data["colour"] is None
        ):
            return None

        try:
            annotation_pil = camera_data["annotation"]
            annotation_img = np.array(annotation_pil)[:, :, :3]
            color_pil = camera_data["colour"]
            color_img = np.array(color_pil)[:, :, :3]
            depth_pil = camera_data["depth"]
            depth_map = np.array(depth_pil).astype(np.float32)

            # Car detection
            car_mask = np.all(annotation_img == self.car_color, axis=-1)
            min_distance = float("inf")

            if np.any(car_mask):
                car_depths = depth_map[car_mask]
                min_distance = np.min(car_depths)
                overlay = color_img.copy()
                overlay[car_mask] = (0, 255, 0)
                color_img = cv2.addWeighted(overlay, 0.4, color_img, 0.6, 0)

                ys, xs = np.where(car_mask)
                if len(xs) > 0 and len(ys) > 0:
                    cx, cy = int(np.mean(xs)), int(np.mean(ys))
                    radar_str = f", Radar: {self.radar_distance:.2f}m" if self.radar_distance is not None else ""
                    cv2.putText(
                        color_img,
                        f"Car: {min_distance:.2f}m{radar_str}",
                        (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            # Lane detection
            lane_mask = np.all(annotation_img == self.lane_color, axis=-1)
            if np.any(lane_mask):
                lane_overlay = color_img.copy()
                lane_overlay[lane_mask] = (255, 255, 0)
                color_img = cv2.addWeighted(lane_overlay, 0.3, color_img, 0.7, 0)
                
                ys, xs = np.where(lane_mask)
                if len(xs) > 0:
                    lane_center_px = int(np.mean(xs))
                    cv2.line(color_img, (lane_center_px, 0), (lane_center_px, color_img.shape[0]), (0, 255, 255), 2)
                    
                    image_center = color_img.shape[1] // 2
                    offset_px = lane_center_px - image_center
                    cv2.putText(
                        color_img,
                        f"Lane Offset: {offset_px}px",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

            # Vehicle center line
            image_center = color_img.shape[1] // 2
            cv2.line(color_img, (image_center, 0), (image_center, color_img.shape[0]), (0, 0, 255), 2)

            cv2.imshow("ADAS Visualization", color_img)
            cv2.waitKey(1)
            return color_img

        except Exception as e:
            print(f"Visualization error: {e}")
            return None


def main():
    set_up_simple_logging()
    beamng_home = r"F:\BeamNG.tech.v0.35.5.0"
    
    # Start visualization server
    viz_server = VisualizationServer()
    server_thread = threading.Thread(target=viz_server.start_server, daemon=True)
    server_thread.start()
    
    try:
        bng = BeamNGpy("localhost", 25252, home=beamng_home)
        bng.open()
        
        # Connect to existing scenario
        print("Connecting to existing BeamNG scenario...")
        sleep(2)
        
        # Get vehicle reference (assuming it exists)
        vehicle1 = Vehicle("ego_vehicle", model="etk800")
        
        # Create camera
        camera = Camera("VisualizationCamera", bng, vehicle1, pos=(0, -1, 1.5), 
                       resolution=(640, 480), field_of_view_y=100,
                       is_render_colours=True, is_render_annotations=True, is_render_depth=True)
        
        print("Visualization running. Press 'q' to quit.")
        
        while True:
            camera_data = camera.poll()
            viz_server.process_camera_data(camera_data)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
            sleep(0.05)
                
    except Exception as e:
        print(f"Visualization error: {e}")
    finally:
        viz_server.stop_server()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
