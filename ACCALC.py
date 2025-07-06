import sys
import numpy as np
import cv2
import msvcrt
from time import sleep

from beamngpy import BeamNGpy, Scenario, Vehicle, angle_to_quat, set_up_simple_logging
from beamngpy.sensors import Camera, IdealRadar, AdvancedIMU
from beamngpy.api.beamng import CameraApi


def get_car_annotation_color(bng):
    cam_api = CameraApi(bng)
    annotations = cam_api.get_annotations()
    print(f"Available annotations: {list(annotations.keys())}")
    for class_name, color in annotations.items():
        if "car" in class_name.lower() or "vehicle" in class_name.lower() or "etk" in class_name.lower():
            print(f"Found car annotation: {class_name} -> {color}")
            return tuple(color)
    # Fallback: return the first annotation that might be a vehicle
    for class_name, color in annotations.items():
        if any(keyword in class_name.lower() for keyword in ["auto", "truck", "sedan", "suv"]):
            print(f"Found vehicle annotation (fallback): {class_name} -> {color}")
            return tuple(color)
    return None


def get_lane_annotation_color(bng):
    cam_api = CameraApi(bng)
    annotations = cam_api.get_annotations()
    lane_colors = []
    
    # Look for different types of lane markings
    for class_name, color in annotations.items():
        if any(keyword in class_name.lower() for keyword in ["dashed_line", "solid_line", "lane", "roadline", "marking", "line"]):
            print(f"Found lane annotation: {class_name} -> {color}")
            lane_colors.append((class_name, tuple(color)))
    
    # Fallback: try to find any road-related annotation
    if not lane_colors:
        for class_name, color in annotations.items():
            if any(keyword in class_name.lower() for keyword in ["street", "pavement", "asphalt"]):
                print(f"Found road annotation (fallback): {class_name} -> {color}")
                lane_colors.append((class_name, tuple(color)))
    
    return lane_colors if lane_colors else [("DEFAULT", (255, 255, 255))]


def process_camera_data(camera_data, car_color, lane_colors, radar_distance=None, curvature=0.0, autonomous_status="AUTO"):
    if (
        not camera_data
        or "annotation" not in camera_data
        or "depth" not in camera_data
        or "colour" not in camera_data
        or camera_data["annotation"] is None
        or camera_data["depth"] is None
        or camera_data["colour"] is None
    ):
        return None, float("inf")

    try:
        annotation_pil = camera_data["annotation"]
        annotation_img = np.array(annotation_pil)[:, :, :3]
        color_pil = camera_data["colour"]
        color_img = np.array(color_pil)[:, :, :3]
        depth_pil = camera_data["depth"]
        depth_map = np.array(depth_pil).astype(np.float32)

        # Car detection
        car_mask = np.all(annotation_img == car_color, axis=-1)
        min_distance = float("inf")
        label = None

        if np.any(car_mask):
            car_depths = depth_map[car_mask]
            min_distance = np.min(car_depths)
            label = "Car"
            overlay = color_img.copy()
            overlay[car_mask] = (0, 255, 0)
            blended = cv2.addWeighted(overlay, 0.4, color_img, 0.6, 0)
            color_img = blended

            ys, xs = np.where(car_mask)
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

        # Lane detection and visualization - improved for multiple lane types
        lane_center_px = None
        for lane_name, lane_color in lane_colors:
            lane_mask = np.all(annotation_img == lane_color, axis=-1)
            if np.any(lane_mask):
                # Highlight lane markings
                lane_overlay = color_img.copy()
                lane_overlay[lane_mask] = (255, 255, 0)  # Cyan for lane markings
                color_img = cv2.addWeighted(lane_overlay, 0.3, color_img, 0.7, 0)
                
                # Calculate lane center from this type of marking
                ys, xs = np.where(lane_mask)
                if len(xs) > 0:
                    current_lane_center = int(np.mean(xs))
                    if lane_center_px is None:
                        lane_center_px = current_lane_center
                    else:
                        # Average multiple lane marking centers
                        lane_center_px = (lane_center_px + current_lane_center) // 2
        
        # Draw lane center line and offset info
        if lane_center_px is not None:
            cv2.line(color_img, (lane_center_px, 0), (lane_center_px, color_img.shape[0]), (0, 255, 255), 2)
            
            # Show lane offset
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

        # Draw vehicle center line
        image_center = color_img.shape[1] // 2
        cv2.line(color_img, (image_center, 0), (image_center, color_img.shape[0]), (0, 0, 255), 2)
        
        # Draw predicted curved path based on curvature
        draw_predicted_path(color_img, curvature, image_center)
        
        # Add autonomous driving status indicator
        height, width = color_img.shape[:2]
        status_text = f"[{autonomous_status}] Autonomous Driving Active"
        status_color = (0, 255, 0) if autonomous_status == "AUTO" else (0, 255, 255)  # Green for auto, cyan for startup
        text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(color_img, (width - text_size[0] - 15, 5), (width - 5, 35), (0, 0, 0), -1)
        cv2.putText(
            color_img,
            status_text,
            (width - text_size[0] - 10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("BeamNG Camera Overlay", color_img)
        cv2.waitKey(1)
        return color_img, min_distance if label else float("inf")

    except Exception as e:
        print(f"Camera processing error: {e}")
        return None, float("inf")


def estimate_lane_center_offset(camera_data, lane_colors, image_width):
    if not camera_data or "annotation" not in camera_data or camera_data["annotation"] is None:
        return 0.0

    annotation_pil = camera_data["annotation"]
    annotation_img = np.array(annotation_pil)[:, :, :3]
    
    # More robust lane detection - look for lane markings in bottom half of image
    height = annotation_img.shape[0]
    bottom_half = annotation_img[height//2:, :]  # Focus on road closer to vehicle
    
    # Collect all lane pixels
    all_lane_xs = []
    
    for lane_name, lane_color in lane_colors:
        mask = np.all(bottom_half == lane_color, axis=-1)
        ys, xs = np.where(mask)
        
        if len(xs) > 10:  # Need minimum number of pixels for reliable detection
            all_lane_xs.extend(xs)
    
    if len(all_lane_xs) < 20:  # Not enough lane markings
        return 0.0
    
    # Use histogram to find lane center more reliably
    hist, bin_edges = np.histogram(all_lane_xs, bins=32, range=(0, image_width))
    
    # Find peaks in the histogram (these are likely lane markings)
    peaks = []
    for i in range(1, len(hist)-1):
        if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 5:
            peak_x = (bin_edges[i] + bin_edges[i+1]) / 2
            peaks.append(peak_x)
    
    if len(peaks) == 0:
        # Fallback to simple average
        lane_center_px = np.mean(all_lane_xs)
    elif len(peaks) == 1:
        # Single peak - might be center line or edge
        lane_center_px = peaks[0]
    else:
        # Multiple peaks - assume outer ones are lane boundaries
        peaks.sort()
        if len(peaks) >= 2:
            lane_center_px = (peaks[0] + peaks[-1]) / 2.0
        else:
            lane_center_px = np.mean(peaks)
    
    # Calculate normalized offset
    image_center_px = image_width / 2.0
    pixel_offset = lane_center_px - image_center_px
    normalized_offset = pixel_offset / (image_width / 2.0)
    
    # Clamp to reasonable range
    normalized_offset = np.clip(normalized_offset, -0.8, 0.8)
    
    return normalized_offset


def steering_control(center_offset, previous_offset=0.0, curvature=0.0, k_p=3.0, k_d=1.0, k_c=1.8, max_steering=0.6):
    """
    Enhanced PD steering control with curvature prediction for smooth lane centering
    k_c: Curvature gain for predictive steering (increased for more responsive turning)
    """
    # Calculate derivative with moderate limiting to prevent jerky movements
    derivative = np.clip(center_offset - previous_offset, -0.03, 0.03)
    
    # PD control calculation with stronger curvature prediction
    steering = k_p * center_offset + k_d * derivative + k_c * curvature
    
    # Apply boost for larger offsets to improve responsiveness
    if abs(center_offset) > 0.15:
        steering *= 1.4  # Stronger boost for larger corrections
    elif abs(center_offset) > 0.08:
        steering *= 1.2  # Moderate boost for medium corrections
    
    # Increased steering limits for better responsiveness
    steering = np.clip(steering, -max_steering, max_steering)
    
    return steering


def smooth_acc_control(vehicle, target_distance, radar_distance, imu_speed, startup_phase=False, min_distance=10.0, max_distance=50.0, max_speed=25.0):
    throttle = 0.0
    brake = 0.0
    
    # Startup phase - gentle acceleration to get moving
    if startup_phase and imu_speed < 5.0:
        throttle = 0.4  # Gentle startup acceleration
        brake = 0.0
        return float(throttle), float(brake)
    
    if radar_distance is None or radar_distance == float("inf"):
        # No obstacle detected - cruise control mode
        if imu_speed < max_speed:
            # Progressive acceleration based on speed deficit
            speed_deficit = max_speed - imu_speed
            throttle = min(0.6, speed_deficit / max_speed * 0.8)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(0.3, (imu_speed - max_speed) / max_speed * 0.2)
    else:
        # Vehicle detected - ACC mode
        # Emergency braking for very close distances
        if radar_distance < 5.0:  # Emergency zone
            throttle = 0.0
            # Strong emergency braking
            brake = min(1.0, 0.5 + (5.0 - radar_distance) / 5.0 * 0.5)
            print(f"EMERGENCY BRAKING: Distance={radar_distance:.1f}m, Brake={brake:.2f}")
        
        elif radar_distance < target_distance:
            # Need to slow down - apply progressive braking
            throttle = 0.0
            distance_ratio = radar_distance / target_distance
            # More aggressive braking for shorter distances
            brake = min(0.8, (1.0 - distance_ratio) * 0.6 + min(0.2, imu_speed / 30.0))
            
        elif radar_distance > target_distance * 1.5:
            # Vehicle is far enough - accelerate cautiously
            distance_excess = min(radar_distance - target_distance, 20.0)
            throttle = min(0.6, distance_excess / 40.0)
            brake = 0.0
            
        else:
            # Maintain current speed in target zone
            if imu_speed > max_speed:
                throttle = 0.0
                brake = min(0.2, (imu_speed - max_speed) / max_speed * 0.1)
            else:
                throttle = min(0.4, (max_speed - imu_speed) / max_speed * 0.3)
                brake = 0.0

    return float(throttle), float(brake)


def calculate_lane_curvature(camera_data, lane_colors, image_width):
    """
    Calculate lane curvature by analyzing lane markings at different distances
    Returns curvature value: positive = right turn, negative = left turn
    """
    if not camera_data or "annotation" not in camera_data or camera_data["annotation"] is None:
        return 0.0

    annotation_pil = camera_data["annotation"]
    annotation_img = np.array(annotation_pil)[:, :, :3]
    height = annotation_img.shape[0]
    
    # Analyze lane markings at different vertical strips (distances)
    # Near: bottom 1/3, Middle: middle 1/3, Far: top 1/3
    strip_height = height // 3
    
    lane_centers = []
    distances = []
    
    for strip_idx in range(3):
        y_start = strip_idx * strip_height
        y_end = (strip_idx + 1) * strip_height
        strip = annotation_img[y_start:y_end, :]
        
        # Find lane markings in this strip
        all_lane_xs = []
        for lane_name, lane_color in lane_colors:
            mask = np.all(strip == lane_color, axis=-1)
            ys, xs = np.where(mask)
            
            if len(xs) > 5:  # Minimum pixels for valid detection
                all_lane_xs.extend(xs)
        
        if len(all_lane_xs) > 10:  # Sufficient lane data
            # Use histogram to find lane center
            hist, bin_edges = np.histogram(all_lane_xs, bins=16, range=(0, image_width))
            
            # Find peaks
            peaks = []
            for i in range(1, len(hist)-1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 3:
                    peak_x = (bin_edges[i] + bin_edges[i+1]) / 2
                    peaks.append(peak_x)
            
            if peaks:
                if len(peaks) >= 2:
                    # Multiple peaks - use outer ones for lane boundaries
                    peaks.sort()
                    lane_center = (peaks[0] + peaks[-1]) / 2.0
                else:
                    # Single peak
                    lane_center = peaks[0]
                
                # Normalize to image center offset
                center_offset = (lane_center - image_width / 2.0) / (image_width / 2.0)
                lane_centers.append(center_offset)
                # Distance weight: near=1.0, middle=2.0, far=3.0
                distances.append(strip_idx + 1.0)
    
    if len(lane_centers) < 2:
        return 0.0  # Not enough data for curvature
    
    # Calculate curvature using polynomial fit
    try:
        if len(lane_centers) >= 3:
            # Quadratic fit for better curvature estimation
            coeffs = np.polyfit(distances, lane_centers, 2)
            # Curvature is 2 * a (second derivative)
            curvature = 2.0 * coeffs[0]
        else:
            # Linear fit - rate of change
            coeffs = np.polyfit(distances, lane_centers, 1)
            curvature = coeffs[0]  # Slope indicates curvature direction
        
        # Scale and limit curvature
        curvature = np.clip(curvature * 2.0, -0.5, 0.5)
        return curvature
        
    except:
        return 0.0


def draw_predicted_path(color_img, curvature, image_center, lookahead_distance=100):
    """
    Draw the predicted curved path based on lane curvature
    """
    try:
        height, width = color_img.shape[:2]
        
        # Starting point (bottom center of image)
        start_x = image_center
        start_y = height - 1
        
        # Draw path points based on curvature
        path_points = []
        
        for i in range(0, lookahead_distance, 3):  # Every 3 pixels for performance
            # Calculate lateral displacement based on curvature
            y = start_y - i
            if y < height * 0.3:  # Stop at top third of image
                break
                
            # Calculate x displacement from center based on curvature
            # Enhanced curve calculation for better visibility
            distance_factor = i / lookahead_distance
            lateral_offset = curvature * (i ** 1.6) * 0.8  # More pronounced curve
            x = int(start_x + lateral_offset)
            
            # Keep within image bounds
            if 0 <= x < width and 0 <= y < height:
                path_points.append((x, y))
        
        # Draw the predicted path
        if len(path_points) > 1:
            # Draw path as connected lines with varying thickness
            for i in range(len(path_points) - 1):
                # Thicker lines at the bottom, thinner at the top
                thickness = max(1, 4 - i // 8)
                cv2.line(color_img, path_points[i], path_points[i + 1], (255, 0, 255), thickness)
            
            # Draw direction indicator at multiple points
            for i in range(5, min(len(path_points), 25), 10):
                cv2.circle(color_img, path_points[i], 3, (255, 0, 255), -1)
                
        # Add curvature info text with background for better visibility
        curvature_text = f"Curvature: {curvature:.4f}"
        text_size = cv2.getTextSize(curvature_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(color_img, (8, 45), (8 + text_size[0] + 4, 75), (0, 0, 0), -1)
        cv2.putText(
            color_img,
            curvature_text,
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2,
            cv2.LINE_AA,
        )
        
    except Exception as e:
        print(f"Path visualization error: {e}")


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

    camera = Camera("FrontCamera", bng, vehicle1, pos=(0, -1, 1.5), resolution=(640, 480), field_of_view_y=100,
                    is_render_colours=True, is_render_annotations=True, is_render_depth=True)
    radar = IdealRadar("idealRADAR1", bng, vehicle1, physics_update_time=0.01)
    imu = AdvancedIMU("imu1", bng, vehicle1, physics_update_time=0.01, pos=(0, 0, 1.7), dir=(0, -1, 0), up=(0, 0, 1))

    print("Setting AI Traffic...")
    vehicle2.ai.set_mode("traffic")
    vehicle1.ai.set_mode("manual")

    print("Initializing autonomous driving system...")
    sleep(3)
    print("ACC + ALC starting automatically in 2 seconds...")
    sleep(2)

    car_color = get_car_annotation_color(bng)
    lane_colors = get_lane_annotation_color(bng)
    if car_color is None:
        print("Warning: Could not find car annotation color! Using default.")
        car_color = (255, 0, 0)  # Default red color
    if not lane_colors:
        print("Warning: Could not find lane annotation colors! Using default.")
        lane_colors = [("DEFAULT", (255, 255, 255))]  # Default white color
    
    print(f"Using car color: {car_color}")
    print(f"Using lane colors: {lane_colors}")

    target_distance = 25.0
    print("ACC + ALC activated autonomously. Press BACKSPACE to exit.")
    print("Vehicle will start moving automatically...")
    
    # Autonomous startup variables
    startup_phase = True
    startup_counter = 0
    startup_duration = 40  # Frames to accelerate during startup (2 seconds at 20fps)
    
    # Enhanced steering smoothing for gradual movements
    previous_steering = 0.0
    previous_offset = 0.0
    previous_curvature = 0.0
    offset_buffer = [0.0] * 10  # Longer smoothing for stability
    steering_buffer = [0.0] * 8  # Much longer smoothing for gradual steering
    curvature_buffer = [0.0] * 6  # Smoothing for curvature prediction
    steering_rate_limit = 0.035   # Increased for more responsive steering changes

    while True:
        radar_data = radar.poll()
        radar_distance = None
        
        # Parse radar data more robustly
        if radar_data:
            try:
                # Handle different radar data formats
                if isinstance(radar_data, dict):
                    # Look for vehicle data in the dictionary
                    for key, value in radar_data.items():
                        if isinstance(value, dict):
                            # Check for vehicle distance data
                            if 'closestVehicles1' in value:
                                vehicle_info = value['closestVehicles1']
                                if isinstance(vehicle_info, dict) and 'relDistX' in vehicle_info:
                                    radar_distance = float(vehicle_info['relDistX'])
                                    break
                            elif 'relDistX' in value:
                                radar_distance = float(value['relDistX'])
                                break
                            elif 'distance' in value:
                                radar_distance = float(value['distance'])
                                break
            except Exception as e:
                print(f"Radar parsing error: {e}")

        camera_data = camera.poll()
        
        # Calculate curvature prediction first
        raw_curvature = calculate_lane_curvature(camera_data, lane_colors, image_width=640)
        
        # Smooth curvature prediction
        curvature_buffer.pop(0)
        curvature_buffer.append(raw_curvature)
        smoothed_curvature = float(np.mean(curvature_buffer))
        
        # Process camera with curvature for visualization
        status = "STARTUP" if startup_phase else "AUTO"
        camera_img, camera_distance = process_camera_data(camera_data, car_color, lane_colors, radar_distance, smoothed_curvature, status)
        
        # Use camera distance as backup if radar fails
        if radar_distance is None and camera_distance != float("inf"):
            radar_distance = camera_distance

        center_offset = estimate_lane_center_offset(camera_data, lane_colors, image_width=640)
        
        # Extended moving average for very smooth offset
        offset_buffer.pop(0)
        offset_buffer.append(center_offset)
        smoothed_offset = float(np.mean(offset_buffer))
        
        # Calculate steering with gentle PD control + curvature prediction
        raw_steering = steering_control(smoothed_offset, previous_offset, smoothed_curvature)
        
        # Extended moving average for very smooth steering
        steering_buffer.pop(0)
        steering_buffer.append(raw_steering)
        target_steering = float(np.mean(steering_buffer))
        
        # Apply rate limiting for gradual steering changes
        steering_change = target_steering - previous_steering
        if abs(steering_change) > steering_rate_limit:
            # Limit the rate of change
            steering_change = np.sign(steering_change) * steering_rate_limit
        
        steering = previous_steering + steering_change
        steering = float(np.clip(steering, -0.6, 0.6))  # Use the same limit as steering_control function
        
        # Update previous values
        previous_steering = steering
        previous_offset = smoothed_offset
        previous_curvature = smoothed_curvature

        imu_data = imu.poll()
        imu_speed = 0.0
        if imu_data and "vel" in imu_data:
            vel = imu_data["vel"]
            imu_speed = np.linalg.norm([vel["x"], vel["y"], vel["z"]])

        # Handle startup phase
        if startup_phase:
            startup_counter += 1
            if startup_counter >= startup_duration or imu_speed > 8.0:
                startup_phase = False
                print("Startup phase complete - switching to full autonomous mode")

        throttle, brake = smooth_acc_control(vehicle1, target_distance, radar_distance, imu_speed, startup_phase)
        
        # Debug output with autonomous status
        if startup_phase:
            print(f"STARTUP: Speed: {imu_speed:.1f}m/s, T: {throttle:.2f}, Counter: {startup_counter}/{startup_duration}")
        elif radar_distance is not None and radar_distance < 30.0:
            print(f"Vehicle: {radar_distance:.1f}m, Speed: {imu_speed:.1f}m/s, T: {throttle:.2f}, B: {brake:.2f}")
        elif imu_speed > 1.0:  # Show cruise status when moving
            print(f"CRUISE: Speed: {imu_speed:.1f}m/s, T: {throttle:.2f}")
        
        # Debug ALC output with curvature information
        if abs(center_offset) > 0.005 or abs(smoothed_curvature) > 0.02:  # Show when steering or curvature is active
            print(f"ALC: Offset={smoothed_offset:.3f}, Curvature={smoothed_curvature:.3f}, Target_Steer={target_steering:.3f}, Final_Steer={steering:.3f}")
        
        vehicle1.control(throttle=throttle, brake=brake, steering=steering)

        # Draw predicted path for visualization
        if camera_img is not None:
            draw_predicted_path(camera_img, smoothed_curvature, image_center=320)
        
        sleep(0.05)
        if msvcrt.kbhit() and msvcrt.getch() == b'\x08':
            print("Backspace pressed. Exiting ACC + ALC.")
            break

    bng.ui.show_hud()
    bng.disconnect()


if __name__ == "__main__":
    main()
