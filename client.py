import requests
from mss import mss
import pygetwindow as gw
from io import BytesIO
from PIL import Image
import keyboard
import time

SERVER_URL = "http://192.168.1.157:5000/upload"  # Replace <server-ip> with your server's IP

def screenshot_active_window():
    """
    Capture the active window as an image.
    """
    active_window = gw.getActiveWindow()
    if active_window:
        bbox = {
            "top": active_window.top,
            "left": active_window.left,
            "width": active_window.width,
            "height": active_window.height
        }
        with mss() as sct:
            screenshot = sct.grab(bbox)
            img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
            return img
    else:
        print("No active window detected!")
        return None

def send_screenshot_to_server():
    """
    Capture the active window screenshot and send it to the server.
    """
    img = screenshot_active_window()
    if img:
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        try:
            response = requests.post(
                SERVER_URL,
                files={"file": buffer}
            )
            if response.status_code == 200:
                print("Screenshot sent successfully.")
            else:
                print("Failed to send screenshot:", response.text)
        except Exception as e:
            print("Error sending screenshot:", str(e))
    else:
        print("No image to send.")

def main():
    print("Press Ctrl + Shift to send a screenshot to the server. Press 'Esc' to exit.")
    
    while True:
        try:
            # Check for Ctrl + Shift key press
            if keyboard.is_pressed('ctrl') and keyboard.is_pressed('shift'):
                print("Taking screenshot...")
                send_screenshot_to_server()
                time.sleep(1)  # Prevent multiple triggers from key holding

            # Exit the script when 'Esc' is pressed
            if keyboard.is_pressed('esc'):
                print("Exiting...")
                break
        except KeyboardInterrupt:
            print("Program stopped.")
            break

if __name__ == "__main__":
    main()
