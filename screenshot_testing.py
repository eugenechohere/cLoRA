import mss
from pynput import mouse, keyboard
import threading
from pathlib import Path
from datetime import datetime
import uuid
import cv2
import numpy as np


class ScreenshotCapture:
    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = str(uuid.uuid4())[:8]
        self.output_dir = Path(f"output/{timestamp}_{random_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.count = 0
        self.keyboard_timer = None
        self.mouse_timer = None
    
    def take_screenshot(self):
        with mss.mss() as sct:
            screenshot = sct.grab(sct.monitors[1])
            # Convert to numpy array
            img = np.array(screenshot)
            # Convert BGRA to BGR (remove alpha channel)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            # Resize to 1080p height, maintain aspect ratio
            height, width = img.shape[:2]
            target_height = 720
            scale = target_height / height
            new_width = int(width * scale)
            img_resized = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_LANCZOS4)
            # Save
            filename = self.output_dir / f"screenshot_{self.count:04d}.png"
            cv2.imwrite(str(filename), img_resized)
            print(f"Saved: screenshot_{self.count:04d}.png")
            self.count += 1
    
    def cancel_keyboard_timer(self):
        if self.keyboard_timer and self.keyboard_timer.is_alive():
            self.keyboard_timer.cancel()
    
    def cancel_mouse_timer(self):
        if self.mouse_timer and self.mouse_timer.is_alive():
            self.mouse_timer.cancel()
    
    def on_mouse_release(self, x, y, button, pressed):
        if not pressed:
            self.cancel_keyboard_timer()
            self.cancel_mouse_timer()
            self.mouse_timer = threading.Timer(0.3, self.take_screenshot)
            self.mouse_timer.start()
    
    def on_key_release(self, key):
        self.cancel_keyboard_timer()
        self.cancel_mouse_timer()
        self.keyboard_timer = threading.Timer(1.0, self.take_screenshot)
        self.keyboard_timer.start()
    
    def run(self):
        print(f"Saving to: {self.output_dir}")
        print("Screenshots resized to 1080p height (aspect ratio preserved)")
        print("Mouse: 300ms debounce | Keyboard: 1s debounce | Ctrl+C to stop")
        
        mouse_listener = mouse.Listener(on_click=self.on_mouse_release)
        keyboard_listener = keyboard.Listener(on_release=self.on_key_release)
        
        mouse_listener.start()
        keyboard_listener.start()
        
        mouse_listener.join()
        keyboard_listener.join()


if __name__ == "__main__":
    ScreenshotCapture().run()
