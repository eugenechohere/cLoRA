import mss
from pynput import mouse, keyboard
import threading
from pathlib import Path
from datetime import datetime
import uuid
from PIL import Image, ImageDraw


class ScreenshotCapture:
    def __init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_id = str(uuid.uuid4())[:8]
        self.output_dir = Path(f"output/{timestamp}_{random_id}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.count = 0
        self.keyboard_timer = None
        self.mouse_timer = None
        self.current_mouse_pos = (0, 0)
        
        # Load cursor image with alpha channel
        cursor_path = Path(__file__).parent / "cursor_image.png"
        self.cursor_img = Image.open(cursor_path).convert("RGBA")
    
    def take_screenshot(self):
        with mss.mss() as sct:
            monitor = sct.monitors[1]
            screenshot = sct.grab(monitor)
            
            # Convert to PIL Image
            img = Image.frombytes("RGB", screenshot.size, screenshot.bgra, "raw", "BGRX")
            
            # Get current mouse position relative to monitor
            mouse_x, mouse_y = self.current_mouse_pos
            relative_x = mouse_x - monitor["left"]
            relative_y = mouse_y - monitor["top"]
            
            # Resize to 720p height, maintain aspect ratio
            target_height = 720
            scale = target_height / img.height
            new_width = int(img.width * scale)
            img = img.resize((new_width, target_height), Image.Resampling.LANCZOS)
            
            # Scale cursor position
            cursor_x = int(relative_x * scale)
            cursor_y = int(relative_y * scale)
            
            # Draw small yellow translucent circle at cursor
            circle_radius = 15
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            draw.ellipse(
                [cursor_x - circle_radius, cursor_y - circle_radius,
                 cursor_x + circle_radius, cursor_y + circle_radius],
                fill=(255, 255, 0, 40)
            )
            img = img.convert("RGBA")
            img = Image.alpha_composite(img, overlay)
            img = img.convert("RGB")
            
            # Scale and paste cursor (75% size)
            cursor_scale = scale * 0.75
            cursor_scaled = self.cursor_img.resize(
                (int(self.cursor_img.width * cursor_scale), int(self.cursor_img.height * cursor_scale)),
                Image.Resampling.LANCZOS
            )
            img = img.convert("RGBA")
            img.paste(cursor_scaled, (cursor_x, cursor_y), cursor_scaled)
            img = img.convert("RGB")
            
            # Save
            filename = self.output_dir / f"screenshot_{self.count:04d}.png"
            img.save(filename)
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
            self.current_mouse_pos = (x, y)
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
        print("Screenshots resized to 720p height (aspect ratio preserved)")
        print("Mouse: 300ms debounce | Keyboard: 1s debounce | Ctrl+C to stop")
        print("Cursor will be drawn at click position")
        
        mouse_listener = mouse.Listener(on_click=self.on_mouse_release)
        keyboard_listener = keyboard.Listener(on_release=self.on_key_release)
        
        mouse_listener.start()
        keyboard_listener.start()
        
        mouse_listener.join()
        keyboard_listener.join()


if __name__ == "__main__":
    ScreenshotCapture().run()
