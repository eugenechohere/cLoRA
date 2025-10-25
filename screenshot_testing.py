import mss
import mss.tools
from pynput import mouse
import time
import threading
from pathlib import Path
from datetime import datetime
import uuid

# Create output directory with timestamp and random ID
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
random_id = str(uuid.uuid4())[:8]
output_dir = Path(f"output/{timestamp}_{random_id}")
output_dir.mkdir(parents=True, exist_ok=True)

count = 0

def take_screenshot_after_delay():
    global count
    time.sleep(0.1)  # Wait 100ms
    with mss.mss() as sct:
        screenshot = sct.grab(sct.monitors[1])
        filename = output_dir / f"screenshot_{count:04d}.png"
        mss.tools.to_png(screenshot.rgb, screenshot.size, output=str(filename))
        print(f"Saved: screenshot_{count:04d}.png")
        count += 1

def on_click(x, y, button, pressed):
    if not pressed:  # Mouse button released (mouse up)
        threading.Thread(target=take_screenshot_after_delay, daemon=True).start()

print("Click anywhere. Screenshot taken 100ms after each mouse release. Ctrl+C to stop.")
print(f"Saving to: {output_dir}")
with mouse.Listener(on_click=on_click) as listener:
    listener.join()
