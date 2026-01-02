import mss
from PIL import Image


def screenshot_raw():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        return img

def take_screenshot(screenshot_box=None):
    screenshot = screenshot_raw().convert("RGB")
    offset = (0, 0)
    if screenshot_box:
        left, top, _, _ = screenshot_box
        offset = (left, top)
        screenshot = screenshot.crop(screenshot_box)
    return screenshot, offset