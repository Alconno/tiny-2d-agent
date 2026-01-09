import mss
from PIL import Image
from core.state import RuntimeState
import pyautogui



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


# In cases of resolution change
def scale_screenshot_box(rs: RuntimeState):
    cw,ch = pyautogui.size()
    pw,ph = rs.desktopWH
    if rs.screenshot_box:
        l,t,r,b = rs.screenshot_box
        rs.screenshot_box = (l*cw//pw, t*ch//ph, r*cw//pw, b*ch//ph)
        rs.desktopWH = (cw, ch)

