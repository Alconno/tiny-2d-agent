from enum import Flag, auto
import time
from core.state import RuntimeState
from core.ocr import run_ocr
from core.logging import get_logger
log = get_logger(__name__) 



class WaitForEvent(Flag):
    TEXT     = auto()
    IMAGE    = auto()

class WaitForEventHandler():
    def __init__(self):
        from ma_utility import (take_screenshot, get_target_image, \
                            find_crop_in_image, extract_box_from_string_target)

        self.take_screenshot_func = take_screenshot
        self.get_target_image_func = get_target_image
        self.find_crop_in_image_func = find_crop_in_image
        self.extract_box_from_string_target = extract_box_from_string_target

    def parse_wait_timer(self, wait_str):
        if not wait_str:
            return 5
        try:
            return int(str(wait_str))
        except:
            import re
            match = re.search(r'\d+', str(wait_str))
            return int(match.group()) if match else 5

    def waitFor(self, rs: RuntimeState):

        parts = [p.strip() for p in rs.target_text.split("|", 1)]
        if len(parts) == 1:
            parts.append(None)
        real_ctx = parts[0]
        wait_timer = self.parse_wait_timer(parts[1]) if len(parts) > 1 else 5
        
        result = False
        start_time = time.time()
        
        if rs.action_event == WaitForEvent.IMAGE:
            while time.time() - start_time < wait_timer:
                log.info(f"Waiting {wait_timer - (time.time() - start_time)} more for {real_ctx}")
                screenshot, offset = self.take_screenshot_func(rs.screenshot_box)
                matching = self.get_target_image_func(rs.models.embd_func, real_ctx, "./clickable_images")
                if matching:
                    _, bbox = self.find_crop_in_image_func(screenshot, matching, offset=offset)
                    if bbox:
                        log.info(f"Found image {real_ctx}")
                        result = True
                        break
                time.sleep(0.1)
        elif rs.action_event == WaitForEvent.TEXT:
            while time.time() - start_time < wait_timer:
                log.info(f"Waiting {wait_timer - (time.time() - start_time)} more for {real_ctx}")
                screenshot, offset = self.take_screenshot_func(rs.screenshot_box)
                emb = run_ocr(screenshot, offset, rs)
                target = self.extract_box_from_string_target(rs, emb)
                if target and target.get("result"):
                    log.debug(target)
                    log.info(f"Found text {real_ctx}")
                    result = True
                    break
                time.sleep(0.1)
        else: 
            result = True
        
        if not result and start_time:
            remaining = max(0, wait_timer - (time.time() - start_time))
            result = remaining <= 0
        
        return not result # failed = False if pass


