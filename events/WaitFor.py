from enum import Flag, auto
import re, time
class WaitForEvent(Flag):
    TEXT     = auto()
    IMAGE    = auto()

class WaitForEventProcessor():
    def __init__(self):
        from utility.dogshitretard import take_screenshot, get_target_image, expand_color_logic, extract_box_target_with_more_ctx
        from utility.image_matching import find_crop_in_image

        self.take_screenshot_func = take_screenshot
        self.get_target_image_func = get_target_image
        self.find_crop_in_image_func = find_crop_in_image
        self.expand_color_logic_func = expand_color_logic
        self.extract_box_target_with_more_ctx_func = extract_box_target_with_more_ctx

    def parse_wait_timer(self, wait_str):
        if not wait_str:
            return 5
        try:
            return int(str(wait_str))
        except:
            import re
            match = re.search(r'\d+', str(wait_str))
            return int(match.group()) if match else 5

    def waitFor(self, action_result, ctx, found_colors, screenshot_box, embd_func, run_ocr_func):
        parts = [p.strip() for p in ctx.split("|", 1)]
        if len(parts) == 1:
            parts.append(None)
        real_ctx = parts[0]
        wait_timer = self.parse_wait_timer(parts[1]) if len(parts) > 1 else 5
        
        result = False
        start_time = time.time()
        
        if action_result == WaitForEvent.IMAGE:
            while time.time() - start_time < wait_timer:
                print(f"Waiting {wait_timer - (time.time() - start_time)} more for {real_ctx}")
                screenshot, offset = self.take_screenshot_func(screenshot_box)
                matching = self.get_target_image_func(embd_func, real_ctx, "./clickable_images")
                if matching:
                    _, bbox = self.find_crop_in_image_func(screenshot, matching, offset=offset)
                    if bbox:
                        print(f"Found image {real_ctx}")
                        result = True
                        break
                time.sleep(0.1)
        elif action_result == WaitForEvent.TEXT:
            while time.time() - start_time < wait_timer:
                print(f"Waiting {wait_timer - (time.time() - start_time)} more for {real_ctx}")
                screenshot, offset = self.take_screenshot_func(screenshot_box)
                emb = run_ocr_func(screenshot, offset, found_colors)
                expanded = self.expand_color_logic_func(target_ctx=real_ctx)
                target = self.extract_box_target_with_more_ctx_func(expanded, emb, embd_func, found_colors)
                if target and target.get("result"):
                    print(target)
                    print(f"Found text {real_ctx}")
                    result = True
                    break
                time.sleep(0.1)
        else: 
            result = True
        
        if not result and start_time:
            remaining = max(0, wait_timer - (time.time() - start_time))
            result = remaining <= 0
        
        return not result # failed = False if pass


