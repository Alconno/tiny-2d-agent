from enum import Flag, auto, Enum
from pynput.mouse import Button, Controller
from pynput.keyboard import Controller as KeyboardController, Key
import time

class MouseButton(Flag):
    LEFT   = auto()
    RIGHT  = auto()
    MIDDLE = auto()
    DOUBLE = auto()
    IMAGE  = auto()
    
    SHIFT_LEFT  = auto()
    SHIFT_RIGHT = auto()
    
    VAR_ALL = auto()
    VAR_TOP = auto()
    
    SPATIAL_LEFT  = auto()
    SPATIAL_RIGHT = auto()
    SPATIAL_ABOVE = auto()
    SPATIAL_BELOW = auto()




class Mouse:
    def __init__(self):
        from utility.dogshitretard import take_screenshot, get_target_image, extract_box_target
        from utility.dogshitretard import get_matching_str, get_spatial_location, apply_offset_to_bbox
        from utility.image_matching import find_crop_in_image

        self.take_screenshot_func = take_screenshot
        self.get_target_image_func = get_target_image
        self.find_crop_in_image_func = find_crop_in_image
        self.extract_box_target = extract_box_target
        self.get_matching_str_func = get_matching_str
        self.get_spatial_location = get_spatial_location
        self.apply_offset_to_bbox = apply_offset_to_bbox
        self.controller = Controller()
        self.kb_controller = KeyboardController()
        self._map = {
            MouseButton.LEFT:   Button.left,
            MouseButton.RIGHT:  Button.right,
            MouseButton.MIDDLE: Button.middle,
            MouseButton.IMAGE:  Button.left,

            MouseButton.SHIFT_LEFT:  Button.left,
            MouseButton.SHIFT_RIGHT: Button.right,
        }

    def move(self, x, y):
        self.controller.position = (x, y)

    def _iterate_buttons(self, buttons):
        for mb, pb in self._map.items():
            if mb in buttons:
                yield pb

    def click(self, x, y, button: MouseButton, count=1):
        self.move(x, y)
        time.sleep(0.1)
        self.move(x + 1, y + 1)
        time.sleep(0.1)

        shift_held = bool(button & (MouseButton.SHIFT_LEFT | MouseButton.SHIFT_RIGHT))
        real_button = button & ~MouseButton.DOUBLE
        total_clicks = 2 if MouseButton.DOUBLE in button else 1
        print(shift_held, real_button, total_clicks)

        if shift_held:
            self.kb_controller.press(Key.shift)
            time.sleep(0.05)

        for _ in range(total_clicks * count):
            for btn in self._iterate_buttons(real_button):
                self.controller.press(btn)
            for btn in self._iterate_buttons(real_button):
                self.controller.release(btn)
            if total_clicks > 1:
                time.sleep(0.05)

        if shift_held:
            self.kb_controller.release(Key.shift)

    def press(self, button: MouseButton):
        for btn in self._iterate_buttons(button & ~MouseButton.DOUBLE):
            self.controller.press(btn)

    def release(self, button: MouseButton):
        for btn in self._iterate_buttons(button & ~MouseButton.DOUBLE):
            self.controller.release(btn)

    def execute(self, action, target):
        if target:
            x, y, w, h = target["result"]["bbox"]
            cx, cy = x + w/2, y + h/2
            button = action["result"]
            self.click(int(cx), int(cy), button)
        else:
            (cx, cy) = self.controller.position
            self.click(int(cx), int(cy), action["result"])

    def process_event(self, parsed_action, ctx, variables, screenshot_box, found_colors, embd_func, run_ocr_func):
        screenshot, offset = self.take_screenshot_func(screenshot_box)
        action_result = parsed_action["result"]
        failed = False

        if action_result & MouseButton.IMAGE:
            matching = self.get_target_image_func(embd_func, ctx, "./clickable_images")
            if matching:
                print("Found image ", ctx)
                _, bbox = self.find_crop_in_image_func(screenshot, matching, offset=offset)
                print("Found bbox: ", bbox)
                if bbox:
                    self.execute(parsed_action, {"result": {"bbox": bbox}})
                    failed = False
            else:
                print("image not found:", ctx)

        elif action_result & MouseButton.VAR_ALL:
            var_name = self.get_matching_str_func(ctx, list(variables.keys()), embd_func) 
            if var_name:
                var_obj = variables.get(var_name)
                var_values = var_obj.get("value") if isinstance(var_obj, dict) else getattr(var_obj, "value", []) or []
                for value in var_values:
                    bbox = value.get("bbox") if isinstance(value, dict) else None
                    if bbox:
                        print("Clicking var value:", value, "\n\n-------------------")
                        self.execute(parsed_action, {"result": {"bbox": bbox}})
                        time.sleep(1)

        elif action_result & MouseButton.VAR_TOP:
            var_name = self.get_matching_str_func(ctx, list(variables.keys()), embd_func)
            if var_name:
                var_obj = variables.get(var_name)
                var_values = var_obj.get("value") if isinstance(var_obj, dict) else getattr(var_obj, "value", []) or []
                if var_values:
                    top = var_values[0]
                    bbox = top.get("bbox") if isinstance(top, dict) else None
                    if bbox:
                        print("Clicking var value:", top, "\n\n-------------------")
                        self.execute(parsed_action, {"result": {"bbox": bbox}})
                        time.sleep(1)

        elif action_result & (MouseButton.SPATIAL_ABOVE | MouseButton.SPATIAL_BELOW |
                            MouseButton.SPATIAL_LEFT | MouseButton.SPATIAL_RIGHT):
            emb = run_ocr_func(screenshot, offset, found_colors)
            
            parts = [p.strip() for p in ctx.split("|", 1)]
            if len(parts) == 1:
                parts.append(None)
            real_ctx = parts[0]
            spatial_search_condition = parts[1] or "object" # 'text' (color edge detection) or 'object'(defaut)(gray edge detection)
            
            target = self.extract_box_target(ctx, emb, embd_func, found_colors)

            if target:
                bbox = target["result"]["bbox"]
                new_bbox = self.get_spatial_location(action_result, bbox, offset, screenshot, spatial_search_condition)
                self.execute(parsed_action, {"result": {"bbox": new_bbox}})

        else:
            emb = run_ocr_func(screenshot, offset, found_colors)
            target = self.extract_box_target(ctx, emb, embd_func, found_colors)
            if target and target['result'] and target['result']['bbox']:
                self.apply_offset_to_bbox(offset, target['result']['bbox'])
            if target:
                self.execute(parsed_action, target)
                failed = False

        return failed
