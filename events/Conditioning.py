from pynput.keyboard import Controller, Key
from enum import Enum, auto
import re
import cv2
import numpy as np

class Condition(Enum):
    IF      = auto()
    END_IF  = auto()

class ConditionProcessor():
    def __init__(self):
        from utility.dogshitretard import take_screenshot, extract_box_target,\
                                        get_target_image, get_matching_str
        from utility.image_matching import find_crop_in_image

        self.take_screenshot_func = take_screenshot
        self.extract_box_target = extract_box_target
        self.get_target_image_func = get_target_image
        self.find_crop_in_image_func = find_crop_in_image
        self.get_matching_str_func = get_matching_str

    """
    Text:           If text 'text' exists
    Text+color:     If blue text 'text' exists
    Image:          If image 'image' exists
    """
    def parse_condition(self, ctx: str):
        ctx = ctx.strip().lower()

        negate = ctx.endswith(" negate")
        if negate:
            ctx = ctx[:-7].strip()

        if ctx.startswith("if "):
            ctx = ctx[3:].strip()

        # TEXT
        m = re.match(r'(?:([\w\s]+)\s+)?text\s+(?:"([^"]+)"|([^\"]+))\s+exists', ctx)
        if m:
            colors, q1, q2 = m.groups()
            return {
                "type": "text",
                "query": (q1 or q2).strip(),
                "colors": [c.lower() for c in re.split(r'\s+(?:or|and)\s+', colors)] if colors else None,
                "negate": negate
            }

        # IMAGE
        m = re.match(r'image\s+(?:"([^"]+)"|([^\"]+))\s+exists', ctx)
        if m:
            return {
                "type": "image",
                "query": (m.group(1) or m.group(2)).strip(),
                "negate": negate
            }

        # VARIABLE
        m = re.match(r'variable\s+(?:"([^"]+)"|([^\"]+))\s+exists', ctx)
        if m:
            return {
                "type": "variable",
                "query": (m.group(1) or m.group(2)).strip(),
                "negate": negate
            }

        return None


    def apply_negate(self, found: bool, condition: dict):
        return not found if condition.get("negate", False) else found

        
    def check_text(self, condition, screenshot_box, run_ocr_func, embd_func):
        screenshot, offset = self.take_screenshot_func(screenshot_box)
        ctx = condition["query"]
        color_list = condition["colors"]

        embd_lines = run_ocr_func(screenshot, offset, color_list)
        target = self.extract_box_target(ctx, embd_lines, embd_func, color_list)

        found = target is not None and target.get("result") is not None and target.get("score", 0) > 0.7
        found = self.apply_negate(found, condition)

        return found, target



    def check_image(self, condition, screenshot_box, embd_func):
        screenshot, _ = self.take_screenshot_func(screenshot_box)
        matching = self.get_target_image_func(
            embd_func, condition["query"], "./clickable_images"
        )

        bbox = None
        if matching:
            _, bbox = self.find_crop_in_image_func(screenshot, matching)

        found = bbox is not None
        found = self.apply_negate(found, condition)

        return found, bbox

    

    def check_variable(self, condition, variables: dict, embd_func):
        var_name = self.get_matching_str_func(
            condition["query"], list(variables.keys()), embd_func
        )

        found = var_name in variables
        found = self.apply_negate(found, condition)

        return found, variables.get(var_name, None)


        

