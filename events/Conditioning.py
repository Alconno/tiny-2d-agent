from pynput.keyboard import Controller, Key
from enum import Enum, auto
import re


class Condition(Enum):
    IF      = auto()
    END_IF  = auto()

class ConditionProcessor():
    def __init__(self):
        from utility.dogshitretard import take_screenshot, extract_box_target_with_more_ctx, expand_color_logic,\
                                        get_target_image, get_matching_str
        from utility.image_matching import find_crop_in_image

        self.take_screenshot_func = take_screenshot
        self.extract_box_target_with_more_ctx_func = extract_box_target_with_more_ctx
        self.expand_color_logic_func = expand_color_logic
        self.get_target_image_func = get_target_image
        self.find_crop_in_image_func = find_crop_in_image
        self.get_matching_str_func = get_matching_str

    """
    Text:           If text 'text' exists
    Text+color:     If blue text 'text' exists
    Image:          If image 'image' exists
    """
    def parse_condition(self, ctx: str):
        ctx = ctx.strip()

        negate = False
        if ctx.lower().startswith("not "):
            negate = True
            ctx = ctx[4:].strip()

        text_match = re.match(
            r'(?:([\w\s]+)\s+)?text\s+(?:"([^"]+)"|([^\"]+))\s+exists',
            ctx, re.I
        )
        if text_match:
            colors_raw, q1, q2 = text_match.groups()
            query = (q1 or q2).strip()

            colors = []
            if colors_raw:
                colors = re.split(r'\s+(?:or|and)\s+', colors_raw.strip(), flags=re.I)
                colors = [c.lower() for c in colors if c]

            return {
                "type": "text",
                "query": query,
                "colors": colors or None,
                "negate": negate
            }

        image_match = re.match(
            r'image\s+(?:"([^"]+)"|([^\"]+))\s+exists',
            ctx, re.I
        )
        if image_match:
            q1, q2 = image_match.groups()
            query = (q1 or q2).strip()

            return {
                "type": "image",
                "query": query,
                "negate": negate
            }

        variable_match = re.match(
            r'variable\s+(?:"([^"]+)"|([^\"]+))\s+exists',
            ctx, re.I
        )
        if variable_match:
            q1, q2 = variable_match.groups()
            query = (q1 or q2).strip()

            return {
                "type": "variable",
                "query": query,
                "negate": negate
            }

    

        
    def check_text(self, condition, screenshot_box, run_ocr_func, embd_func):
        screenshot, offset = self.take_screenshot_func(screenshot_box)
        embd_lines = run_ocr_func(screenshot, offset, condition["colors"]) # offset used because of tiling
        expanded_ctxs = self.expand_color_logic_func(target_ctx=condition["query"])

        target = self.extract_box_target_with_more_ctx_func(expanded_ctxs, embd_lines, embd_func, condition["colors"])

        if target is None:
            return False
        return target.get("result") is not None and target.get("score", 0) > 0.7, target


    def check_image(self, condition, screenshot_box, embd_func):
        screenshot, _ = self.take_screenshot_func(screenshot_box)
        matching = self.get_target_image_func(
            embd_func, condition["query"], "./clickable_images"
        )
        bbox = None
        if matching:
            _, bbox = self.find_crop_in_image_func(screenshot, matching)
        return bbox is not None, bbox
    

    def check_variable(self, condition, variables: dict, embd_func):
        var_name = self.get_matching_str_func(condition["query"], list(variables.keys()), embd_func)

        return var_name in variables, variables.get(var_name, None)


        

