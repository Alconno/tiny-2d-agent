from core.ocr import run_ocr
from core.state import RuntimeState
from core.recording import append_condition_to_recording_seq
from enum import Enum, auto
import re
from core.logging import get_logger
log = get_logger(__name__) 


class Condition(Enum):
    IF      = auto()
    END_IF  = auto()

class ConditionHandler():
    def __init__(self):
        from utility import (take_screenshot, extract_box_from_string_target,\
                              get_target_image, find_crop_in_image, get_matching_str)

        self.take_screenshot_func = take_screenshot
        self.extract_box_target = extract_box_from_string_target
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

        
    def check_text(self, condition, rs: RuntimeState):
        screenshot, offset = self.take_screenshot_func(rs.screenshot_box)
        rs.target_text = condition["query"]
        rs.color_list = condition["colors"]

        embd_lines = run_ocr(screenshot, offset, rs)
        target = self.extract_box_target(rs, embd_lines)

        found = target is not None and target.get("result") is not None and target.get("score", 0) > 0.7
        result = self.apply_negate(found, condition)

        return result, target



    def check_image(self, condition, rs: RuntimeState):
        screenshot, _ = self.take_screenshot_func(rs.screenshot_box)
        matching = self.get_target_image_func(
            rs.models.embd_func, condition["query"], "./clickable_images"
        )

        bbox = None
        if matching:
            _, bbox = self.find_crop_in_image_func(screenshot, matching)

        found = bbox is not None
        result = self.apply_negate(found, condition)

        return result, bbox

    

    def check_variable(self, condition, rs: RuntimeState):
        var_name = self.get_matching_str_func(
            condition["query"], list(rs.variables.keys()), rs.models.embd_func
        )

        found = var_name in rs.variables
        result = self.apply_negate(found, condition)

        return result, rs.variables.get(var_name, None)


    
    def check_condition(self, condition, rs: RuntimeState):
        failed = True
        if condition:
            if condition["type"] == "text": 
                passed, target = self.check_text(condition, rs)
                append_condition_to_recording_seq(rs.current_context)
            elif condition["type"] == "image":
                passed, target = self.check_image(condition, rs)
                append_condition_to_recording_seq(rs.current_context)
            elif condition["type"] == "variable":
                passed, var_obj = self.check_variable(condition, rs)
                append_condition_to_recording_seq(rs.current_context)
            if passed:
                failed = False
                if rs.current_context.sub_contexts and not rs.recording_state["active"]:
                    for sub_ctx in reversed(rs.current_context.sub_contexts):
                        rs.context_queue.appendleft(sub_ctx)
                log.info(f"Condition '{rs.target_text}' has successfully passed!")
            else:
                log.warning(f"Condition '{rs.target_text}' has failed.")
        return failed



    def handle_condition(self, rs: RuntimeState):
        failed = True
        if rs.action_event == Condition.IF:
            condition = self.parse_condition(rs.target_text) # {type, query, colors, negate}
            failed = self.check_condition(condition, rs)
        elif rs.action_event == Condition.END_IF:
            if rs.recording_stack:
                rs.recording_stack.pop()
            failed = False
        return failed