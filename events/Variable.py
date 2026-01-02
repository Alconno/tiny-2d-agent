from enum import Flag, auto
from class_models.Variable import Variable
from core.state import RuntimeState
from core.ocr import run_ocr

class VariableEvent(Flag):
    SET     = auto()


class VariableHandler():
    def __init__(self):
        from utility import (take_screenshot, extract_box_from_string_target,\
                             extract_box_from_numeric_target, apply_offset_to_var)


        self.take_screenshot_func = take_screenshot
        self.extract_box_from_string_target = extract_box_from_string_target
        self.extract_box_from_numeric_target = extract_box_from_numeric_target
        self.apply_offset = apply_offset_to_var

    
    # ctx input like "set var '<name> <type> <desc>'"
    # <type> string: <desc> like <like 'str'>
    # <type> num: <desc> like <'<' val> or <'>' val> or <any>
    # single found => single var, more found => list
    def process_event(self, rs: RuntimeState):
        screenshot, offset = self.take_screenshot_func(rs.screenshot_box)
         
        if rs.action_event == VariableEvent.SET:
            var = Variable.extract_structured_var(rs.target_text)
            is_num = var.type=="number" or var.type=="num"

            emb_lines = run_ocr(screenshot, offset, rs, number_only=is_num)
            
            if is_num:
                # Number
                targets = self.extract_box_from_numeric_target(rs, emb_lines, return_all=True)
            else:
                # String
                # Will not include 'color' field in answer as its only used dynamically during embd matching
                targets = self.extract_box_from_string_target(rs, emb_lines, return_all=True)
                if targets == None: return var
                targets = [
                    {'score': t['score'], 
                     'text': t['span'],
                     'match': t['result']['text'], 
                     'color': None, 
                     'bbox': t['result']['bbox']}
                     for t in targets]
                targets.sort(key=lambda x: x['score'], reverse=True) # descending
            
            var.value = targets
            self.apply_offset(offset, var)

            return var
        
        return None


    def handle_variable(self, rs: RuntimeState):
        if rs.target_text:
            var = self.process_event(rs)
            rs.variables[var.name] = var
            failed = var is None
 

