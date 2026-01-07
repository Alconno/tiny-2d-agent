from enum import Flag, auto
from class_models.Variable import Variable
from core.state import RuntimeState
from core.ocr import run_ocr
from core.logging import get_logger
log = get_logger(__name__) 



class VariableEvent(Flag):
    SET     = auto()


class VariableHandler():
    def __init__(self):
        from ma_utility import (take_screenshot, extract_box_from_string_target,\
                             extract_box_from_numeric_target, apply_offset_to_var)


        self.take_screenshot_func = take_screenshot
        self.extract_box_from_string_target = extract_box_from_string_target
        self.extract_box_from_numeric_target = extract_box_from_numeric_target
        self.apply_offset = apply_offset_to_var

    # ----------------- Extract variables using OCR -----------------
    # ctx input like "set var '<name> <type> <desc>'"
    # <type> string: <desc> like <like 'str'>
    # <type> num: <desc> like <'<' val> or <'>' val> or <any>
    # single found => single var, more found => list
    def process_event(self, rs: RuntimeState):
        screenshot, offset = self.take_screenshot_func(rs.screenshot_box)
         
        if rs.action_event == VariableEvent.SET:
            var = Variable.extract_structured_var(rs.target_text)
            is_num = var.type=="number" or var.type=="num"
            rs.target_text = var.desc

            emb_lines = run_ocr(screenshot, rs, number_only=is_num)
            
            if is_num:
                # Number - {match, value, color, bbox}
                targets = self.extract_box_from_numeric_target(rs, emb_lines, return_all=True)
                if targets == None:
                    log.debug("Did not find any variable values")
                    return var
            
            # No point doing string atm as there is no strong enough GPT to reason properly with extracted strings
            """else:
                # String -  # {score, query, result{bbox, text, embedding, crop}}
                targets = self.extract_box_from_string_target(rs, emb_lines, return_all=True)
                if targets == None: 
                    log.debug("Did not find any variable values")
                    return var
                targets = [
                    {'score': t['score'], 
                     'query': t['query'],
                     'match': t['result']['text'], 
                     'color': None, 
                     'bbox': t['result']['bbox']}
                     for t in targets]
                targets.sort(key=lambda x: x['score'], reverse=True) # descending"""
            
            log.info(f"Found variable values: {[t['match'] for t in targets]}")
            var.value = targets
            self.apply_offset(offset, var)

            return var
        
        return None


    def handle_variable(self, rs: RuntimeState):
        failed = True
        var = None
        if rs.target_text:
            var = self.process_event(rs)
            rs.variables[var.name] = var
            failed = var is None
        return failed,  {"event": rs.action_event, "payload": var}
 

