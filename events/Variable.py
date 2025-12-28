from enum import Flag, auto
from class_models.Variable import Variable

class VariableEvent(Flag):
    SET     = auto()


class VariableProcessor():
    def __init__(self):
        from utility.dogshitretard import extract_box_target,\
                                        extract_numbers_target, take_screenshot, apply_offset_to_var

        self.take_screenshot_func = take_screenshot
        self.extract_box_target = extract_box_target
        self.extract_numbers_target = extract_numbers_target
        self.apply_offset = apply_offset_to_var

    
    # ctx input like "set var '<name> <type> <desc>'"
    # <type> string: <desc> like <like 'str'>
    # <type> num: <desc> like <'<' val> or <'>' val> or <any>
    # single found => single var, more found => list
    def process_event(self, parsed_action, ctx, screenshot_box, found_colors, embd_func, run_ocr_func):
        print("screenshot box: ", screenshot_box)
        screenshot, offset = self.take_screenshot_func(screenshot_box)
        action_result = parsed_action["result"]
         
        if action_result == VariableEvent.SET:
            var = Variable.extract_structured_var(ctx)
            is_num = var.type=="number" or var.type=="num"

            emb_lines = run_ocr_func(screenshot, offset, found_colors, number_only=is_num)
            
            if is_num:
                # Number
                targets = self.extract_numbers_target(ctx, emb_lines, embd_func, found_colors, return_all=True)
            else:
                # String
                # Will not include 'color' field in answer as its only used dynamically during embd matching
                targets = self.extract_box_target(ctx, emb_lines, embd_func, found_colors, return_all=True)
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

 

