from enum import Flag, auto
from class_models.Context import Context
from re import Match
import re

class LoopEvent(Flag):
    START = auto()
    STOP  = auto()


class LoopProcessor():
    def __init__(self):
        pass

    def process(self, action_result, target_text, current_context: Context, recording_stack, \
                is_template: Match[str] | None):
        if action_result == LoopEvent.START:
            raw = target_text.strip()
            if is_template:
                var_name = raw.replace(is_template.group(1), "").strip()
                if not var_name:
                    print("Template loop missing variable name: ", var_name)
                else:
                    current_context.meta = {"loop": var_name}
                    current_context.sub_contexts = []

                    recording_stack[-1].append(current_context)
                    recording_stack.append(current_context.sub_contexts)
            else:
                m = re.search(r"\d+", raw)

                if not m:
                    print("No numeric loop count found.")
                else:
                    count = int(m.group(0))
                    current_context.meta = {"loop_count": count}
                    current_context.sub_contexts = []

                    recording_stack[-1].append(current_context)
                    recording_stack.append(current_context.sub_contexts)
            return False, recording_stack

        elif action_result == LoopEvent.STOP:
            if len(recording_stack) > 1:
                recording_stack.pop()

        return False, recording_stack