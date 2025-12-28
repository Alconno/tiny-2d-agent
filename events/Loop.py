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
                recording_state, context_queue, is_template: Match[str] | None):
        
        if action_result == LoopEvent.START:
            raw = target_text.strip()
            if is_template:
                if recording_state["active"]:
                    var_name = raw.replace(is_template.group(1), "").strip()
                    if not var_name:
                        print("Template loop missing variable name: ", var_name)
                    else:
                        current_context.sub_contexts = []

                        recording_stack[-1].append(current_context)
                        recording_stack.append(current_context.sub_contexts)
                else:
                    # For multiple variables, loop subcontexts will be repeated with filled out template and returned as one
                    if current_context.sub_contexts and not recording_state["active"]:
                        for sub_ctx in reversed(current_context.sub_contexts):
                            context_queue.appendleft(sub_ctx)
            else:
                print("d")
                m = re.search(r"\d+", raw)

                if not m:
                    print("No numeric loop count found.")
                else:
                    count = int(m.group(0))

                    if recording_state["active"]:
                        current_context.sub_contexts = []
                        recording_stack[-1].append(current_context)
                        recording_stack.append(current_context.sub_contexts)
                    else:
                        print("count: ", count)
                        print("q size: ", len(context_queue))
                        for _ in range(count):
                            if current_context.sub_contexts and not recording_state["active"]:
                                for sub_ctx in reversed(current_context.sub_contexts):
                                    print("adding sub ctx: ", sub_ctx.text)
                                    context_queue.appendleft(sub_ctx.copy())

            return False, recording_stack

        elif action_result == LoopEvent.STOP:
            if len(recording_stack) > 1:
                recording_stack.pop()

        return False, recording_stack