from enum import Flag, auto
from core.state import RuntimeState
import re

class LoopEvent(Flag):
    START = auto()
    STOP  = auto()


class LoopHandler():
    def __init__(self):
        pass

    def process(self, rs: RuntimeState):
        
        if rs.action_event == LoopEvent.START:
            raw = rs.target_text.strip()
            if rs.is_template:
                if rs.recording_state["active"]:
                    var_name = raw.replace(rs.is_template.group(1), "").strip()
                    if not var_name:
                        print("Template loop missing variable name: ", var_name)
                    else:
                        rs.current_context.sub_contexts = []

                        rs.recording_stack[-1].append(rs.current_context)
                        rs.recording_stack.append(rs.current_context.sub_contexts)
                else:
                    # For multiple variables, loop subcontexts will be repeated with filled out template and returned as one
                    if rs.current_context.sub_contexts and not rs.recording_state["active"]:
                        for sub_ctx in reversed(rs.current_context.sub_contexts):
                            rs.context_queue.appendleft(sub_ctx)
            else:
                print("d")
                m = re.search(r"\d+", raw)

                if not m:
                    print("No numeric loop count found.")
                else:
                    count = int(m.group(0))

                    if rs.recording_state["active"]:
                        rs.current_context.sub_contexts = []
                        rs.recording_stack[-1].append(rs.current_context)
                        rs.recording_stack.append(rs.current_context.sub_contexts)
                    else:
                        print("count: ", count)
                        print("q size: ", len(rs.context_queue))
                        for _ in range(count):
                            if rs.current_context.sub_contexts and not rs.recording_state["active"]:
                                for sub_ctx in reversed(rs.current_context.sub_contexts):
                                    print("adding sub ctx: ", sub_ctx.text)
                                    rs.context_queue.appendleft(sub_ctx.copy())

            return False, rs.recording_stack

        elif rs.action_result == LoopEvent.STOP:
            if len(rs.recording_stack) > 1:
                rs.recording_stack.pop()

        return False, rs.recording_stack