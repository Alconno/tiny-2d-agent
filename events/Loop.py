from enum import Flag, auto
from core.state import RuntimeState
import re
from core.logging import get_logger
log = get_logger(__name__) 

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
                        log.warning("Template loop missing variable name: ", var_name)
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
                m = re.search(r"\d+", raw)

                if not m:
                    log.warning(f"No numeric loop count found in '{raw}'")
                else:
                    count = int(m.group(0))

                    if rs.recording_state["active"]:
                        rs.current_context.sub_contexts = []
                        rs.recording_stack[-1].append(rs.current_context)
                        rs.recording_stack.append(rs.current_context.sub_contexts)
                    else:
                        for _ in range(count):
                            if rs.current_context.sub_contexts and not rs.recording_state["active"]:
                                for sub_ctx in reversed(rs.current_context.sub_contexts):
                                    rs.context_queue.appendleft(sub_ctx.copy())
            log.info(f"Starting loop {rs.target_text}")
            return False

        elif rs.action_result == LoopEvent.STOP:
            if len(rs.recording_stack) > 1:
                rs.recording_stack.pop()

        return False