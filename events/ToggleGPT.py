from enum import Enum
from core.state import RuntimeState
from core.logging import get_logger
log = get_logger(__name__) 


class ToggleGPT(Enum):
    TOGGLE = bool


class ToggleGPTHandler():
    def __init__(self):
        pass

    def handle(self, rs: RuntimeState):
        if " on" in rs.current_context.text.lower():
            rs.use_gpt = True
        elif " off" in rs.current_context.text.lower():
            rs.use_gpt = False
        else:
            rs.use_gpt = not rs.use_gpt
        log.info(f"GPT is now {'ON' if rs.use_gpt else 'OFF'}. Say 'toggle GPT' to switch again.")
        return False # success