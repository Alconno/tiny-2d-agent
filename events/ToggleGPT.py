from pynput.keyboard import Controller, Key
from enum import Enum, auto
from core.state import RuntimeState
import time


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
        print(f"GPT is now {'ON' if rs.use_gpt else 'OFF'}. Say 'toggle GPT' to switch again.")
        return False # success