from enum import Flag, auto
from core.state import RuntimeState
from utility.dogshitretard import parse_delay
import time

class Timer(Flag):
    SLEEP   = auto()


class TimerHandler():
    def __init__(self):
        pass

    def parse_delay_and_sleep(self, rs: RuntimeState):
        failed = True
        delay_s = parse_delay(rs.target_text)
        if delay_s:
            time.sleep(delay_s)
            failed = False
        return failed
