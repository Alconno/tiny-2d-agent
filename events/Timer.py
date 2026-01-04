from enum import Flag, auto
from core.state import RuntimeState
from ma_utility import parse_delay
import time
from core.logging import get_logger
log = get_logger(__name__) 

class Timer(Flag):
    SLEEP   = auto()


class TimerHandler():
    def __init__(self):
        pass

    def parse_delay_and_sleep(self, rs: RuntimeState):
        failed = True
        delay_s = parse_delay(rs.target_text)
        if delay_s:
            log.info(f"Sleeping {delay_s}s")
            time.sleep(delay_s)
            failed = False
        return failed, {"event": rs.action_event, "payload": delay_s}
