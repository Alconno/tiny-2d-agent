from pynput.keyboard import Controller, Key
from enum import Enum, auto
import time
from core.state import RuntimeState
from core.logging import get_logger
log = get_logger(__name__) 

class KeyboardEvent(Enum):
    WRITE = auto()
    PRESS = auto()

class Keyboard:
    def __init__(self, embd_func):
        from utility import (cmp_txt_and_embs)

        self.embd_func = embd_func
        self.cmp_txt_and_embs = cmp_txt_and_embs

        self.k = Controller()

        self.alias = {k.name.lower(): k for k in Key}
        self.alias.update({
            "return": Key.enter,
            "del": Key.delete,
            "delete": Key.delete,
            "ctrl": Key.ctrl,
            "control": Key.ctrl,
            "alt": Key.alt,
            "shift": Key.shift,
        })

        self.alias_embds = []
        key_embs = embd_func(list(self.alias.keys()))
        for key, emb in zip(self.alias.keys(), key_embs):
            self.alias_embds.append((emb, key))

    def write(self, text: str):
        self.k.type(text)
        log.info(f"Wrote down text '{text}'")


    def press(self, text: str):
        words = text.lower().strip().split()
        keys_to_press = []

        for w in words:
            w = w.strip()
            if not w: continue

            if w in self.alias:
                keys_to_press.append(self.alias[w])
            elif len(w) == 1:
                keys_to_press.append(w)
            else:
                m = self.cmp_txt_and_embs(w, self.alias_embds, self.embd_func)
                if m: keys_to_press.append(self.alias[m["text"]])
                
        if not keys_to_press:
            log.warning(f"No valid keys found in '{text}'")
            return

        for k in keys_to_press: 
            self.k.press(k)
            log.info(f"Pressed key '{k}'")
        time.sleep(0.2)
        for k in reversed(keys_to_press): self.k.release(k)


    def execute(self, rs: RuntimeState):
        if rs.action_event == KeyboardEvent.WRITE:
            self.write(rs.target_text)
        elif rs.action_event == KeyboardEvent.PRESS:
            self.press(rs.target_text)
        return False # Did not fail