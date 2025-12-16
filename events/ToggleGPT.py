from pynput.keyboard import Controller, Key
from enum import Enum, auto
import time


class ToggleGPT(Enum):
    TOGGLE = bool