import sys
from pathlib import Path
from typing import Dict, Tuple

sys.path.append(str(Path(__file__).resolve().parent.parent))

from events.Mouse.MouseEvent import MouseButton
from events.Keyboard import KeyboardEvent
from events.SequenceHandler import SequenceEvent
from events.Timer import Timer
from events.ScreenCapture import ScreenCaptureEvent
from events.Conditioning import Condition
from events.Variable import VariableEvent
from events.Loop import LoopEvent
from events.ToggleGPT import ToggleGPT
from events.WaitFor import WaitForEvent

# Base events
_base_possible_events: Dict[Tuple[str, ...], object] = {
    # Click events
    ("click", "left click"): MouseButton.LEFT,
    ("right click",): MouseButton.RIGHT,
    ("middle click",): MouseButton.MIDDLE,
    ("double click", "open"): MouseButton.LEFT | MouseButton.DOUBLE,

    # Shift click events
    ("shift click", "shift and click", "shift left click"): MouseButton.SHIFT_LEFT,
    ("shift right", "shift right click"): MouseButton.SHIFT_RIGHT,

    # Image click events
    ("click image", "click on image", "find", "select image", "image click", "click icon", "click picture"): MouseButton.IMAGE | MouseButton.LEFT,

    # Coordinate click events
    ("click coord", "click coordinate", "left click coordinate"): MouseButton.COORD | MouseButton.LEFT,

    # Variable click events
    ("click all variable", "click every variable"): MouseButton.VAR_ALL | MouseButton.LEFT,
    ("click variable", "click one variable", "click top variable", "click best variable"): MouseButton.VAR_TOP | MouseButton.LEFT,

    # Spatial awareness click events
    ("click left of", "click left to"): MouseButton.SPATIAL_LEFT | MouseButton.LEFT,
    ("click right of", "click right to"): MouseButton.SPATIAL_RIGHT | MouseButton.LEFT,
    ("click above of", "click on top of", "click above"): MouseButton.SPATIAL_ABOVE | MouseButton.LEFT,
    ("click below of", "click under", "click below"): MouseButton.SPATIAL_BELOW | MouseButton.LEFT,

    # Keyboard events
    ("write", "type"): KeyboardEvent.WRITE,
    ("press",): KeyboardEvent.PRESS,

    # Sequence events
    ("start sequence recording", "start recording sequence", "record sequence", "start recording"): SequenceEvent.START,
    ("end recording", "stop recording", "save recording"): SequenceEvent.SAVE,
    ("play recording", "play sequence", "play"): SequenceEvent.PLAY,
    ("reset recording", "clear recording"): SequenceEvent.RESET,
    ("clear previous", "clear preivous step", "clear previous command", "clear previous event"): SequenceEvent.CLEAR_PREV,

    # Loop events
    ("loop", "start loop", "start looping"): LoopEvent.START,
    ("end loop", "stop loop", "stop looping"): LoopEvent.STOP,

    # Conditioning events
    ("if", "if case"): Condition.IF,
    ("end if", "stop if", "end if case", "stop if case"): Condition.END_IF,

    # Variable events
    ("set var", "set variable", "make var", "make variable"): VariableEvent.SET,

    # Wait For events
    ("wait for", "wait for text"): WaitForEvent.TEXT,
    ("wait for image", "wait for picture", "wait for icon"): WaitForEvent.IMAGE,

    # Misc
    ("wait", "sleep"): Timer.SLEEP,
    ("focus", "capture", "screen", "screenshot"): ScreenCaptureEvent.CAPTURE,
    ("toggle GPT", "GPT toggle", "toggle GPT on", "GPT on", "toggle GPT off", "GPT off"): ToggleGPT.TOGGLE,
}


_extra_clicks = [
    ("right", MouseButton.RIGHT),
    ("middle", MouseButton.MIDDLE),
    ("double", MouseButton.DOUBLE),
    ("shift left", MouseButton.SHIFT_LEFT),
    ("shift right", MouseButton.SHIFT_RIGHT)
]

# Lazy-loaded possible_events
def get_possible_events() -> Dict[Tuple[str, ...], object]:
    """
    Returns the full possible_events dictionary including all extra click variants.
    Computation is done only once on first call.
    """
    if not hasattr(get_possible_events, "_cached"):
        events = dict(_base_possible_events)

        for keys in list(_base_possible_events.keys()):
            key_list = list(keys)
            base_val = _base_possible_events[keys]

            if not isinstance(base_val, MouseButton):
                continue

            is_all_var = any("click all variable" in k for k in key_list)
            is_top_var = any("click variable" in k for k in key_list)
            is_spatial_var = any(" of" in k for k in key_list)
            is_image = any("image" in k or "picture" in k or "icon" in k for k in key_list)
            is_coord = any("coord" in k for k in key_list)

            if not (is_all_var or is_top_var or is_spatial_var or is_image or is_coord):
                continue

            if base_val & MouseButton.LEFT:
                for prefix, click_val in _extra_clicks:
                    new_keys = tuple(f"{prefix} {k}" for k in key_list)
                    events[new_keys] = (base_val & ~MouseButton.LEFT) | click_val

        get_possible_events._cached = events

    return get_possible_events._cached
