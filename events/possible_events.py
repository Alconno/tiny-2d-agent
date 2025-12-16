import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from events.Mouse import MouseButton
from events.Keyboard import KeyboardEvent
from events.SequenceProcessor import SequenceEvent
from events.Timer import Timer
from events.ScreenCapture import ScreenCaptureEvent
from events.Conditioning import Condition
from events.Variable import VariableEvent
from events.Loop import LoopEvent
from events.ToggleGPT import ToggleGPT
from events.WaitFor import WaitForEvent

possible_events = {
    # Click events
    ("click", "left click"): MouseButton.LEFT,
    ("right click",): MouseButton.RIGHT,
    ("middle click",): MouseButton.MIDDLE,
    ("double click", "open"): MouseButton.LEFT | MouseButton.DOUBLE,

    # Shift click events
    ("shift click", "shift and click", "shift left click"): MouseButton.SHIFT_LEFT,
    ("shift right", "shift right click"): MouseButton.SHIFT_RIGHT,

    # Image click events
    ("click image", "click on image", "find", "select image", "image click", "click icon", "click picture"): MouseButton.IMAGE,
    
    # Variable click events
    ("click all variable", "click every variable"): MouseButton.VAR_ALL | MouseButton.LEFT,
    ("click variable", "click one variable","click top variable", "click best variable"): MouseButton.VAR_TOP | MouseButton.LEFT,

    # Spatial awareness click events
    ("click left of", "click left to"): MouseButton.SPATIAL_LEFT | MouseButton.LEFT,
    ("click right of", "click right to"): MouseButton.SPATIAL_RIGHT | MouseButton.LEFT,
    ("click above of", "click on top of", "click above"): MouseButton.SPATIAL_ABOVE | MouseButton.LEFT,
    ("click below of", "click under", "click below"): MouseButton.SPATIAL_BELOW | MouseButton.LEFT,

    # Kb events
    ("write", "type"): KeyboardEvent.WRITE,
    ("press",): KeyboardEvent.PRESS,

    # Sequence events
    ("start sequence recording", "start recording sequence", "record sequence", "start recording"): SequenceEvent.START,
    ("end recording", "stop recording", "save recording"): SequenceEvent.SAVE,
    ("play recording", "play sequence", "play"): SequenceEvent.PLAY,

    # Loop events
    ("start loop", "start looping"): LoopEvent.START,
    ("end loop", "stop loop", "stop looping"): LoopEvent.STOP,

    # Conditioning events
    ("if", "if case"): Condition.IF,
    ("end if", "stop if", "end if case", "stop if case"): Condition.END_IF,

    # Variable events
    ("set var", "set variable", "make var", "make variable"): VariableEvent.SET,

    # Wait For
    ("wait for", "wait for text"): WaitForEvent.TEXT,
    ("wait for image", "wait for picture", "wait for icon"): WaitForEvent.IMAGE,

    # Misc
    ("wait", "sleep"): Timer.SLEEP,
    ("focus", "capture", "screen", "screenshot"): ScreenCaptureEvent.CAPTURE,
    ("toggle GPT", "GPT toggle"): ToggleGPT.TOGGLE,
}                     

extra_clicks = [
    ("right", MouseButton.RIGHT),
    ("middle", MouseButton.MIDDLE),
    ("double", MouseButton.DOUBLE),
    ("shift left", MouseButton.SHIFT_LEFT),
    ("shift right", MouseButton.SHIFT_RIGHT)
]

for keys in list(possible_events.keys()):
    key_list = list(keys)
    base_val = possible_events[keys]
    is_all_var = any("click all variable" in k for k in key_list)
    is_top_var = any("click variable" in k for k in key_list)
    is_spatial_var = any(" of" in k for k in key_list)

    if not (is_all_var or is_top_var or is_spatial_var):
        continue

    if base_val & MouseButton.LEFT:
        for prefix, click_val in extra_clicks:
            # Build a tuple of variations, only prefix first
            new_keys = tuple(f"{prefix} {k}" for k in key_list)
            possible_events[new_keys] = (base_val & ~MouseButton.LEFT) | click_val
