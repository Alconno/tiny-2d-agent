from events.SequenceHandler import SequenceHandler
from events.Mouse.MouseHandler import MouseHandler
from events.Keyboard import Keyboard
from events.Conditioning import ConditionHandler
from events.Variable import VariableHandler
from events.Loop import LoopHandler
from events.WaitFor import WaitForEventHandler
from events.ScreenCapture import ScreenCapture
from events.Timer import TimerHandler
from events.ToggleGPT import ToggleGPTHandler

class EventHandler():
    def __init__(self, models):
        self.sequence = SequenceHandler(models.embd_func, "sequences.json")
        self.keyboard = Keyboard(models.embd_func)
        self.mouse = MouseHandler()
        self.condition = ConditionHandler()
        self.variable = VariableHandler()
        self.loop = LoopHandler()
        self.waitfor = WaitForEventHandler()
        self.screen_capture = ScreenCapture()
        self.timer = TimerHandler()
        self.gpt_toggle = ToggleGPTHandler()
        
