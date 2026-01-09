from collections import deque
from class_models.Context import Context
from fastAPI.access_models import AccessModels
from re import Match
import pyautogui


class RuntimeState:
    def __init__(self, context_queue = None):
        self.current_context: Context = Context("", 4)
        self.context_queue = context_queue
        self.recording_state = {"active": False, "contexts": [], "name": ""}
        self.recording_stack = [self.recording_state["contexts"]]
        self.variables = {}
        self.retries = {}
        self.n_retries = 3
        self.use_gpt = False
        self.screenshot_box = None
        self.desktopWH = pyautogui.size()
        self.prev_hash = None
        self.prev_embd_lines = None
        self.prev_screenshot = None
        self.screenshot_sim_threshold = 0.001
        self.parsed_action_cache = {}      # orig_ctx: parsed action
        self.retry_target_cache = {}       # orig_ctx: (target_text, found_colors)
        self.event_embeds = None
        
        self.models: AccessModels = None
        self.handlers = None
        
        self.color_list = None
        self.is_template: Match[str] | None = None

        self.action_result = None # Whole action dict
        self.action_event = None # Only action event class
        self.target_text = None
        

    def fetch_next_context(self):
        ctx = self.current_context
        if not ctx or not getattr(ctx, "text", None):
            if self.context_queue:
                ctx = self.context_queue.popleft()
            else:
                return None
        if isinstance(ctx, str) or ctx is None:
            ctx = Context()
        self.current_context = ctx
        return ctx
    
    def print_state(self, show_queue=False):
        print("=== RuntimeState ===")
        print("Current Context:", getattr(self.current_context, "text", None))
        print("Screenshot Box:", self.screenshot_box)
        print("Prev Embedding Lines:", self.prev_embd_lines)
        print("Prev Screenshot:", "Yes" if self.prev_screenshot else "No")
        print("Action Result:", self.action_result)
        print("Action Event:", self.action_event)
        print("Target Text:", self.target_text)
        print("Retries:", self.retries)
        print("Variables:", self.variables)
        print("Use GPT:", self.use_gpt)
        print("Parsed Action Cache Keys:", list(self.parsed_action_cache.keys()))
        print("Retry Target Cache Keys:", list(self.retry_target_cache.keys()))
        print("Color List:", self.color_list)
        print("Is Template Match:", self.is_template)
        if show_queue:
            print("Context Queue:", [getattr(c, "text", str(c)) for c in self.context_queue])
        print("===================")