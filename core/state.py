from collections import deque
from class_models.Context import Context
from access_models import AccessModels
from re import Match

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
        self.prev_hash = None
        self.prev_preds = None
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

        self.action_event = None
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