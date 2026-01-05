# ---- Standard / Third-party ----
import os
import time
import tkinter
import logging
from collections import deque

# ---- Environment ----
logging.getLogger("ppocr").setLevel(logging.WARNING)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["NUMEXPR_MAX_THREADS"] = "16"
tkinter.NoDefaultRoot()

# ---- Logging setup ----
from core.logging import setup_logging, get_logger
setup_logging()
logging.getLogger().setLevel(logging.INFO)
log = get_logger("main")
log.info("Setting up main: ")

# ---- Local / project modules ----
log.info("> Project modules...")
from ma_utility import embd_events
from events.possible_events import get_possible_events
from class_models.Context import Context
from core.state import RuntimeState
from services.models import build_models
from services.handlers import EventHandler

# ---- Helpers / Utilities ----
from core.processing import handle_retry, map_event_handlers
from core.main import prepare_rs

# ---- Models & Handlers ----
log.info("> Model connections and handlers")
models = build_models() # gpt, embd, ocr
handlers: EventHandler = EventHandler(models) # Handlers for all events from 'possible_events.py'
context_queue = deque()

# ---- Runtime vars ----
log.info("> Runtime state")
rs = RuntimeState(context_queue)
rs.event_embeds = embd_events(models.embd_func, get_possible_events())
rs.models = models
rs.handlers = handlers

EVENT_HANDLERS_MAP = map_event_handlers(handlers)

class API():
    def __init__(self):
        pass

    def __call__(self, input: str = ""):
        # Ensure current_context exists
        if input:
            rs.current_context = Context(input)
        elif not getattr(rs, "current_context", None):
            rs.current_context = Context("")

        # ---- Preprocess ----
        res, orig_ctx, raw_ctx, ctx_processed = prepare_rs(rs)
        if not res:
            return None, None

        # ---- Event processing ----
        failed = True
        handler = EVENT_HANDLERS_MAP.get(type(rs.action_event))
        if not handler:
            log.error("No handler for event %s", rs.action_event)
        else:
            failed, data = handler(rs)

        # ---- Retry logic ----
        if failed:
            log.warning("Handler failed, retrying...")
            handle_retry(rs, ctx_processed, raw_ctx)
            if rs.retries.get(ctx_processed, 0) > 0:
                time.sleep(1)
                failed, data = self(input)

        return failed, data

