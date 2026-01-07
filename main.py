# ---- Standard / Third-party ----
import os
import time
import tkinter
import logging

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
from services.voice import build_voice_listener

# ---- Helpers / Utilities ----
from core.processing import handle_retry, map_event_handlers
from core.main import prepare_rs

# ---- Models & Handlers ----
log.info("> Model connections and handlers")
models = build_models() # gpt, embd, ocr
handlers: EventHandler = EventHandler(models) # Handlers for all events from 'possible_events.py'
voiceTranscriber, context_queue = build_voice_listener()

# ---- Runtime vars ----
log.info("> Runtime state")
rs = RuntimeState(context_queue)
rs.event_embeds = embd_events(models.embd_func, get_possible_events())
rs.models = models
rs.handlers = handlers

EVENT_HANDLERS_MAP = map_event_handlers(handlers, voiceTranscriber)

# ---- Main loop ----
log.info("Main loop started, hold F8 to say your commands!\n\n")    
while True:
    # --- Fetch new context ---
    rs.fetch_next_context()

    if isinstance(rs.current_context, str):
        rs.current_context = Context(text=rs.current_context)

    if not rs.current_context or not rs.current_context.text:
        time.sleep(0.1)
        continue

    # ---- Preprocess ----
    out = prepare_rs(rs)
    #rs.print_state(show_queue=True)
    if out is False:
        continue

    res, orig_ctx, raw_ctx, ctx_processed = out
    
    # ---- Event processing ----
    failed = True
    handler = EVENT_HANDLERS_MAP.get(type(rs.action_event))
    if not handler:
        log.error("No handler for event %s", rs.action_event)
        failed = True
    else:
        failed, data = handler(rs)

    # ---- Retry logic ----
    if failed and rs.action_event and rs.target_text:
        log.warning("Handler failed, retrying...")
        handle_retry(rs, ctx_processed, raw_ctx)
        time.sleep(1)
        continue


    # ---- Clear caches on success ----
    rs.current_context = Context()
    time.sleep(0.01)
