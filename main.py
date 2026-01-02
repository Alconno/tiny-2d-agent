# ---- Standard / Third-party ----
import tkinter
import os, time,re

# ---- Local / project modules ----
from utility import (embd_events)
from events.possible_events import possible_events
from class_models.Context import Context
from core.state import RuntimeState
from services.models import build_models
from services.handlers import EventHandler
from services.voice import build_voice_listener

# Helpers
from core.processing import process_context, parse_action_and_extract_target,  \
                            extract_template, handle_retry, map_event_handlers

# ---- Logging / environment ----
import logging
logging.getLogger("ppocr").setLevel(logging.WARNING)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tkinter.NoDefaultRoot()

# ---- Models & Handlers ----
models = build_models() # gpt, embd, ocr
handlers: EventHandler = EventHandler(models) # Handlers for all events from 'possible_events.py'
voiceTranscriber, context_queue = build_voice_listener()

# ---- Runtime vars ----
rs = RuntimeState(context_queue)
rs.event_embeds = embd_events(models.embd_func, possible_events)
rs.models = models
rs.handlers = handlers

EVENT_HANDLERS_MAP = map_event_handlers(handlers, voiceTranscriber)

# ---- Main loop ----
print("\n\nMain loop started, hold F8 to say your commands!\n\n")    
while True:
    # --- Fetch new context ---
    rs.fetch_next_context()

    if isinstance(rs.current_context, str):
        rs.current_context = Context(text=rs.current_context)

    if not rs.current_context or not rs.current_context.text:
        time.sleep(0.1)
        continue

    # ---- GPT Processing ----
    # normalize + rewrite user intent (LLM)
    orig_ctx, ctx_processed = process_context(rs)

    if ctx_processed == "nothing":
        print("Canceled command.")
        rs.current_context = Context()
        continue

    # ---- Action parsing (once per context) ----
    raw_ctx = orig_ctx.strip().lower()
    if not parse_action_and_extract_target(rs, raw_ctx, ctx_processed):
        continue

    print("----------------------------------------------------------------------------")
    print(">>>>>>>>>> orig context:", raw_ctx)
    print(">>>>>>>>>> proc context:", ctx_processed)
    print(">>>>>>>>>> action:", rs.action_event)


    # ---- Template check ----
    rs.is_template = extract_template(rs.target_text)
    if rs.is_template and not rs.recording_state["active"]:
        print("You cannot use templates while NOT recording")
        rs.current_context = Context()
        continue
    

    # ---- Event processing ----
    failed = True
    handler = EVENT_HANDLERS_MAP.get(type(rs.action_event))
    if not handler:
        print("No handler for event:", rs.action_event)
        failed = True
    else:
        failed = handler(rs)

    # ---- Retry logic ----
    if failed:
        handle_retry(rs, ctx_processed, raw_ctx)
        time.sleep(1)
        continue


    # ---- Clear caches on success ----
    rs.current_context = Context()
    time.sleep(0.2)
