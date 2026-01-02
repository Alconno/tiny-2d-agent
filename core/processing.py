import re
from core.gpt import apply_gpt_to_context
from core.state import RuntimeState
from ma_utility import extract_action, extract_target_context
from class_models.Context import Context
from core.logging import get_logger
log = get_logger(__name__) 


# Applies GPT to context if needed
# Returns original context and processed one
def process_context(rs: RuntimeState):
    ctx = rs.current_context
    if not ctx or not ctx.text:
        return "", ""

    if ctx.meta is None:
        ctx.meta = {}

    orig_ctx = ctx.text
    ctx_processed = ctx.text

    if rs.use_gpt and not ctx.meta.get("gpt_applied") and rs.models.gpt_func:
        ctx_processed = apply_gpt_to_context(rs)
        ctx.meta["gpt_applied"] = True
        ctx.text = ctx_processed

    return orig_ctx, re.sub(r"[.;!?]", "", ctx_processed).strip()



# Finding which color/s is/are mentioned in context
colors = ["black","white","red","green","blue","yellow","orange","brown","gray","purple"]
color_pattern = r"\b(" + "|".join(colors) + r")\b"
def find_colors(ctx_processed):
    found_colors = re.findall(color_pattern, ctx_processed, re.I)
    color_list = [c.lower() for c in found_colors] or None

    # Remove color string from context
    if color_list:
        ctx_processed = re.sub(color_pattern, "", ctx_processed, flags=re.I)
        ctx_processed = re.sub(r"\s+", " ", ctx_processed).strip()

    return color_list, ctx_processed



# Parses given context, extracting action and target
def parse_action_and_extract_target(rs: RuntimeState, raw_ctx: str, ctx_processed: str):
    if ctx_processed not in rs.parsed_action_cache:
        rs.color_list, ctx_processed = find_colors(ctx_processed)

        parsed_action = extract_action(ctx_processed, rs.event_embeds, rs.models.embd_func)
        if not parsed_action:
            log.debug("Could not extract action")
            rs.current_context = None
            return False

        rs.action_result = parsed_action
        rs.action_event = parsed_action["result"]
        rs.target_text = extract_target_context(parsed_action.get("span",""), ctx_processed)

        rs.parsed_action_cache[raw_ctx] = parsed_action
        rs.retry_target_cache[raw_ctx] = (rs.target_text, rs.color_list)
    else:
        parsed_action = rs.parsed_action_cache[raw_ctx]
        rs.action_event = parsed_action["result"]
        rs.target_text, rs.color_list = rs.retry_target_cache[raw_ctx]
    
    return True



# Tempalte extraction
template_pattern = r"(.+?)\s+(?:as var|variable|as variable|as a template|as a variable|template|is variable|as template|is template)$"
def extract_template(target_text: str):
    return re.match(template_pattern, target_text, re.I)




# Retries
def handle_retry(rs, ctx_processed, raw_ctx):
    rs.retries[ctx_processed] = rs.retries.get(ctx_processed, rs.n_retries) - 1
    if rs.retries[ctx_processed] > 0:
        rs.context_queue.appendleft(rs.current_context)
    else:
        rs.parsed_action_cache.pop(raw_ctx, None)
        rs.retry_target_cache.pop(raw_ctx, None)
    rs.current_context = Context()




# Map every event handler with their class
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
from class_models.Context import Context
from core.state import RuntimeState
from services.handlers import EventHandler
from models.VoiceTranscriber import VoiceTranscriber
from core.recording import append_to_recording_seq
def map_event_handlers(handlers: EventHandler, voiceTranscriber: VoiceTranscriber):
    CONTROL_EVENTS = (SequenceEvent, Condition, LoopEvent)

    def wrap_handler(handler_func):
        # wrapper applied to all handlers
        def wrapped(rs):
            failed = handler_func(rs)

            if (
                rs.recording_state.get("active", False)
                and not isinstance(rs.action_event, CONTROL_EVENTS)
            ):
                append_to_recording_seq(rs)

            return failed
        return wrapped

    return {
        MouseButton: wrap_handler(lambda rs: handlers.mouse.process_event(rs)),
        KeyboardEvent: wrap_handler(lambda rs: handlers.keyboard.execute(rs)),
        Timer: wrap_handler(lambda rs: handlers.timer.parse_delay_and_sleep(rs)),
        WaitForEvent: wrap_handler(lambda rs: handlers.waitfor.waitFor(rs)),
        ScreenCaptureEvent: wrap_handler(lambda rs: handlers.screen_capture.handle_screenshot(rs)),
        VariableEvent: wrap_handler(lambda rs: handlers.variable.handle_variable(rs)),
        LoopEvent: wrap_handler(lambda rs: handlers.loop.process(rs)),
        SequenceEvent: wrap_handler(lambda rs: handlers.sequence.process_sequence_event(rs, voiceTranscriber)),
        Condition: wrap_handler(lambda rs: handlers.condition.handle_condition(rs)),
        ToggleGPT: wrap_handler(lambda rs: handlers.gpt_toggle.handle(rs)),
    }
