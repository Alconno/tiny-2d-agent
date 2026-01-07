import logging
from core.logging import setup_logging, get_logger
setup_logging()
logging.getLogger().setLevel(logging.INFO)
log = get_logger(__file__)

from core.state import RuntimeState
from core.processing import process_context, parse_action_and_extract_target, extract_template
from class_models.Context import Context





def prepare_rs(rs: RuntimeState):
    # ---- GPT Processing ----
    # normalize + rewrite user intent (LLM)
    orig_ctx, ctx_processed = process_context(rs)

    if ctx_processed == "nothing":
        log.info("Command cancelled")
        rs.current_context = Context()
        return False, orig_ctx, None, ctx_processed

    # ---- Action parsing (once per context) ----
    raw_ctx = orig_ctx.strip().lower()
    if not parse_action_and_extract_target(rs, raw_ctx, ctx_processed):
        log.warning("Failed to parse action")
        return False, orig_ctx, raw_ctx, ctx_processed

    log.info(f"Action: {rs.action_event}")
    log.debug(f"Raw context: {raw_ctx}")
    log.debug(f"Processed context: {ctx_processed}")

    # ---- Template check ----
    rs.is_template = extract_template(rs.target_text)
    if rs.is_template and not rs.recording_state["active"]:
        log.warning("Template used outside recording mode")
        rs.current_context = Context()
        return False
    return True, orig_ctx, raw_ctx, ctx_processed