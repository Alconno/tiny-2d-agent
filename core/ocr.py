from ma_utility import embd_ocr_lines, filter_numbers, image_diff_percent, image_hash
from core.state import RuntimeState

def run_ocr(screenshot, offset, rs: RuntimeState, number_only=False):
    diff = image_diff_percent(screenshot, rs.prev_screenshot) if rs.prev_hash else 1e8
    if diff < rs.screenshot_sim_threshold:
        return rs.prev_embd_lines
    preds = rs.models.ocr_func(screenshot, offset)
    if number_only:
        preds = filter_numbers(preds)
    embd_lines = embd_ocr_lines(rs.models.embd_func, preds)
    rs.prev_screenshot, rs.prev_preds, rs.prev_embd_lines, rs.prev_hash = (
        screenshot, preds, embd_lines, image_hash(screenshot)
    )
    return embd_lines
 