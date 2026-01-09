# ---- Embeddings ----
from .embeddings.event_embeddings import embd_events
from .embeddings.similarity import cosine_sim, hybrid_score, cmp_txt_and_embs
from .embeddings.text_matching import get_matching_str, extract_action

# ---- OCR ----
from .ocr.image_matching import get_target_image, find_crop_in_image
from .ocr.image_utils import base64_to_crop, image_hash, image_diff_percent
from .ocr.ocr_processing import embd_ocr_lines, filter_numbers
from .ocr.screenshot import screenshot_raw, take_screenshot, scale_screenshot_box


# ---- Text ----
from .text.normalize import normalize_word, clean_target, generate_ngrams
from .text.numbers import text_to_number, parse_delay, parse_sign_number
from .text.template_vars import extract_vars_from_contexts

# ---- Box / Book extraction ----
from .box_extraction import extract_target_context, extract_box_from_numeric_target, extract_box_from_string_target

# ---- Spatial awareness ----
from .spatial_location import get_spatial_location

# ---- Other ----
from .utils import apply_offset_to_bbox, apply_offset_to_var 