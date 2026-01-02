from core.state import RuntimeState
from utility.embeddings.similarity import hybrid_score
from utility.ocr.image_utils import base64_to_crop
from utility.ocr.color_processing.get_text_color import get_text_color
from utility.ocr.color_processing.color_to_text import get_color_name
from utility.text.numbers import parse_sign_number
from core.logging import get_logger
log = get_logger(__name__) 


def extract_target_context(action_span, context):
    idx = context.lower().find(action_span.lower())
    return context[idx+len(action_span):].lstrip() if idx >= 0 else ""




def extract_box_from_string_target(rs: RuntimeState, embd_lines, return_all=False):
    boost_alpha = 0.2
    color_boost = 1.15
    color_penalty = 0.85
    ctx = rs.target_text

    ctx = ctx.lower().strip()
    if not ctx:
        return None

    ctx_emb = rs.models.embd_func([ctx])
    if ctx_emb is None:
        return None
    ctx_emb = ctx_emb[0]

    if return_all:
        results = []

    best = {"score": -1, "query": ctx, "result": None}

    for line in embd_lines:
        for item in line:
            if not isinstance(item, dict): 
                log.debug(f"Item in wrong format: {item}")
            item_text = item["text"].lower()

            sim = hybrid_score(ctx, item_text, ctx_emb, item["embedding"])

            # Boost ONLY if full query matches
            if ctx == item_text:
                sim *= 1 + boost_alpha * (len(ctx.split()) - 1)

            # Color adjustments
            if sim > 0.6 and item.get("crop") and rs.color_list:
                np_crop = base64_to_crop(item["crop"])
                color_rgb = get_text_color(np_crop)
                color_text = get_color_name(color_rgb)
                sim *= color_boost if color_text in rs.color_list else color_penalty

            if return_all:
                if sim > 0.6:
                    r = {k: v for k, v in item.items() if k != "embedding"}
                    results.append({"score": sim, "query": ctx, "result": r})
            else:
                if sim > best["score"]:
                    r = {k: v for k, v in item.items() if k != "embedding"}
                    best.update({"score": sim, "query": ctx, "result": r})

    if return_all:
        return results if results else None
    else:
        return best if best["score"] > 0.6 else None
    

    

# Tiny helper for extract_box_from_numeric_target
def compare(sign, target, value):
    if sign == ">":  return value > target
    if sign == "<":  return value < target
    if sign in [">=", "=>"]: return value >= target
    if sign in ["<=", "=<"]: return value <= target
    if sign in ["=", "==", "==="]: return value == target
    return value == target

def extract_box_from_numeric_target(rs: RuntimeState, embd_lines, return_all=False):
    color_list = rs.color_list or []
    results = []

    ctx_lower = rs.target_text.lower().strip()

    # "all" context: extract everything
    if ctx_lower in ("all", "all numbers"):
        for line in embd_lines:
            for it in line:
                try:
                    val = float(it["text"].strip().split()[-1])
                except:
                    continue
                results.append({
                    "match": it["text"],
                    "value": val,
                    "color": None,
                    "bbox": it["bbox"],
                    "crop": it.get("crop")
                })

    # parse rules for numeric comparison
    parts = ctx_lower.split()
    parts = [part.strip() for part in parts]
    rules = parse_sign_number(parts[-1] if len(parts) > 1 else parts[0])
    if rules is None:
        return None

    # collect candidates that pass numeric comparison
    candidates = []
    for line in embd_lines:
        for it in line:
            try:
                val = float(it["text"].strip().split()[-1])
            except:
                continue
            if not all(compare(s, t, val) for s, t in rules):
                continue
            np_crop = base64_to_crop(it.get("crop"))
            color_text = get_color_name(get_text_color(np_crop)) if np_crop is not None else None
            candidates.append({"item": it, "val": val, "color": color_text, "bbox": it["bbox"]})

    # batch embed unique candidate colors
    unique_colors = {c["color"] for c in candidates if c["color"] and color_list}
    color_emb_map = dict(zip(unique_colors, rs.models.embd_func(list(unique_colors)))) if unique_colors else {}
    colors_emb = rs.models.embd_func(list(color_list))

    # filter by color similarity if color_list provided
    for c in candidates:
        color = c["color"]
        if color_list and color:
            vec = color_emb_map.get(color)
            if not vec:
                continue
            if max(hybrid_score(cl, color, ce, vec) for cl, ce in zip(color_list, colors_emb)) < 0.65:
                continue
        results.append({
            "match": c["item"]["text"],
            "value": c["val"],
            "color": color,
            "bbox": c["bbox"],
        })

    results.sort(key=lambda x: x['bbox'][1]) if results else None
    if return_all:
        return results or None
    return max(results, key=lambda x: x["value"]) if results else None