import numpy as np
from PIL import Image
import hashlib
import re
from rapidfuzz import fuzz
import jellyfish
import pyautogui
from class_models.Context import Context
import  cv2
import io
import base64
from PIL import Image
from utility.get_text_color import get_text_color
from utility.color_to_text import get_color_name
from core.state import RuntimeState

def normalize_word(s):
    s = s.lower()
    s = s.replace("-", "").replace(" ", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    s = re.sub(r"(.)\1{2,}", r"\1", s)  # collapse loooong repeats
    return s

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def hybrid_score(span, item_text, span_emb, item_emb):
    a, b = normalize_word(span), normalize_word(item_text)
    cos = cosine_sim(span_emb, item_emb)
    ph = fuzz.ratio(jellyfish.metaphone(a), jellyfish.metaphone(b)) / 100
    fz = fuzz.ratio(a,b)/100
    return .45 * ph + 0.35 * fz + 0.20 * cos

def base64_to_crop(b64_str):
    img_bytes = base64.b64decode(b64_str)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    crop_array = np.array(pil_img)
    return crop_array

_emb_cache = {}  # key: text + str(box), value: embedding

def embd_ocr_lines(embd_func, preds):
    embd_lines = []
    to_encode, idxs = [], []
    lines_structure = []

    if preds is None:
        return None

    for line in preds:
        if line is None or len(line) == 0 or len(line[0]) != 3: 
            print("embd_ocr_lines() => received wrong preds structure")
            continue

        entries = [None] * len(line)
        for i, (box, text, crop) in enumerate(line):
            key = text + "_" + "_".join(f"{b:.2f}" for b in box)

            if key in _emb_cache:
                entries[i] = {"bbox": box, "text": text, "embedding": _emb_cache[key], "crop": crop}
            else:
                to_encode.append(text)
                idxs.append((entries, i, key, box, text, crop))
        lines_structure.append(entries)

    if to_encode:
        embeddings = embd_func(to_encode)
        for (entries, i, key, box, text, crop), emb in zip(idxs, embeddings):
            _emb_cache[key] = emb
            entries[i] = {"bbox": box, "text": text, "embedding": emb, "crop": crop}

    for entries in lines_structure:
        embd_lines.append([e for e in entries if e is not None])

    return embd_lines


def embd_events(embd_func, possible_events):
    event_map = [
        (phrase, action)
        for aliases, action in possible_events.items()
        for phrase in aliases
    ]

    phrases = [p for p, _ in event_map]
    embeds = embd_func(phrases)

    return [(phrase, emb, action) for (phrase, action), emb in zip(event_map, embeds)]


def generate_ngrams(words, max_n=5):
    ngrams = []
    L = len(words)
    for n in range(1, max_n+1):
        for i in range(L - n + 1):
            span = " ".join(words[i:i+n])
            ngrams.append(span)
    return ngrams

def clean_target(target: str):
    for word in ("on", "in", "at", "the"):
        if target.lower().startswith(word + " "):
            return target[len(word)+1:].strip()
    return target.strip()



def get_matching_str(ctx: str, cands: list, embd_func):
    assert isinstance(ctx, str)
    assert all(isinstance(c, str) for c in cands)

    in_embd = embd_func(ctx)
    cand_embds = zip(cands, embd_func(cands))

    best = {"score": -1, "result": ""}
    for cand, cand_embd in cand_embds:
        sim = hybrid_score(ctx, cand, in_embd, cand_embd)
        if sim > best["score"]:
            best = {"score": sim, "result": cand}
    return best["result"]


def extract_action(context_text, event_embeds, embd_func, max_n=8, boost_alpha=0.1):
    words = context_text.lower().split()

    spans = []
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            spans.append(" ".join(words[i:i+n]))

    if not spans: return None
    span_embs = embd_func(spans)

    best = {"score": -1, "span": None, "result": None}

    for span, s_emb in zip(spans, span_embs):
        for alias, alias_emb, buttons in event_embeds:
            sim = hybrid_score(span, alias, s_emb, alias_emb)
            if span.lower() == alias.lower():
                sim *= (1 + boost_alpha * (len(span.split()) - 1))
            if sim > best["score"]:
                best = {"score": sim, "span": span, "result": buttons}

    return best if best["score"] > 0.6 else None


def extract_box_target(rs: RuntimeState, embd_lines, return_all=False):
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
                print("Item: ", item)
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




def extract_target_context(action_span, context):
    idx = context.lower().find(action_span.lower())
    return context[idx+len(action_span):].lstrip() if idx >= 0 else ""



def cmp_txt_and_embs(text, emb_pairs, embd_func):
    emb1 = embd_func(text)
    best = {"score": -1, "text": None}

    for emb2, txt in emb_pairs:
        sim = cosine_sim(emb1, emb2)
        if sim > best["score"]:
            best.update({"score": sim, "text": txt})

    return best if best["score"] >= 0.9 else None




import re

WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "hundred": 100, "thousand": 1000
}

def text_to_number(text):
    tokens = text.lower().split()
    total = 0
    current = 0
    for t in tokens:
        if t in WORDS:
            val = WORDS[t]
            if val == 100 or val == 1000:
                current *= val
            else:
                current += val
        else:
            if current:
                total += current
                current = 0
    return total + current

def parse_delay(text):
    text = text.lower().strip()
    
    numeric = re.findall(r"\d+(?:\.\d+)?", text)
    if numeric:
        value = float(numeric[0])

        if "sec" in text or "s " in text:
            return int(value * 1000)
        return int(value) 
    
    num_spelled = text_to_number(text)
    if num_spelled > 0:
        if "sec" in text or "s " in text:
            return num_spelled * 1000
        return num_spelled
    
    return None



def extract_vars_from_steps(steps):
    vars_dict = {}
    visited = set()

    def add(name, typ):
        vars_dict.setdefault(name, typ)

    def walk(ctxs, loop_vars=None):
        loop_vars = loop_vars or set()

        for ctx in ctxs:
            if isinstance(ctx, Context):
                if id(ctx) in visited:
                    continue
                visited.add(id(ctx))

                new_loop_vars = loop_vars.copy()

                if ctx.text:
                    t = ctx.text.strip().lower()

                    # loop detection
                    if t.startswith(("loop", "start loop")) and "as template" in t:
                        var = t.split("as template")[0] \
                            .replace("start loop", "") \
                            .replace("loop", "") \
                            .strip()
                        add(var, "list")
                        new_loop_vars.add(var)

                    # normal template vars
                    for v in re.findall(r"\{\{\s*(.*?)\s*\}\}", ctx.text):
                        if v not in new_loop_vars:
                            add(v, "str")

                if ctx.sub_contexts:
                    walk(ctx.sub_contexts, new_loop_vars)

            elif isinstance(ctx, list):
                walk(ctx, loop_vars)

    walk(steps)
    return [{"name": k, "type": v} for k, v in vars_dict.items()]




def image_hash(img: Image.Image) -> str:
    return hashlib.md5(img.tobytes()).hexdigest()

def image_diff_percent(img1: Image.Image, img2: Image.Image) -> float:
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    if arr1.shape != arr2.shape:
        return 1.0
    diff = np.abs(arr1.astype(int) - arr2.astype(int))
    return np.mean(diff) / 255





import os
import os.path as osp

def get_target_image(embd_func, ctx, path="./clickable_images"):
    image_files = [f for f in os.listdir(path)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"))]
    base = [osp.splitext(f)[0] for f in image_files]
    base_embs = embd_func(base)
    best = cmp_txt_and_embs(ctx, zip(base_embs, base), embd_func)
    if not best:
        return None
    return osp.join(path, image_files[base.index(best["text"])])


"""# Expands a context string that lists multiple colors joined by "or"/"and" 
# into separate context strings for each color, keeping the surrounding text intact.
# Example: "click red or blue button" -> ["click red button", "click blue button"]
def expand_color_logic(target_ctx: str):
    pattern = re.compile(
        r"^(.*?\b)"                 # prefix before colors
        r"((?:\w+\s+(?:or|and)\s+)+\w+)"  # color list group
        r"\s+(\w+)"                 # (text, button, icon)
        r"(.*)$",                   # suffix after noun
        flags=re.I
    )

    m = pattern.search(target_ctx)
    if not m:
        return [target_ctx]
    prefix, color_group, noun, suffix = m.groups()

    colors = re.split(r"\s+(?:or|and)\s+", color_group, flags=re.I)

    results = []
    for color in colors:
        new_ctx = f"{prefix}{color} {noun}{suffix}".strip()
        new_ctx = re.sub(r"\s+", " ", new_ctx)
        results.append(new_ctx)

    return results
"""


import mss
def screenshot_raw():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        raw = sct.grab(monitor)
        img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
        return img

def take_screenshot(screenshot_box=None):
    screenshot = screenshot_raw().convert("RGB")
    offset = (0, 0)
    if screenshot_box:
        left, top, _, _ = screenshot_box
        offset = (left, top)
        screenshot = screenshot.crop(screenshot_box)
    return screenshot, offset




def filter_numbers(preds):
    # preds: [[(bbox, text, color), ...], ...]  (lines)
    filtered = []

    for line in preds:
        new_line = []
        if line is not None and len(line) > 0 and len(line[0]) != 3: 
            print("filter_numbers(preds) => received wrong preds structure")
            continue

        for bbox, text, color in line:
            x, y, w, h = bbox
            text = text.strip()
            if not text:
                continue

            text_len = len(text)
            char_len = (w/max(text_len+1, 1))
            if text_len == 0:
                continue

            for m in re.finditer(r"\d+", text):
                num = m.group(0)
                start, end = m.start(), m.end()
                num_len = end - start
                sub_x = x + start * char_len
                sub_w = num_len * char_len
                sub_bbox = (sub_x, y, sub_w, h)
                new_line.append((sub_bbox, text, color))

        if new_line:
            filtered.append(new_line)

    return filtered


def parse_sign_number(expr: str):
    # Parses chained numeric conditions like '>10<50' or '>=5<=20'.
    # Returns list of (sign, number) tuples.
    expr = expr.strip()
    pattern = r"(>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)"
    matches = re.findall(pattern, expr)
    
    # Case: no explicit sign, just a number like "15"
    if not matches:
        expr = expr.lstrip("=<>") # remove garbage 
        return [("==", float(expr))]
    
    return [(sign, float(num)) for sign, num in matches]

def compare(sign, target, value):
    if sign == ">":  return value > target
    if sign == "<":  return value < target
    if sign in [">=", "=>"]: return value >= target
    if sign in ["<=", "=<"]: return value <= target
    if sign in ["=", "==", "==="]: return value == target
    return value == target


def extract_numbers_target(rs: RuntimeState, embd_lines, return_all=False):
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


 

def apply_offset_to_var(offset, var):
    ox, oy = offset
    if var and var.value:
        for val in var.value:
            x, y, w, h = val['bbox']
            val['bbox'] = (x + ox, y + oy, w, h)

def apply_offset_to_bbox(offset, bbox):
    ox,oy = offset
    bbox[0] += ox
    bbox[1] += oy