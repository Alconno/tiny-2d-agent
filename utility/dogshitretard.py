import numpy as np
from PIL import Image
import hashlib
import re
from rapidfuzz import fuzz
import jellyfish
import pyautogui
from class_models.Context import Context

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


_emb_cache = {}  # key: text + str(box), value: embedding

def embd_ocr_lines(embd_func, preds, use_color=False):
    embd_lines = []
    to_encode, idxs = [], []
    lines_structure = []

    if preds is None:
        return None

    for line in preds:
        if line is not None and len(line) > 0 and len(line[0]) != 3: 
            print("embd_ocr_lines() => received wrong preds structure")
            continue

        entries = [None] * len(line)
        for i, (box, text, text_color) in enumerate(line):
            text = f"{text_color} {text}" if use_color else text
            key = text + "_" + "_".join(f"{b:.2f}" for b in box)

            if key in _emb_cache:
                entries[i] = {"bbox": box, "text": text, "embedding": _emb_cache[key]}
            else:
                to_encode.append(text)
                idxs.append((entries, i, key, box, text))
        lines_structure.append(entries)

    if to_encode:
        embeddings = embd_func(to_encode)
        for (entries, i, key, box, text), emb in zip(idxs, embeddings):
            _emb_cache[key] = emb
            entries[i] = {"bbox": box, "text": text, "embedding": emb}

    for entries in lines_structure:
        embd_lines.append([e for e in entries if e is not None])

    return embd_lines # dict{'box':(x,y,w,h), 'text': txt or clr txt, 'embedding'}


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



def get_matching_str(input: str, cands: list, embd_func):
    # Compares embd similarity between input str and candidate strings
    # Returns best match found
    cand_embds = zip(cands, embd_func(cands))
    in_embd = embd_func(input)
    
    best = {"score": -1, "result": ""}
    for (cand, cand_embd) in cand_embds:
        sim = hybrid_score(input, cand, in_embd, cand_embd)

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



def extract_box_target(ctx, embd_lines, embd_func, color_list=None, return_all=False):
    max_n = 6
    boost_alpha = 0.2
    color_boost = 1.15
    color_penalty = 0.85

    words = ctx.lower().split()
    spans = [" ".join(words[i:i+n]) for n in range(1, max_n+1) for i in range(len(words)-n+1)]
    if not spans: return None

    span_embs = embd_func(spans)
    
    if color_list:
        color_list = [c.lower() for c in color_list]
    if return_all:
        results = []

    best = {"score": -1, "span": None, "result": None}
    for span, s_emb in zip(spans, span_embs):
        for line in embd_lines:
            for item in line:
                sim = hybrid_score(span, item["text"], s_emb, item["embedding"])

                # boost for exact span match
                if span == item["text"].lower():
                    sim *= 1 + boost_alpha * (len(span.split()) - 1)

                # color adjustments
                if color_list and item.get("color"):
                    sim *= color_boost if item["color"].lower() in color_list else color_penalty

                if return_all:
                    if sim > 0.6:
                        r = {k: v for k, v in item.items() if k != "embedding"}
                        results.append({"score": sim, "span": span, "result": r})
                else:
                    if sim > best["score"]:
                        r = {k: v for k, v in item.items() if k != "embedding"}
                        best.update({"score": sim, "span": span, "result": r})

    if return_all:
        return results if results else None
    else:
        return best if best["score"] > 0.6 else None

def extract_box_target_with_more_ctx(ctxs, embd_lines, embd_func, color_list=None, return_all=False):
    if return_all:
        all_results = []
        for cctx in ctxs:
            cand = extract_box_target(cctx, embd_lines, embd_func, color_list, return_all=True)
            if cand:
                all_results.extend(cand)
        return all_results if all_results else None
    else:
        best = None
        best_score = -1e9
        for cctx in ctxs:
            cand = extract_box_target(cctx, embd_lines, embd_func, color_list, return_all=False)
            if cand:
                score = cand.get("score", 0)
                if score > best_score:
                    best_score = score
                    best = cand
        return best




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



def extract_vars_from_steps(steps: list):
    vars_dict = {}

    def add(var_name, var_type):
        if var_name not in vars_dict:
            vars_dict[var_name] = var_type

    visited = set()
    def walk(ctx_list, loop_vars=None):
        loop_vars = loop_vars or set()
        for ctx in ctx_list:
            if isinstance(ctx, Context):
                if id(ctx) in visited: continue
                visited.add(id(ctx))
                
                # handle loop
                new_loop_vars = loop_vars.copy()
                if ctx.meta.get("loop"):
                    loop_var = ctx.meta["loop"].replace(" as template", "").strip()
                    add(loop_var, "list")
                    new_loop_vars.add(loop_var)
                # handle template vars in text
                if ctx.text:
                    for v in re.findall(r"\{\{\s*(.*?)\s*\}\}", ctx.text):
                        if v not in new_loop_vars:
                            add(v, "str")
                # recurse into sub_contexts
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


# Expands a context string that lists multiple colors joined by "or"/"and" 
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


def extract_numbers_target_with_more_ctx(ctxs, embd_lines, embd_func, color_list, return_all=False):
    results = []
    color_list = color_list or []

    for ctx in ctxs:
        ctx_lower = ctx.lower().strip()

        # If context is "all" or "all numbers", extract everything
        if ctx_lower in ("all", "all numbers"):
            for line in embd_lines:
                for it in line:
                    text = it["text"].strip()
                    bbox = it["bbox"]
                    if not text:
                        continue
                    tok = text.split()
                    try:
                        val = float(tok[-1])
                    except:
                        continue
                    color = tok[0] if color_list and len(tok) == 2 and tok[0] in color_list else None
                    results.append({
                        "match": it["text"],
                        "value": val,
                        "color": color,
                        "bbox": bbox
                    })
            continue

        parts = ctx_lower.split()
        rules = parse_sign_number(parts[-1] if len(parts) > 1 else parts[0])
        if rules is None:
            continue

        # target color strings + embeddings
        target_colors = [w for w in parts[:-1] if w in color_list]
        target_embs = embd_func(target_colors) if target_colors else []

        # gather candidate items
        items = []
        for line in embd_lines:
            for it in line:
                text = it["text"].strip()
                bbox = it["bbox"]
                if not text:
                    continue
                tok = text.split()
                color = tok[0] if color_list and len(tok) == 2 and tok[0] in color_list else None
                try:
                    val = float(tok[-1])
                except:
                    continue
                items.append((it, color, val, bbox))

        # embed unique item colors
        unique_colors = [c for c in {c for _, c, _, _ in items} if c]
        item_embs = dict(zip(unique_colors, embd_func(unique_colors))) if unique_colors else {}

        for it, item_color, val, bbox in items:
            if target_colors:
                if not item_color: continue
                item_vec = item_embs.get(item_color)
                if item_vec is None: continue
                best = max(
                    hybrid_score(tc, item_color, te, item_vec)
                    for tc, te in zip(target_colors, target_embs)
                )
                if best < 0.65:
                    continue

            # numeric comparison
            if all(compare(s, t, val) for s, t in rules):
                results.append({
                    "match": it["text"],
                    "value": val,
                    "color": item_color,
                    "bbox": bbox
                })

    # Sort asc by top Y 
    if results:
        results.sort(key=lambda x: x['bbox'][1])

    if return_all:
        return results or None
    return max(results, key=lambda x: x["value"]) if results else None



import matplotlib.pyplot as plt
import matplotlib.patches as patches


from events.Mouse import MouseButton
from utility.image_matching import calculate_edges
def get_spatial_location(spatial_event: MouseButton, bbox, offset, screenshot, spatial_search_condition="object"):
    ASA = 5 # Additional Spatial Awareness
    SAD = 350 # Spatial Awareness Distance - How far something is considered "next to" 
    x,y,w,h = bbox
    ox, oy = offset
    x -= ox
    y -= oy
    bbox = (x, y, w, h)
    orig_box = bbox

    min_height, min_width = max(5, int(h*0.1)), max(7, int(w*0.1))
    is_object_detection = spatial_search_condition == "object"
    is_text_detection = spatial_search_condition == "text"
    #print("is object detection: ", is_object_detection)

    edge_dropout = 0.1 if is_object_detection else 0.25 # forces more focus on text, but generalises objects
    proj_dropout = 0.03 if is_object_detection else 0.1
    proj_gate = 0

    def get_segments(bbox, horizontal=False):
        crop = np.array(screenshot.crop(bbox))
        x_proj, y_proj = calculate_edges(crop, 
                                        use_color=True if spatial_search_condition == "text" else False,
                                        apply_blur=True,
                                        edge_dropout=edge_dropout, 
                                        proj_dropout=proj_dropout)
        proj = x_proj if horizontal else y_proj
        diff = np.diff(np.pad((proj > proj_gate).astype(int), (1, 1)))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]
        return starts, ends
    
    while bbox == orig_box:
        if spatial_event & MouseButton.SPATIAL_ABOVE:
            starts, ends = get_segments((x-ASA, max(0, y-SAD), x+w+ASA, y))
            for y0,y1 in reversed(list(zip(starts,ends))):
                if y1 >= y: continue
                if (y1-y0) >= min_height:
                    bbox = (x, y0, w, y1 - y0)
                    break
        elif spatial_event & MouseButton.SPATIAL_BELOW:
            starts, ends = get_segments((x-ASA, y+h, x+w+ASA, min(screenshot.height-min_height, y+h+SAD)))
            for y0,y1 in zip(starts,ends):
                if y0 <= 1 or y0 > screenshot.height - min_height: continue
                if (y1-y0) >= min_height:
                    bbox = (x, y0 + (y + h), w, y1 - y0)
                    break
        elif spatial_event & MouseButton.SPATIAL_LEFT:
            crop_start_x = max(0, x - SAD)
            starts, ends = get_segments((crop_start_x, y-ASA, x, y+h+ASA), horizontal=True)
            for x0, x1 in reversed(list(zip(starts, ends))):
                if x1 >= x: 
                    continue
                if (x1 - x0) >= min_width:
                    print(x0, x1)
                    bbox = (x0 + crop_start_x, y, x1 - x0, h)
                    break
        elif spatial_event & MouseButton.SPATIAL_RIGHT:
            starts, ends = get_segments((x+w, y-ASA, min(screenshot.width, w+x+SAD), y+h+ASA), horizontal=True)
            for x0,x1 in zip(starts,ends):
                if x0 <= 1 or x0 > screenshot.width - min_width: continue
                if (x1-x0) >= min_width:
                    bbox = (x0+(x+w), y, x1-x0, h)
                    break
        edge_dropout += 0.001
        proj_dropout += 0.01
        if proj_dropout >= 0.4: break

    bx, by, bw, bh = bbox 
    return (bx + ox, by + oy, bw, bh)


