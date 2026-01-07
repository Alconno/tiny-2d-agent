import re
from core.logging import get_logger
log = get_logger(__name__) 



_emb_cache = {}  # key: text + str(box), value: embedding
def embd_ocr_lines(embd_func, preds):
    embd_lines = []
    to_encode, idxs = [], []
    lines_structure = []

    if preds is None:
        return None

    for line in preds:
        if line is None or len(line) == 0 or len(line[0]) != 3: 
            log.debug("embd_ocr_lines() => received wrong preds structure")
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
            entries[i] = {"bbox": list(box), "text": text, "embedding": emb, "crop": crop}

    for entries in lines_structure:
        embd_lines.append([e for e in entries if e is not None])

    return embd_lines




def filter_numbers(preds):
    # preds: [[(bbox, text, color), ...], ...]  (lines)
    filtered = []

    for line in preds:
        new_line = []
        if line is not None and len(line) > 0 and len(line[0]) != 3: 
            log.debug("filter_numbers(preds) => received wrong preds structure")
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