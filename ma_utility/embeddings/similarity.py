import numpy as np
from rapidfuzz import fuzz
import jellyfish
from ma_utility.text.normalize import normalize_word

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def hybrid_score(span, item_text, span_emb, item_emb):
    a, b = normalize_word(span), normalize_word(item_text)
    cos = cosine_sim(span_emb, item_emb)
    ph = fuzz.ratio(jellyfish.metaphone(a), jellyfish.metaphone(b)) / 100
    fz = fuzz.ratio(a,b)/100
    return .45 * ph + 0.35 * fz + 0.20 * cos


def cmp_txt_and_embs(text, emb_pairs, embd_func):
    emb1 = embd_func(text)
    best = {"score": -1, "text": None}

    for emb2, txt in emb_pairs:
        sim = cosine_sim(emb1, emb2)
        if sim > best["score"]:
            best.update({"score": sim, "text": txt})

    return best if best["score"] >= 0.9 else None