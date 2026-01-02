from utility.embeddings.similarity import hybrid_score

ACTION_SCORE_THRESH = 0.6

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

    return best if best["score"] > ACTION_SCORE_THRESH else None