import re



def normalize_word(s):
    s = s.lower()
    s = s.replace("-", "").replace(" ", "")
    s = re.sub(r"[^a-z0-9]", "", s)
    s = re.sub(r"(.)\1{2,}", r"\1", s)  # collapse loooong repeats
    return s


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

