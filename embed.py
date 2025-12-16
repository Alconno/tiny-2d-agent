import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from sentence_transformers import SentenceTransformer

STModel_name = "sentence-transformers/all-mpnet-base-v2"

def get_emb_model():
    print(f"---- Hosting embedding model: {STModel_name} ----")
    return SentenceTransformer(STModel_name)

def embed_text(texts, _model):
    print("EMBEDDING CALL...")
    res = _model.encode(texts, batch_size=32, normalize_embeddings=True)
    return res
