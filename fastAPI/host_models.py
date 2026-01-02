import os
import logging
import traceback
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uvicorn
import pickle
from PIL import Image
import io

from models.OCR.OCR import OCR
from models.qwen06voice_to_command import generate, setup_gpt_model
from models.embeddings import embed_text, get_emb_model

import io, base64
from PIL import Image
import numpy as np

def crop_to_base64(crop: np.ndarray) -> str:
    pil_img = Image.fromarray(crop)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# -------------------- setup --------------------

logging.basicConfig(level=logging.ERROR)
logging.getLogger("ppocr").setLevel(logging.WARNING)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

app = FastAPI(title="Local ML Model Host")

# load models ONCE
ocr = OCR(conf=0.4, downscale=1.0, max_workers=8, upscale=2.0, box_condense=(8,4))
gpt_model, gpt_tokenizer, get_prompt_func = setup_gpt_model()
emb_model = get_emb_model()

print("✅ Models loaded, FastAPI ready")

# -------------------- request schemas --------------------

class EmbedReq(BaseModel):
    text: list[str]

class GPTReq(BaseModel):
    input: str

class OCRReq(BaseModel):
    img: bytes
    ocr_crop_offset: tuple[int, int]

# -------------------- endpoints --------------------

@app.post("/embed")
def embed(req: EmbedReq):
    try:
        embs = embed_text(req.text, emb_model)
        if hasattr(embs, "tolist"):
            embs = embs.tolist()
        return embs
    except Exception:
        return {
            "error": "embed failed",
            "traceback": traceback.format_exc()
        }

@app.post("/gpt")
def gpt(req: GPTReq):
    try:
        return generate(req.input, gpt_model, gpt_tokenizer, get_prompt_func)
    except Exception:
        return {
            "error": "gpt failed",
            "traceback": traceback.format_exc()
        }

@app.post("/ocr")
def ocr_api(file: UploadFile = File(...), ox: int = 0, oy: int = 0):
    try:
        ocr_crop_offset = (ox, oy)
        img_bytes = file.file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        ocr_res = ocr(img, ocr_crop_offset)

        ocr_res_base64 = [
            [(b, t, crop_to_base64(crop)) for b, t, crop in line]
            for line in ocr_res
        ]
        return ocr_res_base64
    except Exception:
        import traceback
        return {
            "error": "ocr failed",
            "traceback": traceback.format_exc()
        }

# -------------------- run --------------------

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=5555,
        log_level="error"
    )
