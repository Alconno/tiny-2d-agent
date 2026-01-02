import base64,io
from PIL import Image
import numpy as np 
import hashlib

def base64_to_crop(b64_str):
    img_bytes = base64.b64decode(b64_str)
    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    crop_array = np.array(pil_img)
    return crop_array

def image_hash(img: Image.Image) -> str:
    return hashlib.md5(img.tobytes()).hexdigest()

def image_diff_percent(img1: Image.Image, img2: Image.Image) -> float:
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    if arr1.shape != arr2.shape:
        return 1.0
    diff = np.abs(arr1.astype(int) - arr2.astype(int))
    return np.mean(diff) / 255
