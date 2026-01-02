import requests
import io, PIL
class AccessModels:
    def __init__(self):
        self.base = "http://127.0.0.1:5555"
        self.session = requests.Session()

    # ---------- helpers ----------

    def _post(self, endpoint, payload, timeout):
        # Defensive check for embed/gpt payload
        if endpoint in ("embed", "gpt"):
            if not isinstance(payload, dict):
                raise TypeError(f"Payload must be dict, got {type(payload)}")
            for key, val in payload.items():
                if isinstance(val, str):
                    continue
                elif isinstance(val, list) and all(isinstance(x, str) for x in val):
                    continue
                else:
                    raise TypeError(f"Payload value for '{key}' must be str or list[str], got {type(val)}")

        r = self.session.post(f"{self.base}/{endpoint}", json=payload, timeout=timeout)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(f"POST /{endpoint} failed ({r.status_code}): {r.text}") from e
        data = r.json()
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(data["error"] + "\n" + data.get("traceback", ""))
        return data

    # ---------- public API ----------

    def embd_func(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        elif isinstance(texts, list):
            if not all(isinstance(t, str) for t in texts):
                raise TypeError("All elements of texts must be strings")
        else:
            raise TypeError(f"embd_func expects str or list[str], got {type(texts)}")
        
        return self._post("embed", {"text": texts}, timeout=120)


    def gpt_func(self, input_text: str):
        return self._post(
            "gpt",
            {"input": input_text},
            timeout=300
        )

    def ocr_func(self, img: PIL.Image, ocr_crop_offset=(0,0)):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = self.session.post(
            f"{self.base}/ocr",
            files={"file": ("screenshot.png", buf, "image/png")},
            data={"ox": ocr_crop_offset[0], "oy": ocr_crop_offset[1]},
            timeout=120
        )
        resp.raise_for_status()
        return resp.json()