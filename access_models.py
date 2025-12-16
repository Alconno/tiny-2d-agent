import requests
import io
class AccessModels:
    def __init__(self):
        self.base = "http://127.0.0.1:5555"
        self.session = requests.Session()

    # ---------- helpers ----------

    def _post(self, endpoint, payload, timeout):
        r = self.session.post(
            f"{self.base}/{endpoint}",
            json=payload,
            timeout=timeout
        )
        r.raise_for_status()
        data = r.json()
        if isinstance(data, dict) and "error" in data:
            raise RuntimeError(data["error"] + "\n" + data.get("traceback", ""))
        return data

    # ---------- public API ----------

    def embd_func(self, texts):
        return self._post(
            "embed",
            {"text": texts},
            timeout=120
        )

    def gpt_func(self, input_text):
        return self._post(
            "gpt",
            {"input": input_text},
            timeout=300
        )

    def ocr_func(self, img, ocr_crop_offset=(0,0)):
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