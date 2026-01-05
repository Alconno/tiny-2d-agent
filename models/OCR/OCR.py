from PIL import Image
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from hezar.models import Model
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt, matplotlib.patches as patches
from sklearn.cluster import DBSCAN
import hashlib
import time
import cv2
from collections import deque
import warnings, os, sys


sys.path.append(".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib.pyplot as plt
import numpy as np


class OCR:
    def __init__(self, conf=0.6, tile_h=800, tile_w=800, tile_overlap=30, downscale=1.0, upscale=1.0, max_workers=4, box_condense=(4,0)):
        self.craft = Model.load("hezarai/CRAFT", device="cuda", link_threshold=0.4)
        self.ocr = PaddleOCR(
            lang='en', 
            rec_char_type='en',
            use_angle_cls=False, 
            use_textline_orientation=False,
            rec_model_dir='./models/OCR/ch_PP-OCRv3_rec_small',
            rec_algorithm='CRNN',
            det_db_box_type="quad"
        )
        self.CONF_THRESH = conf
        self.downscale = downscale
        self.upscale = upscale
        self.max_workers = max_workers
        self.box_condense = box_condense
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.tile_overlap = tile_overlap
        self.box_cache = {}  # {box_id: (last_hash, last_text)}
        

    def _hash_crop(self, crop: np.ndarray):
        return hashlib.md5(crop.tobytes()).hexdigest()

    def _process_tile(self, screenshot, left, top, right, bottom):
        tile = screenshot.crop((left, top, right, bottom))
        boxes = self.craft.predict(tile)[0]["boxes"]
        tile_w, tile_h = right-left, bottom-top
        edge = 3
        out = []
        for x,y,w,h in boxes:
            l,t,r,b = x,y,x+w,y+h
            if (t<=edge or b>=tile_h-edge):
                continue
            out.append((int(l+left), int(t+top), int(w), int(h)))
        return out


   
    def _condense_boxes(self, boxes, scale, x_limit_px=300):
        if not boxes:
            return []

        x_tol = self.box_condense[0] * scale
        y_tol = self.box_condense[1] * scale
        boxes = deque([tuple(b) for b in boxes])
        results = {b: True for b in boxes}

        while boxes:
            box0 = boxes.popleft()
            l0, t0, r0, b0 = box0[0], box0[1], box0[0]+box0[2], box0[1]+box0[3]

            for box1 in list(boxes):
                if not results.get(box1, True):
                    continue
                l1, t1, r1, b1 = box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]
                h0, h1 = box0[3], box1[3]

                x_pass = not (r1 < l0 - x_tol or r0 < l1 - x_tol)
                y_pass = abs(t1 - t0) <= y_tol and abs(h1 - h0) < min(h0, h1) * 0.5

                if x_pass and y_pass:
                    nl, nr = min(l0, l1), max(r0, r1)
                    nt, nb = min(t0, t1), max(b0, b1)
                    if nr - nl >= x_limit_px:
                        continue  # box too long
                    new_box = (nl, nt, nr - nl, nb - nt)

                    # Keep original boxes, just add the merged one
                    results[new_box] = True
                    boxes.remove(box1)
                    boxes.append(new_box)
                    break

        return [b for b, keep in results.items() if keep]
                    

    def split_text_vertically(self, crop, threshold=0.35, proj_noise=0.05,
                            edge_pad=2, visualize=False, target_px=200):

        h, w = crop.shape[:2]

        scale = max(target_px / min(h, w), 1.0)
        min_height = max(8, int(h * 0.05 * scale))

       # Resize and blur
        new_w, new_h = int(w * scale), int(h * scale)
        up = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        blur = cv2.GaussianBlur(up, (15,15), sigmaX=35.0)
        up = cv2.addWeighted(up, 1.6, blur, -.5, 0)
        crop_up = cv2.bilateralFilter(up, d=1, sigmaColor=50, sigmaSpace=50)

        # Compute gradient mag
        gray = cv2.cvtColor(crop_up, cv2.COLOR_RGB2GRAY)
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
        mag = cv2.magnitude(gx, gy)
        mag /= mag.max() + 1e-6
        edges = mag > threshold

        proj = edges.sum(axis=1)
        proj_clean = np.where(proj >= proj.max() * proj_noise, proj, 0)

        # Detect continuous segments as boxes
        diff = np.diff(np.pad((proj_clean > 0).astype(int), (1, 1)))
        starts, ends = np.where(diff == 1)[0], np.where(diff == -1)[0]

        boxes = []
        pad = int(round(edge_pad * scale))

        for y0, y1 in zip(starts, ends):
            if y1 - y0 < min_height:
                continue
            y0s = max(0, y0 - pad)
            y1s = min(new_h, y1 + pad)
            if y1s <= y0s:
                continue
            yo = int(round(y0s / scale))
            yb = int(round(y1s / scale))
            ho = yb - yo
            if ho <= 1 or yo < 0 or yo + ho > h:
                continue
            boxes.append((0, yo, w, ho))

        if visualize and boxes:
            plt.imshow(crop)
            for x, y, bw, bh in boxes:
                plt.gca().add_patch(plt.Rectangle((x, y), bw, bh, fill=False))
            plt.show()

        return boxes


    def __call__(self, screenshot: Image, ocr_crop_offset = (0,0)):
        W, H = screenshot.size 
        scale = 1 / self.downscale if self.downscale != 1.0 else self.upscale if self.upscale > 1.0 else 1.0
        screenshot = screenshot.resize((int(W*scale), int(H*scale)), Image.BILINEAR)
        W, H = screenshot.size 

        all_boxes = [] 
        tiles = [
            (left, top, min(left + self.tile_w, W), min(top + self.tile_h, H))
            for top in range(0, H, max(1, self.tile_h - self.tile_overlap))
            for left in range(0, W, max(1, self.tile_w - self.tile_overlap))]
        with ThreadPoolExecutor(max_workers=self.max_workers) as exe: 
            results = exe.map(lambda args: self._process_tile(screenshot, *args), tiles)
        
        for res in results: 
            all_boxes.extend(res) # [x,y,w,h]

        boxes = self._condense_boxes(all_boxes, scale)
        final_boxes = []

        # ---- Vertical line splitting ----
        for x, y, w, h in boxes:
            if w <= 1 or h <= 1: continue
            r, b = int(x+w), int(y+h)
            if r <= int(x) or b <= int(y): continue

            crop = np.array(screenshot.crop((int(x), int(y), r, b)))
            sub_boxes = self.split_text_vertically(crop, visualize=False)
            if not sub_boxes:
                final_boxes.append((x, y, w, h))
                continue
            for x0, y0, w0, h0 in sub_boxes:
                final_boxes.append((x + x0, y + y0, w0, h0))
        crops = [np.array(screenshot.crop((int(x), int(y), int(x+w), int(y+h)))) 
                for x, y, w, h in final_boxes]

        # ---- Processing crops ----
        def process_crop(args):
            box, crop = args
            hsh = self._hash_crop(crop)
            if hsh in self.box_cache:
                return self.box_cache[hsh]
            
            result = self.ocr.ocr(crop, det=False, cls=False)
            if not result or not result[0] or not result[0][0]:
                return "", crop
            text, conf = result[0][0][0].strip(), result[0][0][1]

            if conf < self.CONF_THRESH:
                text = ""

            self.box_cache[hsh] = (text, crop)
            return text, crop

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            processed_crops = list(ex.map(process_crop, zip(final_boxes, crops)))

        # ---- post-process filtering ----
        filtered = [(box, text, crop)
                    for box, (text, crop) in zip(final_boxes, processed_crops)
                    if text and text.strip() != ""]
        if not filtered: 
            return []

        boxes, texts, crops_out = zip(*filtered)

        boxes = list(boxes)
        texts = list(texts)
        crops_out = list(crops_out)

        # Restore scaling
        boxes = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for x, y, w, h in boxes]

        # Sort and group lines
        items = sorted(zip(boxes, texts, crops_out), key=lambda x: (x[0][1]+x[0][3]//2, x[0][0]))
        Y_lines, cur, prev_y = [], [], items[0][0][1]+items[0][0][3]//2
        for b, t, c in items:
            y_mid = b[1]+b[3]//2
            if abs(y_mid-prev_y) > b[3]*0.4:
                Y_lines.append(cur)
                cur = []
            cur.append((b, t, c))
            prev_y = y_mid
        if cur: Y_lines.append(cur)

        final_lines = []
        for line in Y_lines:
            line = sorted(line, key=lambda x: x[0][0])
            cur, prev_x_end = [], line[0][0][0]+line[0][0][2]
            for b, t, c in line:
                x_start, x_end = b[0], b[0]+b[2]
                if x_start-prev_x_end > 80:
                    final_lines.append(cur)
                    cur = []
                cur.append((b, t, c))
                prev_x_end = x_end
            if cur: final_lines.append(cur)

        # Fix coordinates for cropped screenshots
        if ocr_crop_offset and ocr_crop_offset != (0,0):
            ox, oy = ocr_crop_offset
            new = []
            for line in final_lines:
                new_line = []
                for b, t, c in line:
                    new_b = (b[0] + ox, b[1] + oy, b[2], b[3])
                    new_line.append((new_b, t, c))
                new.append(new_line)
            final_lines = new

        return final_lines