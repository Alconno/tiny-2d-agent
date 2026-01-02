import cv2
import numpy as np
from PIL import Image
import pyautogui
import matplotlib.pyplot as plt   
import os
import os.path as osp
from utility.embeddings.similarity import cmp_txt_and_embs

def bbox_to_transformed(x, y, w, h):
    return np.float32([
        [ [x,     y]     ],
        [ [x,     y + h] ],
        [ [x + w, y + h] ],
        [ [x + w, y]     ]
    ])

def SIFT_search(img, crop, min_match_count=4):
    gray_img  = cv2.GaussianBlur(cv2.cvtColor(img,  cv2.COLOR_BGR2GRAY), (5,5), 0)
    gray_crop = cv2.GaussianBlur(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (5,5), 0)

    # SIFT features
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray_crop, None)
    kp2, des2 = sift.detectAndCompute(gray_img,  None)
    if des1 is None or des2 is None:
        return None, None

    # FLANN matcher
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Ratio+abs dist threshold
    ratio = 0.80
    abs_thresh = 200
    good = [m for m,n in matches if m.distance < ratio*n.distance and m.distance < abs_thresh]
    if len(good) < min_match_count:
        return None, None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    # Homography
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 10.0)
    if M is None or mask.sum() < min_match_count:
        return None, None

    # Transform crop corners
    h, w = gray_crop.shape
    corners = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(4,1,2)
    transformed = cv2.perspectiveTransform(corners, M)

    xs = transformed[:,0,0]
    ys = transformed[:,0,1]
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)
    bbox = (x1, y1, x2 - x1, y2 - y1)
    return bbox, transformed


def calculate_edges(crop, use_color=False, apply_blur=False, blur_sigma=35.0, edge_dropout=0.15, proj_dropout=0.05):
    #cv2.imshow("Crop", crop)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    if apply_blur:
        blur = cv2.GaussianBlur(crop, (15,15), sigmaX=blur_sigma)
        crop = cv2.addWeighted(crop, 1.6, blur, -.5, 0)
    crop = cv2.bilateralFilter(crop, d=1, sigmaColor=50, sigmaSpace=50)
    if use_color:
        crop = crop.astype(np.float32)
        gx = np.zeros(crop.shape[:2], np.float32)
        gy = np.zeros_like(gx)
        for c in range(3):
            gx += cv2.Scharr(crop[..., c], cv2.CV_32F, 1, 0) ** 2
            gy += cv2.Scharr(crop[..., c], cv2.CV_32F, 0, 1) ** 2
    else:
        gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
        gx = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    mag = cv2.magnitude(gx,gy)
    _, edges = cv2.threshold(mag, edge_dropout * mag.max(), 1, cv2.THRESH_BINARY)
    #plt.imshow(edges, cmap="gray")
    #plt.axis("off")
    #plt.show()
    x_proj = edges.sum(axis=0)
    y_proj = edges.sum(axis=1)
    x_proj = np.where(x_proj >= x_proj.max() * proj_dropout, x_proj, 0)
    y_proj = np.where(y_proj >= y_proj.max() * proj_dropout, y_proj, 0)
    return x_proj, y_proj

def projection_score(template_proj, matched_proj):
    t = template_proj.flatten()
    m = matched_proj.flatten()
    return np.dot(t, m) / (np.linalg.norm(t) * np.linalg.norm(m) + 1e-8)
    
def template_match(img, crop, max_scale_variation=1.25, thresh=0.75):
    H, W = img.shape[:2]
    h0, w0 = crop.shape[:2]
    best_score = -1
    best_box = None
    scales = np.linspace(1/max_scale_variation, max_scale_variation, 20)

    for s in scales:
        w, h = int(w0*s), int(h0*s)
        if w <= 0 or h <= 0 or w > W or h > H:
            continue

        t_scaled = cv2.resize(crop, (w, h), interpolation=cv2.INTER_NEAREST)
        res_b = cv2.matchTemplate(img[:,:,0], t_scaled[:,:,0], cv2.TM_CCOEFF_NORMED)
        res_g = cv2.matchTemplate(img[:,:,1], t_scaled[:,:,1], cv2.TM_CCOEFF_NORMED)
        res_r = cv2.matchTemplate(img[:,:,2], t_scaled[:,:,2], cv2.TM_CCOEFF_NORMED)
        res = (res_b + res_g + res_r) / 3
        _, score, _, loc = cv2.minMaxLoc(res)

        x, y = loc
        x_proj, y_proj = calculate_edges(img[y:y+h, x:x+w])
        res_x_proj, res_y_proj = calculate_edges(t_scaled)
        proj_score_x = projection_score(res_x_proj, x_proj)
        proj_score_y = projection_score(res_y_proj, y_proj)

        proj_score = (proj_score_x + proj_score_y) / 2
        score += proj_score*0.1
        """ if proj_score>0.8:
            fig,axs=plt.subplots(1,2, figsize=(12,4))
            axs[0].plot(x_proj)
            axs[0].set_title("x_proj")
            axs[1].plot(res_x_proj)
            axs[1].set_title("res_x_proj")
            for ax in axs:
                ax.set_xlabel("index")
                ax.set_ylabel("Mag")
            plt.tight_layout()
            plt.show()"""
        if score > best_score:
            best_score = score
            best_box = (loc[0], loc[1], w, h)

    if best_score < thresh or best_box is None:
        return None, None

    x, y, w, h = best_box
    transformed = bbox_to_transformed(x, y, w, h)
    return best_box, transformed



def find_crop_in_image(screenshot, crop_path, min_match_count=4, return_new_img=True, offset=None):
    if isinstance(screenshot, Image.Image):
        img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    else:
        print("Wrong image format")
        return None, None

    crop = Image.open(crop_path).convert("RGB")
    img = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
    crop = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
    print(img.shape, crop.shape)
    if img is None or crop is None:
        raise FileNotFoundError("Could not read one of the images")

    bbox, transformed = SIFT_search(img, crop, min_match_count)
    if bbox == None:
        print("running multiscale")
        bbox, transformed = template_match(img, crop)

    if offset is not None and bbox is not None:
        ox, oy = offset
        x, y, w, h = bbox
        bbox = (x + ox, y + oy, w, h)

    if return_new_img:
        out = img.copy()
        if transformed is not None:
            cv2.polylines(out, [np.int32(transformed)], True, (0,255,0), 2, cv2.LINE_AA)
        return out, bbox
    return None, bbox


def get_target_image(embd_func, ctx, path="./clickable_images"):
    image_files = [f for f in os.listdir(path)
                   if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"))]
    base = [osp.splitext(f)[0] for f in image_files]
    base_embs = embd_func(base)
    best = cmp_txt_and_embs(ctx, zip(base_embs, base), embd_func)
    if not best:
        return None
    return osp.join(path, image_files[base.index(best["text"])])



if __name__ == "__main__":
    screenshot = pyautogui.screenshot().convert("RGB")
    img_out, box = find_crop_in_image(screenshot, "./clickable_images/mibombo.png", return_new_img=True)
    if img_out is not None:
        cv2.imshow("Match", img_out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
