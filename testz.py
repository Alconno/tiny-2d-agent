import cv2
import numpy as np
import matplotlib.pyplot as plt
from utility.color_to_text import get_color_name
from PIL import Image
import os
import hdbscan
from sklearn.neighbors import NearestNeighbors
from scipy.stats import mode

import warnings
warnings.filterwarnings(
    "ignore", 
    message=".*force_all_finite.*",
    category=FutureWarning
)

def _get_text_color(crop):
    def dominant_color(pixels, bin_size=4):
        # pixels: [N,3] LAB
        bins = (pixels // bin_size).astype(int)
        # Convert 3D bin to single integer to count frequencies
        keys = bins[:,0]*256*256 + bins[:,1]*256 + bins[:,2]
        vals, counts = np.unique(keys, return_counts=True)
        top_bin = vals[np.argmax(counts)]
        l = (top_bin // (256*256)) * bin_size
        a = ((top_bin // 256) % 256) * bin_size
        b = (top_bin % 256) * bin_size
        return np.array([l,a,b])
    
    h, w, _ = crop.shape
    pixels = crop.reshape(-1, 3).astype(np.uint8)

    N = min(len(pixels), 5000)
    idx = np.random.choice(len(pixels), N, replace=False)
    sample = pixels[idx]

    lab_sample = cv2.cvtColor(sample.reshape(-1,1,3), cv2.COLOR_RGB2LAB).reshape(-1,3)
    lab_pixels = cv2.cvtColor(pixels.reshape(-1,1,3), cv2.COLOR_RGB2LAB).reshape(-1,3)

    # KNN to give every pixel some value
    nn = NearestNeighbors(n_neighbors=1).fit(lab_sample)
    _, nearest=nn.kneighbors(lab_pixels) # [n_pix, 1] with values [0..n_samp]

    clusterer = hdbscan.HDBSCAN(min_cluster_size=60, min_samples=15).fit(lab_sample)
    labels = clusterer.labels_[nearest[:,0]]

    # Calculate Background Regions LAB
    border_thick = max(2, int(min(h, w) * 0.01))# 1% of the smaller image dimension, minimum 2 px
    border = np.concatenate([
        crop[:border_thick, :, :].reshape(-1, 3), crop[-border_thick:, :, :].reshape(-1, 3),
        crop[:, :border_thick, :].reshape(-1, 3), crop[:, -border_thick:, :].reshape(-1, 3),
    ], axis=0)
    lab_bg_pixels = cv2.cvtColor(border.reshape(-1,1,3).astype(np.uint8),cv2.COLOR_RGB2LAB).reshape(-1,3)
    bg_mean = dominant_color(lab_bg_pixels)
    bg_std  = lab_bg_pixels.std(axis=0)
    print(bg_mean)

    # Get unique clusters excluding noise
    unique_lbls = np.setdiff1d(np.unique(labels), [-1]) 
    if len(unique_lbls) == 0:
        return crop.mean(axis=(0,1)), np.array([]), np.array([])

    # Create Mask
    masks = labels[:, None] == unique_lbls # [num_px, num_clust]
    counts = masks.sum(axis=0)
    min_count = max(10, int((w * h) / 1000)) # Roughly proportional to image area
    valid = counts >= min_count
    if not np.any(valid):
        return crop.mean(axis=(0,1)), np.array([]), np.array([])

    # Calculate Cluster color means
    # [num_px, 1, 3] * [num_px, num_clust, 1] = [num_px, num_clust, 3].sum(axis=0) = [num_clust, 3]
    means = (lab_pixels[:,None,:] * masks[...,None]).sum(axis=0) / counts[:, None] 
    eps = np.array([8.0, 6.0, 6.0])  # L, A, B
    dist_from_bg = np.linalg.norm((means - bg_mean) / np.maximum(bg_std, eps), axis=1)
    
    # Calculate Score
    scores = dist_from_bg * (counts / counts.max())
    # lightly penalize clusters that are mostly on border
    label_img = labels.reshape(h,w)
    for i,lbl in enumerate(unique_lbls):
        m = label_img == lbl
        border_pixels = ((m[:border_thick,:].sum() + m[-border_thick:,:].sum() + 
                        m[:,:border_thick].sum() + m[:,-border_thick:].sum()))
        if border_pixels / counts[i] > 0.5:
            scores[i] *= 0.5
    # invalidate very tiny clusters
    scores[~valid] = -1
    best_lbl = unique_lbls[np.argmax(scores)]

    mask = (labels==best_lbl).reshape(h,w)
    text_pixels = crop[mask]
    color = text_pixels.mean(axis=0)
    ys,xs = np.where(mask)
    return color, ys, xs


pics = ['text4.png']

for pic in pics:
    if not os.path.exists(pic):
        continue
    img_pil = Image.open(pic).convert("RGB")
    img_pil_arr = np.array(img_pil).astype(np.float32)

    img = cv2.imread(pic)
    if img is None:
        continue
    img = img[..., ::-1].astype(np.float32)  # BGR -> RGB

    color, ys, xs = _get_text_color(img)
    print("Computed color (RGB 0-255):", color)

    color_norm = np.clip(color / 255.0, 0, 1)
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(img_pil_arr.astype(np.uint8))
    plt.scatter(xs, ys, color='red', s=5)
    plt.axis('off')
    plt.title("Pixels used for color")

    plt.subplot(1,2,2)
    patch = np.ones((50,50,3), dtype=np.float32) * color_norm
    plt.imshow(patch)
    plt.axis('off')
    plt.title(get_color_name(color))

    plt.show()