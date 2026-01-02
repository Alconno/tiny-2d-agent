import math
import numpy as np
import cv2
import hdbscan
from sklearn.neighbors import NearestNeighbors
from skimage.measure import regionprops, label

import warnings
warnings.filterwarnings(
    "ignore", 
    message=".*force_all_finite.*",
    category=FutureWarning
)

def get_text_color(crop):
    def dominant_color(pixels, bin_size=8):
        """
        Return the most frequent color in LAB pixels using coarse binning.
        """
        bins = (pixels // bin_size).astype(int)
        keys = bins[:,0]*256*256 + bins[:,1]*256 + bins[:,2]
        vals, counts = np.unique(keys, return_counts=True)
        top_bin = vals[np.argmax(counts)]
        l = (top_bin // (256*256)) * bin_size
        a = ((top_bin // 256) % 256) * bin_size
        b = (top_bin % 256) * bin_size
        return np.array([l, a, b])

    h, w, _ = crop.shape
    pixels = crop.reshape(-1, 3).astype(np.uint8)
    N = min(len(pixels), 15000)
    idx = np.random.choice(len(pixels), N, replace=False)
    sample = pixels[idx]

    lab_sample = cv2.cvtColor(sample.reshape(-1,1,3), cv2.COLOR_RGB2LAB).reshape(-1,3)
    lab_pixels = cv2.cvtColor(pixels.reshape(-1,1,3), cv2.COLOR_RGB2LAB).reshape(-1,3)

    # KNN to propagate labels to all pixels 
    nn = NearestNeighbors(n_neighbors=1).fit(lab_sample)

    # HDBSCAN clustering 
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(10, N // 50),
        min_samples=max(5, N // 100)
    ).fit(lab_sample)

    _, nearest = nn.kneighbors(lab_pixels)
    labels = clusterer.labels_[nearest[:,0]]

    # Background estimation
    border_thick = max(3, int(min(h, w) * 0.04))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:border_thick,:] = True
    border_mask[-border_thick:,:] = True
    border_mask[:,:border_thick] = True
    border_mask[:,-border_thick:] = True
    border_pixels = crop[border_mask].reshape(-1,3)
    lab_bg_pixels = cv2.cvtColor(border_pixels.reshape(-1,1,3).astype(np.uint8), cv2.COLOR_RGB2LAB).reshape(-1,3)
    bg_mean = dominant_color(lab_bg_pixels)
    bg_std  = np.maximum(lab_bg_pixels.std(axis=0), np.array([5.0,5.0,5.0]))

    # Prepare clusters
    unique_lbls = np.setdiff1d(np.unique(labels), [-1])
    if len(unique_lbls) == 0:
        return crop.mean(axis=(0,1)), np.array([]), np.array([])

    masks = labels[:, None] == unique_lbls
    counts = masks.sum(axis=0)
    min_count = max(10, (w*h)//1000)
    valid = counts >= min_count
    if not np.any(valid):
        return crop.mean(axis=(0,1)), np.array([]), np.array([])

    # Compute cluster means
    means = (lab_pixels[:,None,:] * masks[...,None]).sum(axis=0) / counts[:, None]
    dist_from_bg = np.linalg.norm((means - bg_mean) / np.maximum(bg_std, np.array([8.0,6.0,6.0])), axis=1)

    # Compute smooth border penalty
    label_img = labels.reshape(h,w)
    border_fraction = np.array([
        ((label_img==lbl) & border_mask).sum() / counts[i]
        for i, lbl in enumerate(unique_lbls)
    ])
    border_penalty = np.clip(1.0 - border_fraction, 0.3, 1.0)

    # Apply variance penalty
    cluster_var = np.array([lab_pixels[masks[:,i]].var(axis=0).sum() for i in range(len(unique_lbls))])
    var_penalty = 1.0 / (1.0 + cluster_var/50.0)

    # Compute solidity per cluster
    solidity = np.zeros(len(unique_lbls))
    for i, lbl in enumerate(unique_lbls):
        m = (label_img == lbl).astype(np.uint8)
        l_props = regionprops(label(m))
        if len(l_props) > 0:
            solidity[i] = l_props[0].area / max(l_props[0].convex_area, 1)
        else:
            solidity[i] = 0.0

    # Cap size influence
    size_weight = np.minimum(counts / counts.max(), 0.5)

    # Final score
    scores = dist_from_bg * size_weight * border_penalty * var_penalty * solidity
    scores[~valid] = -1
    
    best_lbl = unique_lbls[np.argmax(scores)]
    mask = (labels == best_lbl).reshape(h,w)
    text_pixels = crop[mask]
    color = text_pixels.mean(axis=0)
    return color
                