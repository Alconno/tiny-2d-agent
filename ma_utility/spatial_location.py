
from events.Mouse.MouseEvent import MouseButton
from ma_utility.ocr.image_matching import calculate_edges
import numpy as np
import matplotlib.pyplot as plt  

def get_spatial_location(spatial_event: MouseButton, bbox, offset, screenshot, spatial_search_condition="object"):
    ASA = 0 # Additional Spatial Awareness
    SAD = 450 # Spatial Awareness Distance - How far something is considered "next to" 
    x,y,w,h = bbox
    ox, oy = offset
    orig_box = bbox

    min_height, min_width = max(10, int(h*0.2)), max(15, int(w*0.2))
    is_box_detection = spatial_search_condition == "box"
    is_object_detection = spatial_search_condition == "object"
    is_text_detection = spatial_search_condition == "text"

    edge_dropout = 0.1 if is_object_detection or is_box_detection else 0.25 # forces more focus on text, but generalises objects
    proj_dropout = 0.08 if is_object_detection or is_box_detection else 0.1
    proj_gate = 0

    # Requires (left, top, right, bottom) bbox format for PIL.crop()
    def get_segments(bbox, horizontal=False, reverse=False, showedges=False):
        crop = np.array(screenshot.crop(bbox))

        x_proj, y_proj, edges = calculate_edges(
            crop, use_color=is_text_detection, apply_blur=True,
            edge_dropout=edge_dropout, proj_dropout=proj_dropout
        )
        proj = x_proj if horizontal else y_proj
        if reverse: proj = proj[::-1]

        diff = np.diff(np.pad((proj > proj_gate).astype(int), (1, 1)))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]

        if showedges:
            plt.imshow(edges, cmap="gray")
            plt.axis("off")
            plt.show()
            plt.figure(figsize=(8, 3))
            plt.plot(y_proj)
            plt.title("Y Projection")
            plt.xlabel("Index (pixels)")
            plt.ylabel("Intensity / Edge strength")
            plt.grid(True)
            plt.show()

        if reverse:
            L = len(proj)
            starts, ends = L - ends, L - starts
        return starts, ends
    
    while bbox == orig_box:
        segments_found = False
        crop_box = None
        if spatial_event & MouseButton.SPATIAL_ABOVE:
            crop_box = (max(0, x-ASA-1), max(0, y-SAD), x+w+ASA+10, y)
        elif spatial_event & MouseButton.SPATIAL_BELOW:
            crop_box = (max(0, x-ASA-1), y+h, min(screenshot.width, x+w+ASA+10), min(screenshot.height, y+h+SAD))
        elif spatial_event & MouseButton.SPATIAL_LEFT:
            crop_box = (max(0, x - SAD), y-ASA, x, y+h+ASA)
        elif spatial_event & MouseButton.SPATIAL_RIGHT:
            crop_box = (x+w, y-ASA, min(screenshot.width, x+w+SAD), y+h+ASA)

        do_reverse = spatial_event & (MouseButton.SPATIAL_ABOVE | MouseButton.SPATIAL_LEFT)
        do_horizontal = spatial_event & (MouseButton.SPATIAL_LEFT | MouseButton.SPATIAL_RIGHT)
        starts, ends = get_segments(crop_box, reverse=do_reverse, horizontal=do_horizontal)

        if not is_box_detection:
            if spatial_event & (MouseButton.SPATIAL_ABOVE | MouseButton.SPATIAL_BELOW):
                for start, end in zip(starts, ends):
                    dist, local_y = abs(end-start), start+crop_box[1]
                    if dist >= min_height:
                        bbox = (x, local_y, w, end-start)
                        segments_found = True
                        break
            else: # left right
                for start,end in zip(starts,ends):
                    dist, local_x = abs(end-start), start+crop_box[0]
                    if dist >= min_width:
                        bbox = (local_x, y, end-start, h)
                        segments_found = True
                        break

        if segments_found:
            break
        
        # increase dropout gradually
        edge_dropout += 0.001
        proj_dropout += 0.01

        # secondary strategy for "empty boxes"
        if proj_dropout >= 0.35 or is_box_detection:
            if spatial_event & (MouseButton.SPATIAL_ABOVE | MouseButton.SPATIAL_BELOW):
                for i in range(len(starts)-1):
                    start, end = ends[i], starts[i+1]
                    dist, local_y = abs(end-start), start+crop_box[1]
                    if dist > min_height:
                        bbox = (x, local_y, w, end-start)
                        segments_found = True
                        break
            else: # left, right
                for i in range(len(starts)-1):
                    start, end = ends[i], starts[i+1]
                    dist, local_x = abs(end-start), start+crop_box[0]
                    if dist > min_width:
                        bbox = (local_x, y, end-start, h)
                        segments_found = True
                        break

            if segments_found or proj_dropout >= 0.6:
                break
                
    bx, by, bw, bh = bbox 
    return bbox, (bx + ox, by + oy, bw, bh)