
from events.Mouse.MouseEvent import MouseButton
from utility.ocr.image_matching import calculate_edges
import numpy as np

def get_spatial_location(spatial_event: MouseButton, bbox, offset, screenshot, spatial_search_condition="object"):
    ASA = 5 # Additional Spatial Awareness
    SAD = 350 # Spatial Awareness Distance - How far something is considered "next to" 
    x,y,w,h = bbox
    ox, oy = offset
    orig_box = bbox

    min_height, min_width = max(5, int(h*0.1)), max(7, int(w*0.1))
    is_object_detection = spatial_search_condition == "object"
    is_text_detection = spatial_search_condition == "text"
    #print("is object detection: ", is_object_detection)

    edge_dropout = 0.1 if is_object_detection else 0.25 # forces more focus on text, but generalises objects
    proj_dropout = 0.03 if is_object_detection else 0.1
    proj_gate = 0

    # Requires (left, top, right, bottom) bbox format for PIL.crop()
    def get_segments(bbox, horizontal=False, reverse=False):
        crop = np.array(screenshot.crop(bbox))

        x_proj, y_proj = calculate_edges(
            crop, use_color=is_text_detection, apply_blur=True,
            edge_dropout=edge_dropout, proj_dropout=proj_dropout
        )
        proj = x_proj if horizontal else y_proj
        if reverse: proj = proj[::-1]

        diff = np.diff(np.pad((proj > proj_gate).astype(int), (1, 1)))
        starts = np.where(diff == 1)[0]
        ends   = np.where(diff == -1)[0]

        if reverse:
            L = len(proj)
            starts, ends = L - ends, L - starts
        return starts, ends
    
    while bbox == orig_box:
        if spatial_event & MouseButton.SPATIAL_ABOVE:
            crop_left, crop_top, crop_right, crop_bottom = max(0, x-ASA-1), max(0, y-SAD), x+w+ASA+10, y
            starts, ends = get_segments((crop_left, crop_top, crop_right, crop_bottom), reverse=True)
            for y0, y1 in zip(starts, ends):
                abs_y0, abs_y1 = y0 + crop_top, y1 + crop_top
                if (abs_y1 - abs_y0) >= min_height:
                    bbox = (x, abs_y0, w, abs_y1 - abs_y0)
                    break
        elif spatial_event & MouseButton.SPATIAL_BELOW:
            starts, ends = get_segments((max(0, x-ASA-1), y+h, 
                                         min(screenshot.width, x+w+ASA+10), min(screenshot.height, y+h+SAD)))
            for y0, y1 in zip(starts, ends):
                local_y = y0 + (int(y + h))
                if local_y <= 1 or local_y > screenshot.height - min_height:
                    continue
                if (y1 - y0) >= min_height:
                    bbox = (x, local_y, w, y1 - y0)
                    break
        elif spatial_event & MouseButton.SPATIAL_LEFT:
            crop_start_x = max(0, x - SAD)
            starts, ends = get_segments((crop_start_x, y-ASA, x, y+h+ASA), horizontal=True, reverse=True)
            for x0, x1 in zip(starts, ends):
                if (x1 - x0) >= min_width:
                    bbox = (x0 + crop_start_x, y, x1 - x0, h)
                    break
        elif spatial_event & MouseButton.SPATIAL_RIGHT:
            starts, ends = get_segments((x+w, y-ASA, min(screenshot.width, x+w+SAD), y+h+ASA), horizontal=True)
            for x0,x1 in zip(starts,ends):
                if x0 <= 1 or x0 > screenshot.width - min_width: continue
                if (x1-x0) >= min_width:
                    bbox = (x0+(x+w), y, x1-x0, h)
                    break
        edge_dropout += 0.001
        proj_dropout += 0.01
        if proj_dropout >= 0.4: break

    bx, by, bw, bh = bbox 
    return (bx + ox, by + oy, bw, bh)