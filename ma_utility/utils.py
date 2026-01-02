def apply_offset_to_var(offset, var):
    ox, oy = offset
    if var and var.value:
        for val in var.value:
            x, y, w, h = val['bbox']
            val['bbox'] = (x + ox, y + oy, w, h)

def apply_offset_to_bbox(offset, bbox):
    ox,oy = offset
    bbox[0] += ox
    bbox[1] += oy