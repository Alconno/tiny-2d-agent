import math

PALETTE = {
    "black": [(0,0,0), (25,25,25), (50,50,50), (10,10,10), (35,35,35)],
    "white": [(245,245,245), (255,255,255), (230,230,230), (240,240,245), (250,250,250), (200, 200, 200), (180, 180, 180)],
    "red": [(255,0,0), (200,0,0), (180,50,50), (220,20,60), (255,80,80), (145, 0, 0), (150, 15, 15), (175, 35, 35)],
    "green": [(0,255,0), (0,200,0), (0,128,0), (34,108,32), (60,180,75), (150, 175, 105)],
    "blue": [(0,0,255), (0,0,200), (79,141,193), (50,58,108), (130,255,255), (70,130,180), (110, 100, 210)],
    "yellow": [(255,255,0), (250, 210, 0), (240,240,50), (248,255,132), (220,220,0), (163,163,44)],
    "orange": [(255,165,0), (255,140,0), (255,180,50), (255,200,100), (230,120,0)],
    "brown": [(150,75,0), (139,69,19), (160,82,45), (165,42,42), (128,64,32)],
    "gray": [(128,128,128), (75,75,75), (100,100,100), (150,150,150)],
    "purple": [(128,0,128), (160,120,200), (150,100,180), (180,130,220), (120,0,160)]
}


def srgb_to_linear(c):
    c = c / 255.0
    if c <= 0.04045: return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4

def rgb_to_xyz(r,g,b):
    r_lin, g_lin, b_lin = srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b)
    x = r_lin*0.4124564 + g_lin*0.3575761 + b_lin*0.1804375
    y = r_lin*0.2126729 + g_lin*0.7151522 + b_lin*0.0721750
    z = r_lin*0.0193339 + g_lin*0.1191920 + b_lin*0.9503041
    return x, y, z

def xyz_to_lab(x, y, z):
    xr, yr, zr = x/0.95047, y/1.0, z/1.08883
    def f(t):
        if t > 0.008856: return t**(1/3)
        return 7.787037*t + 16/116
    fx, fy, fz = f(xr), f(yr), f(zr)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return L, a, b

def rgb_to_lab(r,g,b):
    x, y, z = rgb_to_xyz(r,g,b)
    return xyz_to_lab(x,y,z)

# --- CIEDE2000 implementation ---
def ciede2000(lab1, lab2):
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    avg_L = 0.5*(L1+L2)
    C1 = math.hypot(a1,b1)
    C2 = math.hypot(a2,b2)
    avg_C = 0.5*(C1+C2)
    G = 0.5*(1 - math.sqrt((avg_C**7)/(avg_C**7+25**7)))
    a1p, a2p = (1+G)*a1, (1+G)*a2
    C1p, C2p = math.hypot(a1p,b1), math.hypot(a2p,b2)
    h1p = math.degrees(math.atan2(b1,a1p)) % 360
    h2p = math.degrees(math.atan2(b2,a2p)) % 360
    dLp = L2 - L1
    dCp = C2p - C1p
    dhp = 0
    if C1p*C2p != 0:
        dh = h2p - h1p
        if abs(dh) <= 180: dhp = dh
        elif dh > 180: dhp = dh - 360
        else: dhp = dh + 360
    dHp = 2*math.sqrt(C1p*C2p)*math.sin(math.radians(dhp/2))
    avg_Lp = (L1+L2)/2
    avg_Cp = (C1p+C2p)/2
    if C1p*C2p == 0:
        avg_hp = h1p + h2p
    else:
        dh = abs(h1p - h2p)
        if dh <= 180: avg_hp = (h1p+h2p)/2
        elif (h1p+h2p)<360: avg_hp = (h1p+h2p+360)/2
        else: avg_hp = (h1p+h2p-360)/2
    T = 1 - 0.17*math.cos(math.radians(avg_hp-30)) + 0.24*math.cos(math.radians(2*avg_hp)) + \
        0.32*math.cos(math.radians(3*avg_hp+6)) - 0.20*math.cos(math.radians(4*avg_hp-63))
    delta_ro = 30*math.exp(-((avg_hp-275)/25)**2)
    Rc = 2*math.sqrt((avg_Cp**7)/(avg_Cp**7+25**7))
    Sl = 1 + (0.015*((avg_Lp-50)**2))/math.sqrt(20+(avg_Lp-50)**2)
    Sc = 1 + 0.045*avg_Cp
    Sh = 1 + 0.015*avg_Cp*T
    Rt = -math.sin(math.radians(2*delta_ro))*Rc
    kL = kC = kH = 1
    dE = math.sqrt((dLp/(kL*Sl))**2 + (dCp/(kC*Sc))**2 + (dHp/(kH*Sh))**2 + Rt*(dCp/(kC*Sc))*(dHp/(kH*Sh)))
    return dE

PALETTE_LAB = {}
for name, rgbs in PALETTE.items():
    if not isinstance(rgbs, list):
        rgbs = [rgbs]
    PALETTE_LAB[name] = [rgb_to_lab(*rgb) for rgb in rgbs]

def match_color_name(rgb, palette_lab=PALETTE_LAB, top_n=1):
    lab = rgb_to_lab(*rgb)
    distances = []
    for name, lab_list in palette_lab.items():
        for lab_p in lab_list:
            d = ciede2000(lab, lab_p)
            distances.append((name, d))
    distances.sort(key=lambda x: x[1])
    return distances[:top_n] if top_n != 1 else distances[:1]

# --- Get color string ---
def get_color_name(rgb, palette=PALETTE_LAB):
    best_name, dist = match_color_name(rgb, palette_lab=palette, top_n=1)[0]

    # bright gray → white
    if best_name == "gray" and rgb[0]>230 and rgb[1]>230 and rgb[2]>230:
        return "white"
    return best_name