"""Three-panel 2-D illustration of integration over the material part of one cut element (for explaining slide 3).
(1) ordinary 3x3 Gauss points ignore the boundary; (2) graded subdivision; (3) the smallest boundary box is split into
triangles, the boundary is replaced by a straight line in each triangle, and only the material side is integrated.
Usage: python3 cutfem_integration_fig.py <out.png>"""
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SS = 3                                   # supersampling
P, PAD, TOP, BOT = 520 * SS, 60 * SS, 110 * SS, 150 * SS
W, H = 3 * P + 4 * PAD, P + TOP + BOT
FONT = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
f_t, f_c = ImageFont.truetype(FONT, 30 * SS), ImageFont.truetype(FONT, 23 * SS)
INK, MAT, MAT2, AIR, EDGE, ORG, RED = '#243447', '#86A9CF', '#5B85B7', '#FFFFFF', '#243447', '#E8A33D', '#C00000'


def c(x):                                # centre line of the wall band (element-local coordinates, y up)
    return 0.28 + 0.45 * x * x
WB = 0.17


def inside(x, y):
    return np.abs(y - c(x)) <= WB


img = Image.new('RGB', (W, H), 'white')
d = ImageDraw.Draw(img)


def frame(k):
    return PAD + k * (P + PAD), TOP


def to_px(k, x, y, box=(0.0, 0.0, 1.0)):
    """Element-local (x, y) -> pixel in panel k; box = (x0, y0, size) of the region shown in the panel."""
    x0, y0 = frame(k); bx, by, bs = box
    return x0 + (x - bx) / bs * P, y0 + (1 - (y - by) / bs) * P


def paint_region(k, box=(0.0, 0.0, 1.0), col=MAT):
    x0, y0 = frame(k); bx, by, bs = box
    xs = bx + (np.arange(P) + 0.5) / P * bs
    ys = by + (1 - (np.arange(P) + 0.5) / P) * bs
    m = inside(xs[None, :], ys[:, None])
    arr = np.array(img)
    sub = arr[y0:y0 + P, x0:x0 + P]
    rgb = np.array([int(col[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
    sub[m] = rgb
    img.paste(Image.fromarray(arr))


def caption(k, title, lines):
    x0, _ = frame(k)
    d.text((x0, 30 * SS), title, font=f_t, fill=INK)
    for i, ln in enumerate(lines):
        d.text((x0, TOP + P + 22 * SS + i * 36 * SS), ln, font=f_c, fill=INK)


# ---------------- panel 1: ordinary Gauss points
paint_region(0)
d = ImageDraw.Draw(img)
d.rectangle([to_px(0, 0, 1), to_px(0, 1, 0)], outline=EDGE, width=4 * SS)
g = [0.5 - 0.5 * np.sqrt(0.6), 0.5, 0.5 + 0.5 * np.sqrt(0.6)]
for gx in g:
    for gy in g:
        px, py = to_px(0, gx, gy); r = 11 * SS
        if inside(gx, gy):
            d.ellipse([px - r, py - r, px + r, py + r], fill=INK)
        else:
            d.line([px - r, py - r, px + r, py + r], fill=RED, width=5 * SS); d.line([px - r, py + r, px + r, py - r], fill=RED, width=5 * SS)
caption(0, '① 普通做法：整块单元 3×3 个高斯点', ['蓝色是材料，白色是空气', '红叉落在空气里，材料边界附近又没有点', '→ 直接用会算错刚度'])

# ---------------- panel 2: graded subdivision
paint_region(1, col='#DCE6F2')
d = ImageDraw.Draw(img)
S = 400
def classify(x0, y0, s):
    xs = x0 + (np.arange(S) + 0.5) / S * s
    ys = y0 + (np.arange(S) + 0.5) / S * s
    m = inside(xs[None, :], ys[:, None])
    return 'full' if m.all() else ('empty' if not m.any() else 'part')
zoom = None
for i in range(4):
    for j in range(4):
        x0, y0, s = i / 4, j / 4, 0.25
        kind = classify(x0, y0, s)
        if kind == 'full':
            d.rectangle([to_px(1, x0, y0 + s), to_px(1, x0 + s, y0)], fill=MAT2)
        elif kind == 'part':
            for a in range(2):
                for b in range(2):
                    xa, yb, t = x0 + a * s / 2, y0 + b * s / 2, s / 2
                    kk = classify(xa, yb, t)
                    if kk == 'full':
                        d.rectangle([to_px(1, xa, yb + t), to_px(1, xa + t, yb)], fill=MAT2)
                    elif kk == 'part':
                        d.rectangle([to_px(1, xa, yb + t), to_px(1, xa + t, yb)], fill='#F6D9A8')
                        # a box crossed only by the lower boundary, for the zoom
                        xs = xa + (np.arange(60) + 0.5) / 60 * t
                        ys = yb + (np.arange(60) + 0.5) / 60 * t
                        frac = inside(xs[None, :], ys[:, None]).mean()
                        only_lower = (yb + t) < c(xs).min() + WB - 0.01
                        if only_lower and (zoom is None or abs(frac - 0.5) < abs(zoom[3] - 0.5)):
                            zoom = (xa, yb, t, frac)
                    d.rectangle([to_px(1, xa, yb + t), to_px(1, xa + t, yb)], outline=ORG, width=2 * SS)
        d.rectangle([to_px(1, x0, y0 + s), to_px(1, x0 + s, y0)], outline='#9AA4AE', width=2 * SS)
d.rectangle([to_px(1, 0, 1), to_px(1, 1, 0)], outline=EDGE, width=4 * SS)
if zoom:
    xa, yb, t, _ = zoom
    d.rectangle([to_px(1, xa, yb + t), to_px(1, xa + t, yb)], outline=RED, width=5 * SS)
caption(1, '② 逐级细分', ['深蓝：全是材料，按公式直接积分', '白色：全是空气，跳过', '橙色：边界小块，再对半细分；红框放大见 ③'])

# ---------------- panel 3: smallest boundary box, triangles cut by straight lines
xa, yb, t, _ = zoom
box = (xa - 0.08 * t, yb - 0.08 * t, 1.16 * t)
paint_region(2, box=box, col='#DCE6F2')         # true material, faint
d = ImageDraw.Draw(img)

def f1(x, y):                                    # lower boundary of the band: material where f1 <= 0
    return (c(x) - WB) - y

def clip(poly):
    out = []
    for k in range(len(poly)):
        p, q = poly[k], poly[(k + 1) % len(poly)]
        fp, fq = vals[p], vals[q]
        if fp <= 0:
            out.append(p)
        if (fp <= 0) != (fq <= 0):
            s_ = fp / (fp - fq)
            r = (p[0] + s_ * (q[0] - p[0]), p[1] + s_ * (q[1] - p[1]))
            vals[r] = 0.0
            out.append(r)
    return out

corners = [(xa, yb), (xa + t, yb), (xa + t, yb + t), (xa, yb + t)]
vals = {pt: f1(*pt) for pt in corners}
tris = [[corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]]
for tri in tris:
    kept = clip(tri)
    if len(kept) >= 3:
        d.polygon([to_px(2, x, y, box) for x, y in kept], fill=MAT2)
        # illustrative Gauss points: 3-point rule on each fan triangle
        for m in range(1, len(kept) - 1):
            A, B, C = np.array(kept[0]), np.array(kept[m]), np.array(kept[m + 1])
            for bc in ((2/3, 1/6, 1/6), (1/6, 2/3, 1/6), (1/6, 1/6, 2/3)):
                pt = bc[0] * A + bc[1] * B + bc[2] * C
                px, py = to_px(2, pt[0], pt[1], box); r = 9 * SS
                d.ellipse([px - r, py - r, px + r, py + r], fill='white', outline=INK, width=3 * SS)
    d.polygon([to_px(2, x, y, box) for x, y in tri], outline=ORG, width=4 * SS)
# true curved boundary, dashed
xs = np.linspace(box[0], box[0] + box[2], 400)
xs = xs[(c(xs) - WB >= box[1]) & (c(xs) - WB <= box[1] + box[2])]
pts = [to_px(2, x, c(x) - WB, box) for x in xs]
for k in range(0, len(pts) - 1, 8):
    d.line([pts[k], pts[min(k + 4, len(pts) - 1)]], fill=RED, width=4 * SS)
d.rectangle([to_px(2, xa, yb + t, box), to_px(2, xa + t, yb, box)], outline=RED, width=5 * SS)
caption(2, '③ 最小的边界小块（放大）', ['切成三角形（三维是四面体）', '每个三角形里把边界近似成直线（红虚线为真实边界）',
                                    '削掉空气一侧，剩下的部分放高斯点积分'])

img = img.resize((W // SS, H // SS), Image.LANCZOS)
img.save(sys.argv[1])
print(sys.argv[1], img.size, 'zoom box', zoom)
