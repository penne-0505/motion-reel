"""15-second motion design reel — 2D scenes, compositing and post.

Timeline is locked to 120 BPM: 1 beat = 0.5 s = 30 frames @ 60 fps, 30 beats total.
  b 0–2   S1 ignition   dot → line → slate
  b 2–8   S2 type       MAKE / IT / MOVE / echo stack / zoom through the O
  b 8–14  S3 geometry   ripple grid → diamonds → flow field → stripes
  b14–20  S4 dimension  Blender render (render3d/)
  b20–24  S5 particles  stream → flow → 動 → implode / explode
  b24–30  S6 end card
"""
import math, os, sys, functools
import numpy as np
import cairo, cv2
from fontTools.ttLib import TTFont
from fontTools.pens.basePen import BasePen

W, H, FPS, BPM = 1920, 1080, 60, 120
BEAT = 60 / BPM
TOTAL = 900
ROOT = os.path.dirname(os.path.abspath(__file__))


def hexc(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))


INK, PAPER = hexc('0A0A0F'), hexc('EDEAE3')
SIGNAL, COBALT = hexc('FF4D1C'), hexc('2B4BFF')

# ───────────────────────── easing ─────────────────────────


def clamp01(x): return 0.0 if x < 0 else 1.0 if x > 1 else x
def seg(b, a, c): return clamp01((b - a) / (c - a))
def lerp(a, b, t): return a + (b - a) * t
def mixc(c1, c2, t): return tuple(lerp(a, b, t) for a, b in zip(c1, c2))
def e_out_expo(x): return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)
def e_in_expo(x): return 0.0 if x <= 0 else 2 ** (10 * x - 10)
def e_out_cubic(x): return 1 - (1 - x) ** 3
def e_in_cubic(x): return x ** 3
def e_io_cubic(x): return 4 * x ** 3 if x < .5 else 1 - (-2 * x + 2) ** 3 / 2
def e_io_quart(x): return 8 * x ** 4 if x < .5 else 1 - (-2 * x + 2) ** 4 / 2


def e_io_expo(x):
    if x <= 0: return 0.0
    if x >= 1: return 1.0
    return 2 ** (20 * x - 10) / 2 if x < .5 else (2 - 2 ** (-20 * x + 10)) / 2


def e_out_back(x, s=1.70158):
    x -= 1
    return 1 + (s + 1) * x ** 3 + s * x ** 2


def spring(x, freq=2.0, damp=7.0):
    """Damped spring 0→1 with overshoot; x in beats."""
    if x <= 0: return 0.0
    return 1 - math.exp(-damp * x) * math.cos(2 * math.pi * freq * x)

# ───────────────────────── fonts ─────────────────────────


class _Pen(BasePen):
    def __init__(self, gs):
        super().__init__(gs)
        self.contours, self.cur = [], None

    def _moveTo(self, p): self.cur = [('M', p)]
    def _lineTo(self, p): self.cur.append(('L', p))
    def _curveToOne(self, a, b, c): self.cur.append(('C', a, b, c))

    def _qCurveToOne(self, p1, p2):
        p0 = self._getCurrentPoint()
        c1 = (p0[0] + 2 / 3 * (p1[0] - p0[0]), p0[1] + 2 / 3 * (p1[1] - p0[1]))
        c2 = (p2[0] + 2 / 3 * (p1[0] - p2[0]), p2[1] + 2 / 3 * (p1[1] - p2[1]))
        self.cur.append(('C', c1, c2, p2))

    def _closePath(self):
        if self.cur: self.contours.append(self.cur)
        self.cur = None
    _endPath = _closePath


class Font:
    def __init__(self, path, number=0, variable=False):
        self.f = TTFont(path, fontNumber=number, lazy=True)
        self.upm = self.f['head'].unitsPerEm
        self.cmap = self.f.getBestCmap()
        self.variable = variable
        self.cap = self.f['OS/2'].sCapHeight or self.upm * 0.72
        self._static = None if variable else self.f.getGlyphSet()

    @functools.lru_cache(maxsize=4096)
    def glyph(self, ch, wght=900):
        gs = self.f.getGlyphSet(location={'wght': wght, 'opsz': 32}) if self.variable else self._static
        name = self.cmap[ord(ch)]
        pen = _Pen(gs)
        gs[name].draw(pen)
        return pen.contours, gs[name].width

    def layout(self, text, size, wght=900, tracking=0.0):
        """→ (list of (ch, x_px, advance_px, contours), total_width_px)."""
        s = size / self.upm
        out, x = [], 0.0
        w = int(round(wght / 10) * 10)
        for ch in text:
            cs, adv = self.glyph(ch, w)
            out.append((ch, x, adv * s, cs))
            x += adv * s + tracking
        return out, x - tracking


def path_contours(ctx, contours, x, y, s):
    for c in contours:
        for op in c:
            if op[0] == 'M': ctx.move_to(x + op[1][0] * s, y - op[1][1] * s)
            elif op[0] == 'L': ctx.line_to(x + op[1][0] * s, y - op[1][1] * s)
            else:
                (a, b), (c_, d), (e, f) = op[1], op[2], op[3]
                ctx.curve_to(x + a * s, y - b * s, x + c_ * s, y - d * s, x + e * s, y - f * s)
        ctx.close_path()


def contour_area(c):
    pts = [op[-1] for op in c]
    a = 0.0
    for (x1, y1), (x2, y2) in zip(pts, pts[1:] + pts[:1]):
        a += x1 * y2 - x2 * y1
    return a / 2


def contour_bbox(c):
    xs = [p[0] for op in c for p in op[1:]]
    ys = [p[1] for op in c for p in op[1:]]
    return min(xs), min(ys), max(xs), max(ys)


SANS = MONO = CJK = None


def load_fonts():
    global SANS, MONO, CJK
    SANS = Font('/usr/share/fonts/Adwaita/AdwaitaSans-Regular.ttf', variable=True)
    MONO = Font('/usr/share/fonts/Adwaita/AdwaitaMono-Regular.ttf')
    CJK = Font('/usr/share/fonts/noto-cjk/NotoSansCJK-Black.ttc', number=0)

# ───────────────────────── cairo helpers ─────────────────────────


def new_surface(fmt=cairo.FORMAT_RGB24):
    surf = cairo.ImageSurface(fmt, W, H)
    ctx = cairo.Context(surf)
    ctx.set_antialias(cairo.ANTIALIAS_BEST)
    return surf, ctx


def surf_rgb(surf):
    a = np.ndarray((H, W, 4), np.uint8, surf.get_data(), strides=(surf.get_stride(), 4, 1))
    return a[:, :, 2::-1].astype(np.float32) / 255.0


def surf_rgba(surf):
    a = np.ndarray((H, W, 4), np.uint8, surf.get_data(), strides=(surf.get_stride(), 4, 1))
    rgb = a[:, :, 2::-1].astype(np.float32) / 255.0
    return rgb, a[:, :, 3].astype(np.float32) / 255.0


def fill_bg(ctx, c):
    ctx.set_source_rgb(*c); ctx.paint()


def rrect(ctx, x, y, w, h, r):
    r = max(0.0, min(r, w / 2, h / 2))
    if r < 0.01:
        ctx.rectangle(x, y, w, h); return
    ctx.new_sub_path()
    ctx.arc(x + w - r, y + r, r, -math.pi / 2, 0)
    ctx.arc(x + w - r, y + h - r, r, 0, math.pi / 2)
    ctx.arc(x + r, y + h - r, r, math.pi / 2, math.pi)
    ctx.arc(x + r, y + r, r, math.pi, 3 * math.pi / 2)
    ctx.close_path()


def mono_text(ctx, text, x, y, size, color, alpha=1.0, tracking=0.0, anchor='l'):
    run, w = MONO.layout(text, size, tracking=tracking)
    s = size / MONO.upm
    if anchor == 'r': x -= w
    elif anchor == 'c': x -= w / 2
    for ch, gx, adv, cs in run:
        path_contours(ctx, cs, x + gx, y, s)
    ctx.set_source_rgba(*color, alpha); ctx.fill()
    return w


def word(ctx, text, size, wght=900, tracking=0.0):
    run, w = SANS.layout(text, size, wght, tracking)
    return run, w, size / SANS.upm, SANS.cap * size / SANS.upm

# ───────────────────────── S1 ignition (b 0–2) ─────────────────────────


def s1(ctx, b):
    fill_bg(ctx, INK)
    cx, cy = W / 2, H / 2
    if b < 1.0:
        r = 15 * e_out_back(seg(b, 0.04, 0.30), 2.6)
        pulse = 1 + 0.12 * math.exp(-((b - 0.34) / 0.05) ** 2)
        sq = e_io_cubic(seg(b, 0.42, 0.60)) * (1 - seg(b, 0.60, 0.66))
        st = e_io_expo(seg(b, 0.60, 1.0))
        w0, h0 = 2 * r * pulse * (1 + 0.45 * sq), 2 * r * pulse * (1 - 0.35 * sq)
        w, h = lerp(w0, W + 60, st), lerp(h0, 4, st)
        ctx.set_source_rgb(*PAPER)
        rrect(ctx, cx - w / 2, cy - h / 2, w, h, h / 2); ctx.fill()
        return
    # the line thickens into a paper slate
    o = e_out_expo(seg(b, 1.0, 1.32))
    hh = lerp(4, H + 8, o)
    ctx.set_source_rgb(*PAPER)
    ctx.rectangle(0, cy - hh / 2, W, hh); ctx.fill()
    # slate: typed mono title + crosshair rules
    ctx.save()
    ctx.rectangle(0, cy - hh / 2, W, hh); ctx.clip()
    title = 'MOTION DESIGN REEL'
    n = int(len(title) * seg(b, 1.16, 1.62) + 0.999)
    run, tw = MONO.layout(title, 50, tracking=12)
    x0 = cx - tw / 2
    mono_text(ctx, title[:n], x0, cy + 18, 50, INK, tracking=12)
    if n < len(title) or (b * 4) % 1 < 0.5:
        cxp = x0 + (run[n][1] if n < len(title) else tw + 10)
        ctx.set_source_rgb(*SIGNAL); ctx.rectangle(cxp + 4, cy - 26, 28, 50); ctx.fill()
    rl = e_io_expo(seg(b, 1.3, 1.85)) * (W / 2 - 180)
    ctx.set_source_rgb(*INK)
    for sgn in (-1, 1):
        ctx.rectangle(cx + sgn * (tw / 2 + 60) - (rl if sgn < 0 else 0), cy - 1, rl, 2)
    ctx.fill()
    a = seg(b, 1.45, 1.7)
    mono_text(ctx, '2026', cx, cy + 90, 22, INK, a * 0.6, tracking=6, anchor='c')
    mono_text(ctx, '15 SEC  /  120 BPM', cx, cy - 64, 22, INK, a * 0.6, tracking=6, anchor='c')
    ctx.restore()

# ───────────────────────── S2 type (b 2–8) ─────────────────────────


def draw_word_letters(ctx, run, s, x, base, per_letter):
    """per_letter(i, ch, gx, adv, contours) → draws itself."""
    for i, (ch, gx, adv, cs) in enumerate(run):
        per_letter(i, ch, x + gx, adv, cs)


def s2_make(ctx, b):
    fill_bg(ctx, SIGNAL)
    size = 440
    run, w, s, cap = word(ctx, 'MAKE', size, 900, -12)
    drift = 1 + 0.035 * (b - 2)
    ctx.save()
    ctx.translate(W / 2, H / 2); ctx.scale(drift, drift); ctx.translate(-W / 2, -H / 2)
    x0, base = W / 2 - w / 2, H / 2 + cap / 2
    ctx.save()
    ctx.rectangle(0, base - cap - 80, W, cap + 82); ctx.clip()
    for i, (ch, gx, adv, cs) in enumerate(run):
        st = 2.0 + i * 0.07
        p = e_out_expo(seg(b, st, st + 0.38))
        dy = (1 - p) * cap * 1.15
        path_contours(ctx, cs, x0 + gx, base + dy, s)
    ctx.set_source_rgb(*INK); ctx.fill()
    ctx.restore()
    u = e_io_expo(seg(b, 2.28, 2.72))
    ctx.set_source_rgb(*INK)
    ctx.rectangle(x0, base + 46, w * u, 16); ctx.fill()
    ctx.restore()


def s2_it(ctx, b):
    fill_bg(ctx, COBALT)
    wg = 100 + 800 * e_out_expo(seg(b, 3.0, 3.55))
    size = 780
    run, w, s, cap = word(ctx, 'IT', size, wg, -10)
    ang = -(1 - spring(b - 3.0, 1.7, 6.5)) * math.pi / 2
    px, base = W / 2 - w / 2, H / 2 + cap / 2
    ctx.save()
    ctx.translate(px, base); ctx.rotate(ang); ctx.translate(-px, -base)
    for ch, gx, adv, cs in run:
        path_contours(ctx, cs, px + gx, base, s)
    ctx.set_source_rgb(*PAPER); ctx.fill()
    ctx.restore()
    # tick marks sweeping, echoing the rotation
    ctx.set_source_rgba(*PAPER, 0.35)
    for k in range(13):
        a = -math.pi / 2 + k * math.pi / 24
        if a > ang + 0.02: continue
        ctx.save(); ctx.translate(px, base); ctx.rotate(a)
        ctx.rectangle(w + 60, -1.5, 40, 3); ctx.restore()
    ctx.fill()


MOVE_SIZE = 400


def _o_counter(run, s, x0, base):
    """Return inner contour of the O and its centre in screen space."""
    ch, gx, adv, cs = run[1]
    inner = min(cs, key=lambda c: abs(contour_area(c)))
    x1, y1, x2, y2 = contour_bbox(inner)
    ccx = x0 + gx + (x1 + x2) / 2 * s
    ccy = base - (y1 + y2) / 2 * s
    return inner, (ccx, ccy), (x2 - x1) * s, (y2 - y1) * s, x0 + gx


def s2_move(ctx, b, hole=False):
    """MOVE, echo stack and the zoom-through; when hole=True the O's counter is left transparent."""
    run, w, s, cap = word(ctx, 'MOVE', MOVE_SIZE, 900, -10)
    x0, base = W / 2 - w / 2, H / 2 + cap / 2
    inner, (ocx, ocy), cw, ch_, ogx = _o_counter(run, s, x0, base)

    # global transform: echo shrink, anticipation, exponential zoom into the counter
    sc_echo = lerp(1, 0.46, e_io_cubic(seg(b, 5.0, 5.3))) if b < 5.65 else lerp(0.46, 1, e_in_expo(seg(b, 5.65, 6.0)))
    if b >= 6.0: sc_echo = 1.0
    ant = lerp(1, 0.9, e_io_cubic(seg(b, 6.15, 6.6)))
    zx = seg(b, 6.6, 7.92)
    z = ant * (70 / 0.9) ** (zx ** 2.2) if zx > 0 else ant
    anchor = (lerp(ocx, W / 2, e_io_cubic(zx)), lerp(ocy, H / 2, e_io_cubic(zx)))
    zoom = z * sc_echo

    ctx.save()
    ctx.translate(*anchor); ctx.scale(zoom, zoom); ctx.translate(-ocx, -ocy)
    # background with optional counter hole
    ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
    ctx.rectangle(ocx - W * 3 / zoom - W, ocy - H * 3 / zoom - H, (W * 6) / zoom + 2 * W, (H * 6) / zoom + 2 * H)
    if hole:
        path_contours(ctx, [inner], ogx, base, s)
    ctx.set_source_rgb(*INK); ctx.fill()
    ctx.set_fill_rule(cairo.FILL_RULE_WINDING)

    def letter_tf(i, gx, adv):
        st = 4.0 + i * 0.125
        sx = spring(b - st, 2.2, 7.5) if b < 5 else 1.0
        sy = 1 / max(sx, 0.35) ** 0.35 if sx < 1 else 1 + (sx - 1) * -0.6
        return sx, sy

    # echo rows (outlines)
    rowh = cap * 1.22
    env = e_out_expo(seg(b, 5.0, 5.35)) * (1 - e_in_expo(seg(b, 5.6, 5.95))) if 5.0 <= b < 6.0 else 0
    if env > 0.001:
        ctx.set_line_width(3.2 / sc_echo)
        for k in range(-4, 5):
            if k == 0: continue
            appear = e_out_expo(seg(b, 5.0 + abs(k) * 0.05, 5.4 + abs(k) * 0.05))
            yo = k * rowh * appear * (1 - e_in_expo(seg(b, 5.6, 5.95)))
            xo = math.sin(2 * math.pi * (b - 5.0) * 1.2 + k * 0.8) * 180 * env
            for i, (ch, gx, adv, cs) in enumerate(run):
                path_contours(ctx, cs, x0 + gx + xo, base + yo, s)
                ctx.set_source_rgba(*(SIGNAL if i == 1 else PAPER), appear * env)
                ctx.stroke()
    # main word
    for i, (ch, gx, adv, cs) in enumerate(run):
        sx, sy = letter_tf(i, gx, adv)
        if sx <= 0.001: continue
        lx = x0 + gx + adv / 2
        ctx.save()
        ctx.translate(lx, base); ctx.scale(sx, sy); ctx.translate(-lx, -base)
        path_contours(ctx, cs, x0 + gx, base, s)
        ctx.restore()
        ctx.set_source_rgb(*(SIGNAL if i == 1 else PAPER)); ctx.fill()
    ctx.restore()
    # when the counter swallows the frame, S2 is gone
    return zoom * min(cw, ch_) > max(W, H) * 1.6


def s2(ctx, b):
    if b < 3: s2_make(ctx, b)
    elif b < 4: s2_it(ctx, b)
    else:
        if b >= 6.55:
            # S3 lives behind the counter, pushing forward as we fly in
            zx = seg(b, 6.6, 7.92)
            k = lerp(0.42, 1.0, e_io_cubic(zx))
            ctx.save()
            ctx.translate(W / 2, H / 2); ctx.scale(k, k); ctx.translate(-W / 2, -H / 2)
            s3(ctx, b)
            ctx.restore()
            if not s2_move_done(b):
                s2_move(ctx, b, hole=True)
        else:
            s2_move(ctx, b)


@functools.lru_cache(maxsize=None)
def _zoom_done_threshold():
    """Smallest beat at which the O's counter fully contains the frame."""
    for i in range(0, 300):
        b = 6.6 + i * 0.005
        if _move_covered(b): return b
    return 8.0


def _move_covered(b):
    run, w = SANS.layout('MOVE', MOVE_SIZE, 900, -10)
    s = MOVE_SIZE / SANS.upm
    cap = SANS.cap * s
    x0, base = W / 2 - w / 2, H / 2 + cap / 2
    inner, (ocx, ocy), cw, ch_, _ = _o_counter(run, s, x0, base)
    zx = seg(b, 6.6, 7.92)
    z = 0.9 * (70 / 0.9) ** (zx ** 2.2)
    ax, ay = lerp(ocx, W / 2, e_io_cubic(zx)), lerp(ocy, H / 2, e_io_cubic(zx))
    ea, eb = z * cw / 2 * 0.92, z * ch_ / 2 * 0.92
    return all(((cx - ax) / ea) ** 2 + ((cy - ay) / eb) ** 2 < 1 for cx in (0, W) for cy in (0, H))


def s2_move_done(b):
    return b >= _zoom_done_threshold()

# ───────────────────────── S3 geometry (b 6.5–14) ─────────────────────────


GRID_I, GRID_J, CELL = 16, 9, 120


def _angle_near(a, ref):
    """Lines are symmetric under 180° — pick the representative of a closest to ref."""
    return ref + ((a - ref + math.pi / 2) % math.pi - math.pi / 2)


def attractor(b):
    t = b - 11.0
    ax = W / 2 + 600 * math.sin(2 * math.pi * 0.33 * t) * e_out_cubic(seg(b, 11, 11.5))
    ay = H / 2 + 270 * math.sin(2 * math.pi * 0.5 * t + 0.3) * e_out_cubic(seg(b, 11, 11.5))
    back = e_io_cubic(seg(b, 12.4, 12.9))
    return lerp(ax, W / 2, back), lerp(ay, H / 2, back)


def s3(ctx, b):
    fill_bg(ctx, INK)
    cxg, cyg = (GRID_I - 1) / 2, (GRID_J - 1) / 2
    ripple_beats = [8, 9, 10, 10.5, 11]
    for j in range(GRID_J):
        for i in range(GRID_I):
            x, y = CELL / 2 + CELL * i, CELL / 2 + CELL * j
            d = math.hypot(i - cxg, j - cyg)
            P = sum(math.exp(-((b - k - d / 11) / 0.09) ** 2) for k in ripple_beats if b > k - 0.3)
            breathe = 0.5 + 0.5 * math.sin(2 * math.pi * (b * 0.5) - d * 0.6)
            # phase A: circles
            size = 24 + 6 * breathe + 62 * P
            col = mixc(PAPER, SIGNAL, clamp01(P * 1.4))
            wdt, hgt, rad, rot = size, size, size / 2, 0.0
            # phase B: → diamonds (checker)
            m = e_io_cubic(seg(b, 10 + (i + j) * 0.022, 10.45 + (i + j) * 0.022))
            if m > 0:
                sz = lerp(size, 84 + 14 * P, m)
                wdt = hgt = sz
                rad = lerp(size / 2, 3, m)
                rot = m * math.pi / 4
                cob = (i + j) % 2 == 1
                col = mixc(col, COBALT if cob else mixc(PAPER, SIGNAL, clamp01(P * 1.3)), m)
            # phase C: flow-field needles
            f = e_io_cubic(seg(b, 11 + d * 0.025, 11.4 + d * 0.025))
            if f > 0:
                ax, ay = attractor(b - math.hypot(x - W / 2, y - H / 2) / 5000)
                dpx = math.hypot(ax - x, ay - y)
                th = _angle_near(math.atan2(ay - y, ax - x), math.pi / 4)
                near = math.exp(-dpx / 330)
                wdt = lerp(wdt, 70 + 50 * near, f)
                hgt = lerp(hgt, 7, f)
                rad = lerp(rad, 3.5, f)
                rot = lerp(rot, th, f)
                col = mixc(col, mixc(PAPER, SIGNAL, near), f)
            # phase D: align horizontal, stretch into stripes, thicken to fill
            g = e_io_cubic(seg(b, 12.45 + abs(j - cyg) * 0.05 + abs(i - cxg) * 0.012, 12.95 + abs(j - cyg) * 0.05 + abs(i - cxg) * 0.012))
            if g > 0:
                rot = lerp(rot, _angle_near(0, rot), g)
                wdt = lerp(wdt, CELL + 2, g)
                col = mixc(col, PAPER, g)
                rad = lerp(rad, 0, g)
                th2 = e_in_expo(seg(b, 13.35 + abs(j - cyg) * 0.035, 13.97))
                hgt = lerp(hgt, CELL + 2, th2)
            ctx.save()
            ctx.translate(x, y); ctx.rotate(rot)
            rrect(ctx, -wdt / 2, -hgt / 2, wdt, hgt, rad)
            ctx.restore()
            ctx.set_source_rgb(*col); ctx.fill()
    # the recurring dot, now an attractor
    if 11.0 <= b < 13.0:
        ax, ay = attractor(b)
        r = 22 * e_out_back(seg(b, 11.0, 11.25), 2.2) * (1 - e_in_expo(seg(b, 12.7, 12.98)))
        ctx.set_source_rgba(*SIGNAL, 0.18); ctx.arc(ax, ay, r * 3.2, 0, 2 * math.pi); ctx.fill()
        ctx.set_source_rgb(*SIGNAL); ctx.arc(ax, ay, r, 0, 2 * math.pi); ctx.fill()

# ───────────────────────── S5 particles (b 20–24.6) ─────────────────────────


class Particles:
    def __init__(self, seed=7, pitch=11):
        rng = np.random.default_rng(seed)
        # target: 動 on a halftone grid
        surf, ctx = new_surface()
        fill_bg(ctx, (0, 0, 0))
        cs, adv = CJK.glyph('動')
        size = 880
        s = size / CJK.upm
        path_contours(ctx, cs, W / 2 - adv * s / 2, H / 2 + size * 0.38, s)
        ctx.set_source_rgb(1, 1, 1); ctx.fill()
        mask = surf_rgb(surf)[:, :, 0] > 0.5
        gy, gx = np.mgrid[pitch // 2:H:pitch, pitch // 2:W:pitch]
        inside = mask[gy, gx]
        self.tx = gx[inside].astype(np.float64) - W / 2
        self.ty = gy[inside].astype(np.float64) - H / 2
        n = self.n = len(self.tx)
        self.tz = np.zeros(n)
        # free positions: an elliptical cloud; entry from the left (continuing the whip)
        r = np.sqrt(rng.random(n)) * 1.0
        th = rng.random(n) * 2 * math.pi
        self.fx = W / 2 + np.cos(th) * r * W * 0.62
        self.fy = H / 2 + np.sin(th) * r * H * 0.55
        self.sx = self.fx - rng.uniform(700, 2800, n)
        self.ph = rng.random(n) * 2 * math.pi
        self.delay = rng.random(n)
        self.blast = rng.gamma(2.2, 650, n) + 250
        self.bang = rng.normal(0, 0.18, n)
        # 重 in paper, 力 (power) in signal, a few cobalt sparks
        col = np.tile(np.array(PAPER, np.float32), (n, 1))
        col[self.tx > 80] = SIGNAL
        col[rng.random(n) < 0.05] = (0.42, 0.52, 1.0)
        self.col = col
        self.bright = (0.75 + 0.25 * rng.random(n)).astype(np.float32)
        yy, xx = np.mgrid[-3:4, -3:4]
        k = np.clip(3.6 - np.sqrt(xx ** 2 + yy ** 2), 0, 1).astype(np.float32)
        self.kernel = k

    def positions(self, b):
        u = np.clip((b - 20.0 - self.delay * 0.2) / 0.85, 0, 1)
        st = 1 - (1 - u) ** 4
        x = self.sx + (self.fx - self.sx) * st
        y = self.fy.copy()
        # galaxy swirl: angular offset falls off with radius
        dx, dy = x - W / 2, y - H / 2
        rad = np.sqrt(dx * dx + dy * dy)
        spin = 2.4 * e_io_cubic(seg(b, 20.25, 21.9)) + 0.5 * seg(b, 20.25, 22.5)
        a = spin / (1 + rad / 380)
        ca, sa = np.cos(a), np.sin(a)
        x, y = W / 2 + dx * ca - dy * sa * 0.6, H / 2 + dx * sa / 0.6 * 0.36 + dy * ca
        # convergence onto the rotating glyph
        c = np.clip((b - 21.35 - self.delay * 0.5) / 0.72, 0, 1)
        c = np.where(c < 0.5, 4 * c ** 3, 1 - (-2 * c + 2) ** 3 / 2)
        ang = lerp(-0.5, 0.36, e_io_cubic(seg(b, 21.3, 24.0)))
        cA, sA = math.cos(ang), math.sin(ang)
        rx = self.tx * cA - self.tz * sA
        rz = self.tx * sA + self.tz * cA
        persp = 1600 / (1600 + rz)
        k = 1 - 0.3 * e_in_cubic(seg(b, 23.45, 24.0))
        jit = 1.2 * np.sin(b * 23 + self.ph * 7) * (1 - c * 0.7)
        gx = W / 2 + rx * persp * k + jit
        gy = H / 2 + self.ty * persp * k + jit * 0.7
        px = x + (gx - x) * c
        py = y + (gy - y) * c
        if b >= 24.0:
            e = e_out_expo(seg(b, 24.0, 24.9))
            dx, dy = px - W / 2, py - H / 2
            a = np.arctan2(dy, dx) + self.bang
            dist = self.blast * e
            px = px + np.cos(a) * dist
            py = py + np.sin(a) * dist
        return px, py, persp

    def splat(self, acc, b, weight):
        px, py, persp = self.positions(b)
        fx, fy = np.floor(px), np.floor(py)
        wx, wy = px - fx, py - fy
        xi, yi = fx.astype(np.int64), fy.astype(np.int64)
        ok = (xi >= 0) & (xi < W - 1) & (yi >= 0) & (yi < H - 1)
        xi, yi, wx, wy = xi[ok], yi[ok], wx[ok], wy[ok]
        br = (self.bright * np.clip(persp, 0.85, 1.2) ** 2)[ok] * weight
        idx = yi * W + xi
        dens = np.zeros(W * H)
        for off, wt in ((0, (1 - wx) * (1 - wy)), (1, wx * (1 - wy)), (W, (1 - wx) * wy), (W + 1, wx * wy)):
            for ch in range(3):
                acc[:, :, ch] += np.bincount(idx + off, self.col[ok, ch] * br * wt, W * H).reshape(H, W).astype(np.float32)

    def render(self, bs):
        acc = np.zeros((H, W, 3), np.float32)
        for b in bs: self.splat(acc, b, 1.0 / len(bs))
        return cv2.filter2D(acc, -1, self.kernel, borderType=cv2.BORDER_CONSTANT)


PARTICLES = None


def s5_image(b_list):
    """Additive particle frame averaged over sub-frames (float RGB on ink)."""
    acc = PARTICLES.render(b_list)
    glow = 1.0 + 0.6 * e_in_cubic(seg(b_list[-1], 23.45, 24.0))
    img = 1 - np.exp(-acc * 1.9 * glow)
    return np.array(INK, np.float32) * (1 - img.max(axis=2, keepdims=True)) + img


# ───────────────────────── S6 end card (b 24–30) ─────────────────────────


def dot_drop(b):
    """y offset (px, negative = up) and squash for the orange period."""
    g_t = 25.0, 25.42
    if b < g_t[0]: return None
    if b < g_t[1]:
        u = seg(b, *g_t)
        return -(1 - u ** 2) * 720, 1 + 0.35 * u ** 2   # stretch while falling
    t = b - g_t[1]
    if t < 0.34:
        u = t / 0.34
        h = 4 * u * (1 - u) * 90
        sq = 0.42 * math.exp(-t * 26) * math.cos(t * 40)
        return -h, 1 - sq
    t2 = t - 0.34
    sq = 0.2 * math.exp(-t2 * 14) * math.cos(t2 * 38)
    return 0.0, 1 - sq


def s6(ctx, b, final_pulse=True):
    fill_bg(ctx, PAPER)
    L = 120
    yl = 648
    # rules
    p = e_io_expo(seg(b, 24.12, 24.85))
    ctx.set_source_rgb(*INK)
    ctx.rectangle(L, yl, (W - 2 * L) * p, 3); ctx.fill()
    q = e_io_expo(seg(b, 24.55, 25.15))
    ctx.rectangle(1340, yl, 3, 250 * q); ctx.fill()
    # name
    size = 310
    run, w, s, cap = word(ctx, 'CLAUDE', size, 900, -10)
    base = yl - 52
    ctx.save()
    ctx.rectangle(0, 0, W, base + 4); ctx.clip()
    for i, (ch, gx, adv, cs) in enumerate(run):
        st = 24.28 + i * 0.055
        pp = e_out_expo(seg(b, st, st + 0.55))
        path_contours(ctx, cs, L - 8 + gx, base + (1 - pp) * cap * 1.2, s)
    ctx.set_source_rgb(*INK); ctx.fill()
    ctx.restore()
    # the dot returns as the full stop
    dd = dot_drop(b)
    if dd is not None:
        r = 33
        dx = L - 8 + w + 26 + r
        yo, sq = dd
        pulse = 1.0
        if final_pulse:
            for kb in (26, 27, 28, 29):
                if b >= kb: pulse += 0.07 * math.exp(-(b - kb) * 9)
        ctx.save()
        ctx.translate(dx, base + yo)
        ctx.scale(pulse / math.sqrt(max(sq, .2)), pulse * sq)
        ctx.arc(0, -r, r, 0, 2 * math.pi)
        ctx.restore()
        ctx.set_source_rgb(*SIGNAL); ctx.fill()
    # role, typed
    role = 'Motion Designer'
    n = int(len(role) * seg(b, 25.45, 26.25) + 0.999)
    run2, w2 = SANS.layout(role, 68, 560, -1)
    s2_ = 68 / SANS.upm
    for ch, gx, adv, cs in run2[:n]:
        path_contours(ctx, cs, L + gx, yl + 108, s2_)
    ctx.set_source_rgb(*INK); ctx.fill()
    if b >= 25.4:
        cx_ = L + (run2[n][1] if n < len(role) else w2 + 8)
        if n < len(role) or (b * 2) % 1 < 0.5:
            ctx.set_source_rgb(*SIGNAL); ctx.rectangle(cx_ + 4, yl + 56, 30, 60); ctx.fill()
    # disciplines
    tags = ['2D', '3D', 'TYPE', 'SYSTEMS', 'SOUND']
    x = L
    for i, t in enumerate(tags):
        a = e_out_cubic(seg(b, 26.0 + i * 0.1, 26.4 + i * 0.1))
        ww = mono_text(ctx, t, x + (1 - a) * -18, yl + 178, 24, INK, a * 0.75, tracking=4)
        x += ww + 34
        if i < len(tags) - 1:
            ctx.set_source_rgba(*SIGNAL, a); ctx.arc(x - 17, yl + 170, 3.5, 0, 2 * math.pi); ctx.fill()
    # index of what came before, top right; label above the name
    a0 = e_out_cubic(seg(b, 26.2, 26.7))
    mono_text(ctx, 'SHOWREEL  —  2026', L, 250, 22, INK, a0 * 0.6, tracking=5)
    for i, (num, name) in enumerate([('01', 'TYPE'), ('02', 'GEOMETRY'), ('03', 'DIMENSION'), ('04', 'PARTICLES')]):
        a = e_out_cubic(seg(b, 26.6 + i * 0.1, 27.0 + i * 0.1))
        yy = 172 + i * 38
        mono_text(ctx, name, W - L + (1 - a) * 16, yy, 21, INK, a * 0.85, tracking=3, anchor='r')
        mono_text(ctx, num, W - L - 250 + (1 - a) * 16, yy, 21, SIGNAL, a, tracking=3, anchor='r')
    # right column
    rows = [('REEL', '2026'), ('RUNTIME', '15.00 s'), ('TEMPO', '120 BPM'), ('MADE WITH', 'cairo · blender · numpy')]
    for i, (k, v) in enumerate(rows):
        a = e_out_cubic(seg(b, 26.35 + i * 0.12, 26.8 + i * 0.12))
        yy = yl + 52 + i * 48
        mono_text(ctx, k, 1372 + (1 - a) * -16, yy, 21, INK, a * 0.5, tracking=3)
        mono_text(ctx, v, W - L + (1 - a) * -16, yy, 21, INK, a * 0.92, tracking=1, anchor="r")

# ───────────────────────── HUD ─────────────────────────


SECTIONS = [(1.9, '01', 'TYPE'), (8.0, '02', 'GEOMETRY'), (14.0, '03', 'DIMENSION'), (20.0, '04', 'PARTICLES'), (24.0, '05', 'END')]


def hud(b, f):
    """White HUD drawn on alpha; composited with difference blend."""
    surf, ctx = new_surface(cairo.FORMAT_ARGB32)
    if not (1.95 <= b < 24.0): return None
    a = e_out_cubic(seg(b, 1.95, 2.1))
    white = (1, 1, 1)
    m = 44
    ctx.set_source_rgba(1, 1, 1, a)
    for (x, y, sx, sy) in [(m, m, 1, 1), (W - m, m, -1, 1), (m, H - m, 1, -1), (W - m, H - m, -1, -1)]:
        ctx.rectangle(x, y, 26 * sx, 2 * sy); ctx.rectangle(x, y, 2 * sx, 26 * sy)
    ctx.fill()
    mono_text(ctx, 'CLAUDE  /  MOTION REEL', m + 40, m + 22, 17, white, a * 0.9, tracking=3)
    sec, fr = divmod(f, FPS)
    mono_text(ctx, f'TC 00:00:{sec:02d}:{fr:02d}', W - m - 40, m + 22, 17, white, a * 0.9, tracking=3, anchor='r')
    # section label with a roll transition
    cur = max(i for i, s_ in enumerate(SECTIONS) if b >= s_[0])
    ctx.save()
    ctx.rectangle(m + 30, H - m - 44, 520, 34); ctx.clip()
    for i in (cur - 1, cur):
        if i < 0: continue
        r = e_io_cubic(seg(b, SECTIONS[cur][0], SECTIONS[cur][0] + 0.3))
        dy = (1 - r) * 30 if i == cur else -r * 30
        if i != cur and r >= 1: continue
        mono_text(ctx, SECTIONS[i][1], m + 40, H - m - 18 + dy, 17, white, a * 0.55, tracking=3)
        mono_text(ctx, SECTIONS[i][2], m + 90, H - m - 18 + dy, 17, white, a * 0.95, tracking=3)
    ctx.restore()
    # bar position
    beat_in_bar = int(math.floor(b)) % 4
    for k in range(4):
        x = W - m - 40 - (3 - k) * 20 - 12
        ctx.rectangle(x, H - m - 30, 12, 12)
        if k == beat_in_bar:
            ctx.set_source_rgba(1, 1, 1, a); ctx.fill()
        else:
            ctx.set_source_rgba(1, 1, 1, a * 0.9); ctx.set_line_width(1.5); ctx.stroke()
    mono_text(ctx, '120 BPM', W - m - 140, H - m - 18, 17, white, a * 0.9, tracking=3, anchor='r')
    return surf_rgba(surf)[1]

# ───────────────────────── post ─────────────────────────


IMPACTS = [(1.0, .5), (2.0, .7), (3.0, .6), (4.0, .6), (8.0, .8), (14.0, .7), (20.0, .8), (24.0, 1.6)]


def impact_env(b, rate):
    return sum(s * math.exp(-(b - k) * rate) for k, s in IMPACTS if b >= k)


def post(img, b, f):
    rng = np.random.default_rng(1000 + f)
    # camera shake
    sh = impact_env(b, 9) * 9
    if sh > 0.2:
        dx = sh * math.sin(b * 97.0)
        dy = sh * math.cos(b * 71.0)
        M = np.float32([[1.012, 0, dx - W * .006], [0, 1.012, dy - H * .006]])
        img = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    # chromatic aberration (radial)
    k = 0.0009 + impact_env(b, 7) * 0.0045
    out = img.copy()
    for ch, sgn in ((0, 1), (2, -1)):
        s = 1 + sgn * k
        M = np.float32([[s, 0, W / 2 * (1 - s)], [0, s, H / 2 * (1 - s)]])
        out[:, :, ch] = cv2.warpAffine(img[:, :, ch], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    img = out
    # bloom
    if b < 1.0: strength = 0.35
    elif b < 2.0 or 13.4 <= b < 20 or b >= 24.0: strength = 0.0
    elif b < 4.0: strength = 0.12
    elif b < 8.0: strength = 0.3
    elif b < 13.4: strength = 0.45
    else: strength = 1.0
    if strength > 0:
        lum = img.max(axis=2, keepdims=True)
        bright = img * np.clip((lum - 0.6) / 0.4, 0, 1)
        small = cv2.resize(bright, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        bl = cv2.GaussianBlur(small, (0, 0), 6) * 0.6 + cv2.GaussianBlur(small, (0, 0), 22) * 0.4
        img = img + cv2.resize(bl, (W, H), interpolation=cv2.INTER_LINEAR) * strength
    # flash on the big hit
    if 24.0 <= b < 24.45:
        fl = (1 - e_out_cubic(seg(b, 24.0, 24.42))) * 0.95
        img = img + (1 - img) * fl
    # vignette + grain
    img = img * VIGNETTE
    g = rng.normal(0, 0.016, (H // 2, W // 2)).astype(np.float32)
    g = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
    img = img + g[:, :, None]
    return np.clip(img, 0, 1)


VIGNETTE = None


def _vignette():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    return (1 - 0.09 * r)[:, :, None]

# ───────────────────────── frame assembly ─────────────────────────


def subframe_plan(b):
    if 6.55 <= b < 8.0: return 10, 1.0
    if 24.0 <= b < 24.9: return 12, 0.8
    if 20.0 <= b < 24.0: return 6, 0.6
    return 5, 0.5


def draw_2d(b):
    surf, ctx = new_surface()
    if b < 2: s1(ctx, b)
    elif b < 8: s2(ctx, b)
    elif b < 14: s3(ctx, b)
    else: s6(ctx, b)
    return surf_rgb(surf)


def particles_over_endcard(img, bs):
    """Leftover blast particles streaking across the end card (drawn as ink / signal)."""
    acc = PARTICLES.render(bs)
    fade = 1 - seg(bs[-1], 24.2, 24.9)
    a = np.clip(1 - np.exp(-acc.max(axis=2, keepdims=True) * 0.8), 0, 1) * fade
    tint = np.where(acc[:, :, :1] > acc[:, :, 1:2] * 1.6, np.array(SIGNAL, np.float32), np.array(INK, np.float32))
    return img * (1 - a) + tint * a


def render_frame(f):
    t_mid = (f + 0.5) / FPS
    b = t_mid / BEAT
    if 14 <= b < 20:
        p = os.path.join(ROOT, 'render3d', f'{f - 420 + 1:04d}.png')
        img = cv2.imread(p, cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32) / 255
    else:
        n, shutter = subframe_plan(b)
        bs = [((f + 0.5 + ((k + 0.5) / n - 0.5) * shutter) / FPS) / BEAT for k in range(n)]
        if 20 <= b < 24:
            img = s5_image(bs)
        else:
            img = sum(draw_2d(x) for x in bs) / n
            if 24 <= b < 24.9:
                img = particles_over_endcard(img, bs)
    h = hud(b, f)
    if h is not None:
        a = h[:, :, None]
        img = img * (1 - a) + np.abs(1 - img) * a
    img = post(img, b, f)
    out = (img * 255 + np.random.default_rng(f).random((H, W, 1)) * 0.999).astype(np.uint8)
    cv2.imwrite(os.path.join(ROOT, 'frames', f'{f:04d}.png'), out[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
    return f


def init_worker():
    global PARTICLES, VIGNETTE
    load_fonts()
    PARTICLES = Particles()
    VIGNETTE = _vignette()


if __name__ == '__main__':
    import multiprocessing as mp
    os.makedirs(os.path.join(ROOT, 'frames'), exist_ok=True)
    if len(sys.argv) > 1:
        frames = []
        for a in sys.argv[1:]:
            if '-' in a:
                s_, e_ = map(int, a.split('-')); frames += list(range(s_, e_))
            else: frames.append(int(a))
    else:
        frames = list(range(TOTAL))
    with mp.Pool(min(15, len(frames)), initializer=init_worker) as pool:
        for i, f in enumerate(pool.imap_unordered(render_frame, frames)):
            if i % 60 == 0: print('frame', f, flush=True)
