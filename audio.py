"""Soundtrack for the reel — every sound synthesised with numpy, placed on the same beat grid as the picture.

120 BPM, F minor. Hits are keyed to picture events (cuts, landings, typing, the blast).
Writes soundtrack.wav (48 kHz, 16-bit stereo, 15.0 s).
"""
import math, os, wave
import numpy as np

SR = 48000
DUR = 15.0
N = int(SR * DUR)
ROOT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(3)


def T(b): return b * 0.5          # beats → seconds
def hz(m): return 440 * 2 ** ((m - 69) / 12)   # midi → Hz


dry = np.zeros((2, N + SR * 3))
send = np.zeros((2, N + SR * 3))    # reverb send
duck_src = np.zeros(N + SR * 3)     # kick envelope for sidechain
duckable = np.zeros((2, N + SR * 3))


def place(sig, b, gain=1.0, pan=0.0, rev=0.0, bus='dry'):
    i = int(T(b) * SR)
    if i >= dry.shape[1]: return
    sig = sig[:dry.shape[1] - i]
    l, r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    tgt = duckable if bus == 'duck' else dry
    tgt[0, i:i + len(sig)] += sig * gain * l * 1.414
    tgt[1, i:i + len(sig)] += sig * gain * r * 1.414
    if rev:
        send[0, i:i + len(sig)] += sig * gain * rev * l
        send[1, i:i + len(sig)] += sig * gain * rev * r


def tt(d): return np.arange(int(d * SR)) / SR


def band(x, lo, hi, soft=0.15):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    m = np.clip((f - lo) / (lo * soft + 1), 0, 1) * np.clip((hi - f) / (hi * soft + 1), 0, 1)
    return np.fft.irfft(X * m, len(x))


def noise(d): return rng.standard_normal(int(d * SR))


def sweep(d, f0, f1, q=0.5, curve=2.0):
    """Noise through a band that moves from f0 to f1 (overlap-add STFT filter)."""
    n = int(d * SR); x = rng.standard_normal(n + 4096)
    win, hop = np.hanning(4096), 1024
    out = np.zeros(n + 4096)
    freqs = np.fft.rfftfreq(4096, 1 / SR)
    for s in range(0, n, hop):
        u = (s / n) ** curve
        fc = f0 * (f1 / f0) ** u
        m = np.exp(-0.5 * (np.log(np.maximum(freqs, 1) / fc) / q) ** 2)
        out[s:s + 4096] += np.fft.irfft(np.fft.rfft(x[s:s + 4096] * win) * m) * win
    return out[:n] / 1.5

# ───────── instruments ─────────


def kick(d=0.45, punch=1.0):
    t = tt(d)
    f = 44 + 120 * np.exp(-t * 30)
    ph = 2 * np.pi * np.cumsum(f) / SR
    s = np.sin(ph) * np.exp(-t * 6.5) * punch
    s[:150] += band(noise(150 / SR), 2000, 9000) * np.linspace(1, 0, 150) * 0.5
    return np.tanh(s * 1.6)


def boom(d=2.6):
    t = tt(d)
    f = 30 + 45 * np.exp(-t * 5)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 1.3)


def clap(d=0.35):
    t = tt(d)
    env = np.exp(-t * 20)
    for k in (0.0, 0.009, 0.019):
        env += np.where(t >= k, np.exp(-(t - k) * 140), 0) * 0.8
    s = band(noise(d), 900, 6000) * env
    s += np.sin(2 * np.pi * 185 * t) * np.exp(-t * 35) * 0.4
    return s * 0.6


def hat(open_=False):
    d = 0.35 if open_ else 0.06
    t = tt(d)
    return band(noise(d), 7000, 18000) * np.exp(-t * (11 if open_ else 70)) * 0.5


def crash(d=3.0):
    t = tt(d)
    return band(noise(d), 3500, 17000) * np.exp(-t * 1.5) * 0.5


def saw(f, t, nh=24, bright=8.0, detune=(0.0,)):
    s = np.zeros_like(t)
    for dc in detune:
        ff = f * 2 ** (dc / 1200)
        ph0 = rng.random() * 6.28
        for k in range(1, nh + 1):
            if ff * k > 16000: break
            s += np.sin(2 * np.pi * ff * k * t + ph0 * k) / k * math.exp(-k / bright)
    return s / len(detune)


def bass_note(m, d=0.22, bright=5.0):
    t = tt(d)
    env = np.minimum(t / 0.004, 1) * np.exp(-t * 7)
    br = bright * (0.4 + 0.6 * np.exp(-t * 12))
    s = saw(hz(m), t, 18, bright) * env
    s += np.sin(2 * np.pi * hz(m) * t) * env * 0.8
    return s


def pad(chord, d, attack=0.35, bright=6.0):
    t = tt(d)
    env = np.minimum(t / attack, 1) * np.minimum(1, (d - t) / 0.4)
    s = sum(saw(hz(m), t, 14, bright, (-9, 0, 8)) for m in chord)
    return s * env / len(chord)


def pluck(m, d=0.3, bright=1.0):
    t = tt(d)
    f = hz(m)
    s = np.sin(2 * np.pi * f * t) + 0.35 * np.sin(4 * np.pi * f * t) * bright + 0.12 * np.sin(6 * np.pi * f * t) * bright
    return s * np.exp(-t * 16) * np.minimum(t / 0.002, 1)


def blip(f, d=0.12):
    t = tt(d)
    ff = f * (1 + 0.5 * np.exp(-t * 60))
    return np.sin(2 * np.pi * np.cumsum(ff) / SR) * np.exp(-t * 30)


def tick(f=3200, d=0.025):
    t = tt(d)
    return (np.sin(2 * np.pi * f * t) * 0.6 + band(noise(d), 2500, 9000) * 0.5) * np.exp(-t * 220)


def thud(f0=130, d=0.4):
    t = tt(d)
    f = f0 * (0.45 + 0.55 * np.exp(-t * 20))
    s = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 11)
    s[:400] += band(noise(400 / SR), 200, 3000) * np.linspace(0.6, 0, 400)
    return np.tanh(s * 1.4)


def pop(d=0.12):
    t = tt(d)
    f = 520 + 700 * np.exp(-t * 55)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 28)


def reverse(sig): return sig[::-1].copy()

# ───────── arrangement ─────────


# chords (midi), one per bar of 4 beats starting at b2
Fm = [53, 56, 60, 63]          # F3 Ab3 C4 Eb4
Db = [49, 53, 56, 60]          # Db3 F3 Ab3 C4
Ab = [56, 60, 63, 67]          # Ab3 C4 Eb4 G4
Eb = [51, 55, 58, 62]          # Eb3 G3 Bb3 D4
Fm9 = [41, 53, 56, 60, 63, 67]
PROG = [(2, Fm), (6, Db), (10, Ab), (14, Eb), (18, Fm), (22, Db)]
ROOTS = {2: 41, 6: 37, 10: 44, 14: 39, 18: 41, 22: 37}


def chord_at(b):
    c = Fm
    for s, ch in PROG:
        if b >= s: c = ch
    return c


def root_at(b):
    r = 41
    for s, m in ROOTS.items():
        if b >= s: r = m
    return r


def kicks():
    beats = [0]
    beats += list(range(2, 7))
    beats += list(range(8, 14))
    beats += list(range(14, 20))
    beats += [22, 22.5, 23, 23.25, 23.5]
    for b in beats:
        place(kick(punch=1.3 if b in (0, 8, 14) else 1.0), b, 0.62)
        i = int(T(b) * SR); t = tt(0.4)
        duck_src[i:i + len(t)] = np.maximum(duck_src[i:i + len(t)], np.exp(-t * 9))


def drums():
    for b in (3, 5, 9, 11, 13, 15, 17, 19):
        place(clap(), b, 0.36, 0.05, rev=0.25)
    # hats: 16ths with accents; open hat on the off-beat
    for sec in ((2, 6.5), (8, 13.5), (14, 19.5)):
        b = sec[0]
        while b < sec[1] - 1e-6:
            pos = round((b % 1) * 4)
            if pos == 2: place(hat(True), b, 0.1, 0.3, rev=0.1)
            else: place(hat(), b, 0.09 if pos == 0 else 0.05, -0.35 if pos % 2 else 0.25)
            b += 0.25
    # tension roll into the zoom-through
    b = 6.5
    while b < 8:
        step = 0.25 if b < 7.25 else 0.125
        place(clap(0.12), b, 0.1 + 0.3 * (b - 6.5) / 1.5, 0, rev=0.2)
        b += step
    # build roll into the blast
    b = 22.0
    while b < 23.75:
        step = 0.25 if b < 22.75 else 0.125 if b < 23.25 else 0.0625
        place(clap(0.1), b, 0.12 + 0.4 * (b - 22) / 1.75, 0, rev=0.25)
        place(hat(), b, 0.1, 0.4 if int(b * 16) % 2 else -0.4)
        b += step
    for b, g in ((2, .22), (8, .3), (14, .25), (20, .22), (24, .8)):
        place(crash(), b, g, 0, rev=0.3)
    place(reverse(crash(1.4)), 8 - 2.8, 0.45, 0, rev=0.2)      # reverse swell into the grid
    place(reverse(crash(1.0)), 24 - 2.0, 0.5, 0, rev=0.2)


def bass():
    for b0, b1 in ((2, 6.5), (8, 13.5), (14, 19.5)):
        b = b0
        while b < b1 - 1e-6:
            m = root_at(b)
            if round((b % 1) * 2) == 1:   # off-beat 8ths
                place(bass_note(m, 0.22), b, 0.3, bus='duck')
            b += 0.5
        # sustained sub under each section
        t = tt(T(b1 - b0))
        f = np.array([hz(root_at(b0 + x / SR * 2) - 12) for x in range(0, len(t), 480)])
        f = np.repeat(f, 480)[:len(t)]
        sub = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.minimum(1, np.minimum(t / 0.02, (t[-1] - t) / 0.05))
        place(sub, b0, 0.16, bus='duck')


def pads():
    place(pad(Fm, T(2.2), attack=0.8, bright=3), 0.4, 0.16, rev=0.5, bus='duck')
    for (s, ch), (e, _) in zip(PROG, PROG[1:] + [(24, None)]):
        bright = 7 if s < 20 else 4 + 5 * (s >= 22)
        place(pad(ch, T(e - s) + 0.2, attack=0.15, bright=bright), s, 0.12, rev=0.45, bus='duck')
    # final chord: wide and bright, rings out
    fin = pad(Fm9, 4.0, attack=0.02, bright=10)
    place(fin * np.exp(-tt(4.0) * 0.55), 24, 0.33, rev=0.6)
    place(pad([m + 12 for m in Fm9[1:]], 3.5, attack=0.6, bright=4) * 0.6, 24.5, 0.14, rev=0.8)


def melodic_fx():
    # dot pop at the start and the letters of MOVE
    place(blip(1760), 0.05, 0.35, rev=0.4)
    place(blip(2637, 0.08), 0.34, 0.2, rev=0.4)
    for i, m in enumerate([72, 75, 79, 84]):
        place(pluck(m, 0.25), 4 + i * 0.125, 0.22, -0.3 + i * 0.2, rev=0.3)
    # echo stack stutter
    for k in range(8):
        place(pluck(chord_at(5)[k % 4] + 12, 0.12), 5 + k * 0.125, 0.14, 0.5 if k % 2 else -0.5, rev=0.3)
    # grid ripples → blips from the centre
    for b in (8, 9, 10, 10.5, 11):
        place(blip(1568 if b != 10.5 else 2093), b, 0.22, 0, rev=0.5)
    # flow-field arpeggio
    b = 11.0
    k = 0
    while b < 13.4:
        ch = chord_at(b)
        m = ch[k % 4] + 12 + 12 * ((k // 4) % 2)
        place(pluck(m, 0.2, 0.7), b, 0.13, 0.6 if k % 2 else -0.6, rev=0.35, bus='duck')
        b += 0.25; k += 1
    # 3D: landings, jump, hops
    for i in range(5):
        place(thud(150 - i * 8), 14 + 0.25 + i * 0.4, 0.55, -0.5 + i * 0.25, rev=0.15)
    t = tt(0.5)
    up = np.sin(2 * np.pi * np.cumsum(220 + 500 * (t / 0.5) ** 1.5) / SR) * np.exp(-t * 3) * np.minimum(t / 0.03, 1)
    place(up, 17.0, 0.12, rev=0.4)
    for i in range(5):
        place(thud(110 - i * 5, 0.5), 18 + i * 0.035, 0.35, -0.5 + i * 0.25, rev=0.2)
    for i, m in enumerate([77, 80, 84, 87, 89]):
        place(pluck(m, 0.25), 18.45 + i * 0.13, 0.2, -0.5 + i * 0.25, rev=0.4)
    # particle build: rising arp
    b, k = 20.0, 0
    while b < 23.75:
        ch = chord_at(b)
        m = ch[k % 4] + 12 * (1 + (k // 4) % 3)
        br = 0.3 + 0.7 * (b - 20) / 3.75
        place(pluck(m, 0.18, br), b, 0.06 + 0.1 * (b - 20) / 3.75, 0.6 if k % 2 else -0.6, rev=0.4)
        b += 0.25 if b < 22 else 0.125; k += 1
    # end card: dot landings, typing, tags
    place(pop(), 25.42, 0.4, 0.2, rev=0.3)
    place(pop(0.08), 25.76, 0.18, 0.2, rev=0.3)
    role = 'Motion Designer'
    for i in range(len(role)):
        place(tick(3000 + (i % 3) * 400), 25.45 + (i / len(role)) * 0.8, 0.12, -0.2, rev=0.1)
    for i in range(5):
        place(tick(2200, 0.03), 26.0 + i * 0.1, 0.1, 0.3, rev=0.2)
    for i in range(4):
        place(tick(1800, 0.03), 26.35 + i * 0.12, 0.08, 0.5, rev=0.2)
    for b in (26, 27, 28, 29):
        place(kick(0.3, 0.5) * 0.5, b, 0.35)
    # slate typing
    for i in range(18):
        place(tick(3400), 1.16 + i * (0.46 / 18), 0.08, 0.1)


def transitions():
    place(sweep(T(0.45), 300, 6000, 0.5), 0.55, 0.25, rev=0.3)           # line stretch
    place(boom(1.2) * 0.6, 1.0, 0.35)
    place(sweep(T(0.35), 5000, 800, 0.6, 0.6), 3.0, 0.22, 0.2, rev=0.2)  # IT swing
    z = sweep(T(1.4), 200, 9000, 0.35, 2.2) * np.linspace(0.2, 1, int(T(1.4) * SR)) ** 2
    place(z, 6.6, 0.55, rev=0.3)                                           # zoom-through
    place(sweep(T(0.6), 400, 8000, 0.4, 2.0), 13.4, 0.35, rev=0.3)        # stripes fill
    place(sweep(T(0.5), 800, 7000, 0.5, 1.5) * np.linspace(0.3, 1, int(T(0.5) * SR)), 19.5, 0.5, 0.4, rev=0.2)  # whip
    place(sweep(T(0.8), 6000, 300, 0.5, 0.5), 20.0, 0.25, -0.3, rev=0.4)  # particles arrive
    # riser: pitch + noise, cuts dead a sixteenth before the blast
    d = T(1.75)
    t = tt(d)
    f = 180 * (2400 / 180) ** ((t / d) ** 1.8)
    r = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.3 + sweep(d, 400, 11000, 0.4, 1.8) * 0.8
    r *= (t / d) ** 2
    place(r, 22.0, 0.45, rev=0.3)
    # the blast
    place(boom(3.0), 24, 1.5)
    place(kick(0.5, 1.6), 24, 1.3)
    place(band(noise(1.2), 200, 12000) * np.exp(-tt(1.2) * 4), 24, 0.55, rev=0.6)
    place(sweep(T(1.2), 9000, 200, 0.6, 0.5), 24.02, 0.3, rev=0.5)


def reverb(x, secs=2.4, damp=0.5):
    n = int(secs * SR)
    t = np.arange(n) / SR
    out = np.zeros_like(x)
    for c in range(2):
        ir = rng.standard_normal(n) * np.exp(-t / damp)
        ir = band(ir, 150, 9000)
        ir[:int(0.012 * SR)] = 0
        ir /= np.sqrt(np.sum(ir ** 2))
        L = x.shape[1] + n
        nfft = 1 << (L - 1).bit_length()
        out[c] = np.fft.irfft(np.fft.rfft(x[c], nfft) * np.fft.rfft(ir, nfft), nfft)[:x.shape[1]]
    return out


kicks(); drums(); bass(); pads(); melodic_fx(); transitions()

duck = 1 - 0.65 * duck_src
mix = dry + duckable * duck + reverb(send) * 0.55
mix = mix[:, :N]
# master: gentle glue, soft clip, fade tail
mix = np.tanh(mix * 0.7) / np.tanh(0.7)
fade = np.ones(N); fade[-int(0.35 * SR):] = np.linspace(1, 0, int(0.35 * SR)) ** 2
mix *= fade
mix /= np.max(np.abs(mix)) / 0.89
pcm = (mix.T * 32767).astype(np.int16)
with wave.open(os.path.join(ROOT, 'soundtrack.wav'), 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print('wrote soundtrack.wav', pcm.shape)
