#!/usr/bin/env python3
"""Original score + sound design for the 12 s 娜乌斯嘉AI channel ident.

120 BPM, 4/4, A minor, Am–F–C–G.  48 kHz / 16-bit / stereo / exactly 12.0 s (6 bars).
Every sound effect is placed on the frame of its visual action in index.html
(the timing constants below mirror the ones in the page).

    python3 gen_audio.py            -> writes music.wav next to this script
    STEMS=dir python3 gen_audio.py  -> also writes music/sfx stems (for analysis)
"""
import os
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

SR = 48000
DUR = 12.0
N = int(SR * DUR)
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261027)

# ----------------------------------------------------------------- timeline (mirrors index.html)
T_STR = 1.4                      # road straightens, car speeds into the sun
T_BOOM = 2.0                     # sun explosion
SWEEP0, SWEEPD = 2.8, 0.3
T_FLY = 6.0                      # red pixel flies above 娜
LAND = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0]
PERIOD_T = 9.5                   # period lands + glitch + subtitle reveal (all together)
SUB_T0 = 9.5
SUB_LOCK0, SUB_LOCKD, SUB_N = 9.58, 0.045, 11
T_SUBFL = 10.2                   # subtitle locked: second sonic logo
T_GLINT, T_GLINTD = 10.1, 0.62   # glint across the title in the hold
T_CRT = 11.0
GLITCH1 = (PERIOD_T, PERIOD_T + 11 / 60)
GLITCH2 = (PERIOD_T + 13 / 60, PERIOD_T + 23 / 60)
SILENCES = [(1.75, 2.0), (10.75, 11.0)]


def S(t):
    return int(round(t * SR))


def tt(n):
    return np.arange(n) / SR


def mtof(m):
    return 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)


# ----------------------------------------------------------------- dsp helpers
def fade_in(x, ms=3.0):
    n = min(len(x) if x.ndim == 1 else x.shape[-1], max(2, int(SR * ms / 1000)))
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    x = x.copy()
    x[..., :n] *= w
    return x


def fade_out(x, ms=6.0):
    n = min(x.shape[-1], max(2, int(SR * ms / 1000)))
    w = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, n))
    x = x.copy()
    x[..., -n:] *= w
    return x


def env_ad(n, att, tau):
    t = tt(n)
    e = np.exp(-t / tau)
    a = max(1, int(att * SR))
    e[:a] *= 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, a))
    return e


def sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output='sos')


def filt(x, kind, f, order=2):
    return signal.sosfilt(sos(kind, f, order), x, axis=-1)


def _blep(p, dt):
    y = np.zeros_like(p)
    m = p < dt
    x = p[m] / dt[m]
    y[m] = x + x - x * x - 1
    m = p > 1 - dt
    x = (p[m] - 1) / dt[m]
    y[m] = x * x + x + x + 1
    return y


def _f(f, n):
    f = np.asarray(f, float)
    return np.full(n, float(f)) if f.ndim == 0 else f[:n]


def saw(f, n, ph0=0.0):
    f = _f(f, n)
    dt = f / SR
    p = (ph0 + np.cumsum(dt)) % 1.0
    return 2 * p - 1 - _blep(p, dt)


def square(f, n, duty=0.5, ph0=0.0):
    f = _f(f, n)
    dt = f / SR
    p = (ph0 + np.cumsum(dt)) % 1.0
    return np.where(p < duty, 1.0, -1.0) + _blep(p, dt) - _blep((p + 1 - duty) % 1.0, dt)


def sine(f, n, ph0=0.0):
    f = _f(f, n)
    return np.sin(2 * np.pi * (ph0 + np.cumsum(f) / SR))


def noise(n):
    return rng.standard_normal(n)


def svf_lp(x, fc):
    """time-varying 2-pole (TPT state-variable) low-pass; fc per sample"""
    y = np.zeros_like(x)
    g = np.tan(np.pi * np.clip(fc, 20, SR * 0.45) / SR)
    k = 1.4
    ic1 = ic2 = 0.0
    for i in range(len(x)):
        gi = g[i]
        a1 = 1 / (1 + gi * (gi + k))
        v3 = x[i] - ic2
        v1 = a1 * ic1 + gi * a1 * v3
        v2 = ic2 + gi * v1
        ic1 = 2 * v1 - ic1
        ic2 = 2 * v2 - ic2
        y[i] = v2
    return y


def svf_bp(x, fc, q=2.0):
    y = np.zeros_like(x)
    g = np.tan(np.pi * np.clip(fc, 20, SR * 0.45) / SR)
    k = 1 / q
    ic1 = ic2 = 0.0
    for i in range(len(x)):
        gi = g[i]
        a1 = 1 / (1 + gi * (gi + k))
        v3 = x[i] - ic2
        v1 = a1 * ic1 + gi * a1 * v3
        v2 = ic2 + gi * v1
        ic1 = 2 * v1 - ic1
        ic2 = 2 * v2 - ic2
        y[i] = v1
    return y


def st(x, pan=0.0, width=0.0):
    """mono -> stereo with constant-power pan"""
    if x.ndim == 2:
        return x
    a = (pan + 1) * np.pi / 4
    return np.stack([x * np.cos(a) * 1.4142, x * np.sin(a) * 1.4142])


def add(buf, x, t0, gain=1.0, pan=0.0):
    x = st(x, pan)
    i0 = S(t0)
    if i0 < 0:
        x = x[:, -i0:]
        i0 = 0
    n = min(x.shape[1], N - i0)
    if n > 0:
        buf[:, i0:i0 + n] += gain * x[:, :n]


def addm(buf, x, t0, gain=1.0, pan=0.0):
    """music add: anything that starts before a silent half-beat is cut (10 ms fade) at its start,
    so no tail resumes after the silence"""
    x = st(x, pan)
    for a0, _ in SILENCES:
        if t0 < a0:
            lim = S(a0) - S(t0)
            if lim < x.shape[1]:
                x = x[:, :max(lim, 2)].copy()
                r = min(S(0.01), x.shape[1])
                x[:, -r:] *= np.linspace(1, 0, r)
            break
    add(buf, x, t0, gain)


def norm(x):
    m = np.max(np.abs(x))
    return x / m if m > 0 else x


# ----------------------------------------------------------------- music
mus_pad = np.zeros((2, N))
mus_arp = np.zeros((2, N))
mus_bass = np.zeros((2, N))
mus_drm = np.zeros((2, N))

PROG = ['Am', 'F', 'C', 'G', 'Am', 'G']
PAD = {'Am': [57, 60, 64], 'F': [57, 60, 65], 'C': [55, 60, 64], 'G': [55, 59, 62]}
ROOT = {'Am': 33, 'F': 29, 'C': 36, 'G': 31}
ARP = {'Am': [69, 72, 76, 81], 'F': [69, 72, 77, 81], 'C': [67, 72, 76, 79], 'G': [67, 71, 74, 79]}
ARP_PAT = [0, 2, 3, 2, 1, 2, 3, 2, 0, 2, 3, 2, 1, 3, 2, 1]


def pad_chord(notes, dur, att=0.06, rel=0.35):
    n = S(dur + rel)
    L = np.zeros(n)
    R = np.zeros(n)
    for m in notes:
        f = mtof(m)
        for d, side in ((-0.11, 'L'), (0.0, 'B'), (0.12, 'R')):
            v = saw(f * 2 ** (d / 12), n, ph0=rng.random()) * 0.33
            if side in 'LB':
                L += v * (1.0 if side == 'L' else 0.7)
            if side in 'RB':
                R += v * (1.0 if side == 'R' else 0.7)
    sub = sine(mtof(notes[0] - 12), n) * 0.45
    L += sub
    R += sub
    e = np.ones(n)
    a = S(att)
    e[:a] = np.linspace(0, 1, a) ** 1.5
    r0 = S(dur)
    e[r0:] = np.exp(-tt(n - r0) / (rel / 3.5))
    return np.stack([L, R]) * e


for b, ch in enumerate(PROG):
    t0 = 2.0 * b
    if b == len(PROG) - 1:
        x = pad_chord(PAD['G'], 0.75, att=0.04, rel=0.06)
    else:
        x = pad_chord(PAD[ch], 2.0, att=0.08 if b else 0.012)
    addm(mus_pad, x, t0)
# final resolve on Am at the CRT switch-off
addm(mus_pad, fade_in(pad_chord(PAD['Am'], 0.05, att=0.004, rel=0.9), 2), T_CRT, 1.1)

pad_open = filt(mus_pad, 'low', 1500, 2)
pad_muff = filt(mus_pad, 'low', 520, 2)
tN = tt(N)
xf = np.clip((tN - 1.75) / 0.25, 0, 1)
mus_pad = pad_muff * (1 - xf) * 0.62 + pad_open * xf * 0.7
# synthwave pump from the kick
pump = np.ones(N)
for k in range(4, 22):
    b = k * 0.5
    i0 = S(b)
    seg = tN[i0:] - b
    pump[i0:] *= 1 - 0.38 * np.exp(-seg / 0.14) * (seg < 0.5)
mus_pad *= pump

# arpeggio (16ths) — the melodic hook
for b, ch in enumerate(PROG):
    for s in range(16):
        t0 = 2.0 * b + 0.125 * s
        if t0 >= 10.75 or 1.75 <= t0 < 2.0:
            continue
        m = ARP[ch][ARP_PAT[s]]
        n = S(0.16)
        f = mtof(m)
        v = 0.55 * square(f, n, 0.3) + 0.45 * saw(f * 1.003, n)
        acc = 1.0 if s % 4 == 0 else (0.8 if s % 2 == 0 else 0.68)
        e = env_ad(n, 0.003, 0.085)
        v = filt(v * e, 'low', 3200, 2)
        addm(mus_arp, v, t0, acc, pan=(-0.25 if s % 2 else 0.25))
        ve = filt(v, 'low', 2400, 1)
        for d, g, pn in ((0.1875, 0.32, -0.7), (0.375, 0.22, 0.7), (0.5625, 0.12, -0.7)):
            if t0 + d < 10.75 and not (1.75 <= t0 + d < 2.0):
                addm(mus_arp, ve, t0 + d, acc * g, pan=pn)
arp_muff = filt(mus_arp, 'low', 900, 2)
mus_arp = arp_muff * (1 - xf) * 0.75 + mus_arp * xf

# bass: synthwave 8ths with a short glide, 2.0 → 10.75, plus the final note
BASS_PAT = [0, 0, 12, 0, 0, 12, 0, 12]
events = []
for b in range(1, 6):
    ch = PROG[b]
    for e8 in range(8):
        t0 = 2.0 * b + 0.25 * e8
        if t0 >= 10.75:
            continue
        events.append((t0, ROOT[ch] + BASS_PAT[e8], 0.215))
events.append((T_CRT, 33, 0.75))
prev_f = mtof(events[0][1])
for (t0, m, d) in events:
    n = S(d + 0.03)
    f1 = mtof(m)
    glide = prev_f * (f1 / prev_f) ** np.clip(tt(n) / 0.028, 0, 1)
    prev_f = f1
    v = 0.7 * saw(glide, n) + 0.55 * sine(glide, n)
    e = np.ones(n)
    a = S(0.004)
    e[:a] = np.linspace(0, 1, a)
    r0 = S(d)
    e[r0:] = np.linspace(1, 0, n - r0)
    if d > 0.5:
        e *= np.exp(-tt(n) / 0.35)
    fc = 300 + 900 * np.exp(-tt(n) / 0.06)
    v = svf_lp(v * e, fc)
    addm(mus_bass, np.tanh(1.6 * v), t0)


# drums
def kick(heavy=1.0):
    n = S(0.5)
    t = tt(n)
    f = 46 + 125 * np.exp(-t / 0.032)
    body = sine(f, n) * np.exp(-t / (0.26 * heavy))
    click = filt(noise(n), 'high', 1800) * np.exp(-t / 0.004) * 0.35
    x = np.tanh(1.8 * (body + click))
    return fade_in(x, 2)


def snare():
    n = S(0.35)
    t = tt(n)
    nz = filt(noise(n), 'band', [1300, 7000]) * np.exp(-t / 0.11)
    tone = sine(190 - 25 * t, n) * np.exp(-t / 0.06) * 0.8
    return fade_in(np.tanh(1.3 * (nz * 0.9 + tone)), 2)


def hat(open_=False):
    n = S(0.25 if open_ else 0.08)
    t = tt(n)
    x = filt(noise(n), 'high', 7200, 4) * np.exp(-t / (0.07 if open_ else 0.02))
    return fade_in(x, 2.0)


for k in range(4, 22):                     # 2.0 … 10.5
    b = k * 0.5
    addm(mus_drm, kick(), b, 0.95)
for b in range(1, 6):
    for off in (0.5, 1.5):
        addm(mus_drm, snare(), 2.0 * b + off, 0.55, pan=0.05)
for k in range(0, 43):
    t0 = k * 0.25
    if 1.75 <= t0 < 2.0 or t0 >= 10.75:
        continue
    off = (k % 2 == 1)
    addm(mus_drm, hat(open_=(off and k % 8 == 7)), t0, 0.16 if off else 0.1, pan=0.3)
addm(mus_drm, kick(1.6), T_CRT, 1.0)

music = 0.32 * mus_pad + 0.30 * mus_arp + 0.42 * mus_bass + 0.62 * mus_drm

# ----------------------------------------------------------------- sound effects
sfx = np.zeros((2, N))


def whoosh(dur, f0, f1, q=1.6, peak=0.5, seed_pan=0.0):
    n = S(dur)
    t = tt(n)
    fc = f0 * (f1 / f0) ** (t / dur)
    x = svf_bp(noise(n), fc, q)
    u = t / dur
    e = np.where(u < peak, np.sin(np.pi / 2 * u / peak) ** 2, np.cos(np.pi / 2 * (u - peak) / (1 - peak)) ** 2)
    return fade_in(x * e, 3)


def boom(f0=60, f1=32, tau=0.35, dur=0.9):
    n = S(dur)
    t = tt(n)
    f = f1 + (f0 - f1) * np.exp(-t / 0.08)
    x = sine(f, n) * np.exp(-t / tau)
    x += 0.6 * filt(noise(n), 'low', 260) * np.exp(-t / 0.05)
    x = np.tanh(2.2 * x)              # harmonics so it survives a phone speaker
    return fade_in(x, 2.5)


def click(f=3200, tau=0.006, nz=0.5, dur=0.03):
    n = S(dur)
    t = tt(n)
    x = sine(f, n) * np.exp(-t / tau) + nz * filt(noise(n), 'band', [f * 0.6, min(f * 2, 20000)]) * np.exp(-t / (tau * 0.7))
    return fade_in(x, 2)


# 0.0 opening hook: fast rush + engine that follows the car's speed
add(sfx, whoosh(0.62, 260, 3200, q=1.2, peak=0.18), 0.0, 0.55)
add(sfx, boom(90, 40, 0.18, 0.5), 0.0, 0.35)


def speed(t):
    t = np.asarray(t)
    return np.where(t < 0.5, 950 + 1900 * (1 - t / 0.5) ** 2,
                    np.where(t < T_STR, 950 + 1650 * ((t - 0.5) / (T_STR - 0.5)) ** 2, 2600.0))


n = S(1.75)
t = tt(n)
sp = speed(t) / 2600
away = np.clip((t - T_STR) / 0.35, 0, 1)
f_eng = (46 + 40 * sp) * (1 + 0.5 * away) * (1 + 0.006 * np.sin(2 * np.pi * 7 * t))
eng = 0.6 * saw(f_eng, n) + 0.5 * square(f_eng * 0.5, n, 0.4)
eng = svf_lp(eng, 260 + 700 * sp)
e = np.clip(t / 0.05, 0, 1) * (0.55 + 0.45 * sp) * (1 - 0.75 * away)
e[-S(0.008):] *= np.linspace(1, 0, S(0.008))
amb = np.zeros((2, N))
add(amb, eng * e, 0.0, 0.2)
# sun inhales (1.45–1.75): rising swell, cut by the silence
n = S(0.3)
t = tt(n)
sw = svf_bp(noise(n), 400 * (8 ** (t / 0.3)), 1.4) * (t / 0.3) ** 2
add(sfx, fade_out(sw, 6), 1.45, 0.45)

# 2.0 the pixel sun bursts: low impact + glass + sand that sweeps down and stops in 0.8 s
add(sfx, boom(64, 30, 0.42, 1.1), T_BOOM, 1.0)
gl = np.zeros(S(0.6))
for i in range(60):
    f = rng.uniform(2300, 9000)
    o = int(rng.exponential(0.035) * SR)
    d = rng.uniform(0.015, 0.11)
    m = S(0.25)
    if o + m > len(gl):
        continue
    gl[o:o + m] += rng.uniform(0.3, 1.0) * sine(f, m, rng.random()) * np.exp(-tt(m) / d)
gl += 1.2 * filt(noise(len(gl)), 'band', [3000, 11000]) * np.exp(-tt(len(gl)) / 0.09)
add(sfx, fade_in(norm(gl), 2), T_BOOM, 0.62, pan=0.0)
n = S(0.8)
t = tt(n)
grains = noise(n) * (rng.random(n) < 0.18) * 3.0
for ch, pan in ((0, -0.5), (1, 0.5)):
    g = np.roll(grains, 37 * ch)
    y = svf_lp(g, 9500 * (0.075 ** (t / 0.8)))
    y *= (1 - t / 0.8) ** 1.6 * np.clip(t / 0.01, 0, 1)
    add(sfx, fade_in(y, 3), T_BOOM, 0.42, pan=pan)

# 2.4–4.4 suction: denser and denser "whoosh" grains + a big sweep per glyph + a soft tick per settled pixel column
tg = 2.38
gap = 0.13
while tg < 4.35:
    dur = 0.09
    add(sfx, whoosh(dur, 900, 3200, q=2.2, peak=0.7), tg, 0.13 + 0.1 * (tg - 2.4) / 2, pan=rng.uniform(-0.7, 0.7))
    gap = max(0.032, gap * 0.93)
    tg += gap
for k in range(5):
    s0 = SWEEP0 + SWEEPD * k
    add(sfx, whoosh(0.34, 700, 4200, q=1.3, peak=0.75), s0 - 0.03, 0.5, pan=-0.6 + 0.3 * k)
    for c in range(20):
        if c % 2 == 0:
            continue
        x = (c + 1) / 20
        tl = s0 + SWEEPD * np.arccos(1 - 2 * x) / np.pi
        add(sfx, click(rng.uniform(2600, 4200), 0.004, 0.4), tl, 0.16, pan=-0.6 + 0.3 * k + 0.012 * c)

# 4.5 / 4.58 red extrusion pops out, layer by layer ("咔")
for t0, g in ((4.5, 0.75), (4.58, 0.6)):
    n = S(0.12)
    t = tt(n)
    x = filt(noise(n), 'band', [1400, 5200]) * np.exp(-t / 0.012) + 0.9 * sine(250 - 900 * t, n) * np.exp(-t / 0.035)
    add(sfx, fade_in(np.tanh(1.5 * x), 2), t0, g)
# 5.0 "AI" turns warm yellow: a bright two-partial chime
n = S(0.5)
t = tt(n)
ch = sine(1318.5, n) * np.exp(-t / 0.16) + 0.5 * sine(1975.5, n) * np.exp(-t / 0.1) + 0.5 * square(659.25, n, 0.25) * np.exp(-t / 0.03)
add(sfx, fade_in(ch, 2), 5.0, 0.7)
# 5.5 glow powers on left → right: buzzing flicker that follows the visual flicker
n = S(0.5)
t = tt(n)
hum = saw(100, n) * 0.6 + saw(200.5, n) * 0.4
hum = filt(hum, 'band', [180, 2600])
gate = np.zeros(n)
FL = [1, 0, 0.7, 0, 0, 1, 0.4, 1, 1, 0.8]
for k in range(5):
    on = 0.065 * k
    idx = np.clip(((t - on) / 0.016).astype(int), 0, 9)
    g = np.where(t < on, 0.0, np.where(t - on < 0.16, np.array(FL)[idx], 1.0))
    gate = np.maximum(gate, g)
gate = uniform_filter1d(gate, S(0.002))
hum *= gate * np.exp(-np.clip(t - 0.3, 0, None) / 0.12)
add(sfx, fade_in(hum, 3), 5.5, 0.75)
add(sfx, click(1800, 0.008, 0.8, 0.05), 5.5, 0.8)
# 8.7–9.5 reverse-reverb "inhale" into the period landing / glitch hit
n = S(1.3)
t = tt(n)
ir = noise(n) * np.exp(-t / 0.32)
src = np.zeros(n)
for m in (57, 60, 64, 69):
    src[:S(0.2)] += sine(mtof(m), S(0.2)) * np.exp(-tt(S(0.2)) / 0.05)
src[:S(0.01)] += noise(S(0.01))
wet = signal.fftconvolve(src, ir)[:n]
wet = filt(wet, 'band', [300, 7000])
rev = norm(wet[::-1])[-S(0.8):]
rev *= np.linspace(0, 1, len(rev)) ** 1.2
rev = fade_out(rev, 14)
add(sfx, st(rev, 0.0), PERIOD_T - 0.028 - len(rev) / SR, 0.8)

# 9.5 glitch: hard hit + 0.4 s of digital "滋啦", segment by segment with the picture (30 Hz), 2nd burst weaker
# (this hit also carries the subtitle's strong beat, so the subtitle has no separate impact of its own)
add(sfx, boom(75, 34, 0.3, 0.8), PERIOD_T, 1.0)
add(sfx, kick(1.3), PERIOD_T, 0.8)
add(sfx, click(2800, 0.01, 1.2, 0.06), PERIOD_T, 0.7)
for (a, b), g in ((GLITCH1, 1.0), (GLITCH2, 0.55)):
    tseg = a
    while tseg < b - 1e-6:
        nb = (np.floor(tseg * 30 + 1e-6) + 1) / 30
        d = min(nb - tseg, b - tseg)
        n = S(d)
        kind = rng.integers(0, 3)
        if kind == 0:
            x = square(rng.uniform(180, 1400), n, rng.uniform(0.1, 0.5))
        elif kind == 1:
            hold = int(rng.integers(6, 40))
            x = np.repeat(noise(n // hold + 1), hold)[:n]
        else:
            x = np.round(noise(n) * 2) / 2 * square(rng.uniform(60, 300), n)
        x = filt(x, 'band', [250, 9000]) * rng.uniform(0.7, 1.0)
        x = fade_out(fade_in(x, 2.0), 2.0)
        add(sfx, x, tseg, 0.42 * g, pan=rng.uniform(-0.6, 0.6))
        tseg += d

# 6.0 the red pixel flies to 娜, 6.37 it drops
add(sfx, whoosh(0.28, 600, 3000, q=1.8, peak=0.6), T_FLY, 0.32, pan=-0.3)
n = S(0.13)
add(sfx, fade_in(sine(1400 * 0.45 ** (tt(n) / 0.13), n) * np.linspace(0.2, 1, n) ** 2, 3), T_FLY + 0.37, 0.12, pan=-0.5)

# 6.5–9.0 landings: 8-bit note climbing the scale + a soft weight thump
SCALE = [69, 71, 72, 74, 76, 77]
for j, t0 in enumerate(LAND):
    n = S(0.2)
    t = tt(n)
    f = mtof(SCALE[j]) * (1 + 1.0 * np.exp(-t / 0.008))
    x = square(f, n, 0.25) * np.exp(-t / 0.075)
    x = filt(x, 'low', 6000, 1)
    add(sfx, fade_in(x, 2), t0, 0.55, pan=-0.6 + 0.24 * j)
    n = S(0.12)
    t = tt(n)
    th = np.tanh(2 * sine(160 - 600 * t, n) * np.exp(-t / 0.035))
    add(sfx, fade_in(th, 2), t0, 0.6, pan=-0.6 + 0.24 * j)


# 9.5 the period lands: crisp "叮" + sonic logo (motif A4–E5–A5), repeated at 10.2 when the subtitle has locked
def bell(f, dur=0.6, tau=0.25, idx=2.2):
    n = S(dur)
    t = tt(n)
    mod = idx * np.exp(-t / 0.08) * sine(f * 3.5, n)
    x = np.sin(2 * np.pi * np.cumsum(np.full(n, f)) / SR + mod) * np.exp(-t / tau)
    return fade_in(x, 2)


def logo(t0, gain):
    for i, (m, tau) in enumerate(((69, 0.16), (76, 0.16), (81, 0.24))):
        n = S(0.7)
        t = tt(n)
        f = mtof(m)
        x = 0.7 * sine(f, n) + 0.3 * square(f, n, 0.5) * np.exp(-t / 0.03)
        x = filt(x * np.exp(-t / tau), 'low', 5000, 1) + 0.35 * sine(f * 2, n) * np.exp(-t / (tau * 0.6))
        add(sfx, fade_in(x, 2), t0 + 0.11 * i, gain, pan=-0.2 + 0.2 * i)


add(sfx, bell(1760, 0.5, 0.12, 2.0), PERIOD_T, 0.6, pan=0.4)
add(sfx, click(2600, 0.01, 0.3, 0.05), PERIOD_T, 0.3, pan=0.4)
logo(PERIOD_T, 0.5)
# 10.1–10.72 glint sweep over the title
GD = T_GLINTD
n = S(GD)
t = tt(n)
sh = np.zeros(n)
for fm in (2637.0, 3520.0, 4186.0):
    sh += sine(fm * (1 + 0.12 * t / GD), n, rng.random()) * (0.6 + 0.4 * np.sin(2 * np.pi * 14 * t + fm))
sh += 0.5 * filt(noise(n), 'band', [6000, 11000])
sh *= np.sin(np.pi * t / GD) ** 2
add(sfx, np.stack([sh * (1 - t / GD), sh * (t / GD)]), T_GLINT, 0.16)
# 9.22–9.47 short riser into the landing downbeat
n = S(0.28)
t = tt(n)
ri = sine(320 * (5 ** (t / 0.28)), n) * 0.5 + svf_bp(noise(n), 800 * (6 ** (t / 0.28)), 2.0)
ri *= (t / 0.28) ** 1.6
ri = fade_out(ri[:S(0.25)], 12)
add(sfx, ri, PERIOD_T - 0.28, 0.55)
# 9.5 the title lifts (the impact itself is the glitch hit above)
add(sfx, whoosh(0.35, 1800, 500, q=1.5, peak=0.2), SUB_T0, 0.2)
# 9.5–10.05 decoding chatter + a light "嗒" for every locked character
for k in range(int(0.55 / 0.05)):
    for i in range(3):
        n = S(0.012)
        add(sfx, fade_out(fade_in(square(rng.uniform(1400, 3200), n, 0.5), 2), 2), SUB_T0 + 0.05 * k + 0.016 * i, 0.028, pan=rng.uniform(-0.6, 0.6))
for i in range(SUB_N):
    add(sfx, click(2400, 0.006, 0.9, 0.04), SUB_LOCK0 + SUB_LOCKD * i, 0.5, pan=-0.6 + 1.2 * i / (SUB_N - 1))
logo(T_SUBFL, 0.5)

# 11.0 CRT switch-off "噗": heaviest hit — sub boom, puff, downward zap, static
add(sfx, boom(70, 28, 0.5, 1.2), T_CRT, 1.0)
n = S(0.4)
t = tt(n)
puff = filt(noise(n), 'low', 900) * np.exp(-t / 0.06)
zap = sine(2200 * (0.03 ** np.clip(t / 0.14, 0, 1)), n) * np.exp(-t / 0.1)
stat = filt(noise(n) * (rng.random(n) < 0.06), 'high', 2500) * np.exp(-t / 0.07)
add(sfx, fade_in(np.tanh(1.4 * (puff + 0.45 * zap + 1.2 * stat)), 2), T_CRT, 0.8)
# 11.5 dot flashes, 11.7 short "哔" as it dies
add(sfx, bell(2093, 0.3, 0.07, 1.0), T_CRT + 0.5, 0.16)
n = S(0.075)
bp = filt(square(1046.5, n, 0.5), 'low', 4500, 2)
add(sfx, fade_out(fade_in(bp, 3), 8), T_CRT + 0.7, 0.3)

# ----------------------------------------------------------------- mix
# sidechain: music ducks under every effect
env = np.max(np.abs(sfx), axis=0)
env = maximum_filter1d(env, S(0.01))
a = np.exp(-1 / (SR * 0.12))
env = signal.lfilter([1 - a], [1, -a], env)
duck_db = -10.0 * np.clip(env / 0.25, 0, 1)
duck = 10 ** (duck_db / 20)
music_d = music * duck

# the two silent half beats
gate = np.ones(N)
for a0, a1 in SILENCES:
    i0, i1 = S(a0), S(a1)
    r = S(0.01)
    gate[i0:i1] = 0
    gate[i0 - r:i0] *= np.linspace(1, 0, r)
sfx_g = sfx

mix = music_d + 1.15 * (sfx_g + amb)


def peaking(f0, gain_db, q):
    A = 10 ** (gain_db / 40)
    w = 2 * np.pi * f0 / SR
    al = np.sin(w) / (2 * q)
    b = [1 + al * A, -2 * np.cos(w), 1 - al * A]
    aa = [1 + al / A, -2 * np.cos(w), 1 - al / A]
    return np.array(b) / aa[0], np.array(aa) / aa[0]


mix = filt(mix, 'high', 40, 4)                       # < 40 Hz rolled off
b, a = peaking(5800, 2.5, 0.7)                       # a little 4–8 kHz air for phone speakers
mix = signal.lfilter(b, a, mix, axis=-1)
mix *= gate                                          # gate after the filters so nothing rings into the silences
music_d = music_d * gate
sfx_g = sfx_g * gate


# ----------------------------------------------------------------- loudness (ITU-R BS.1770-4) / true peak
def lufs(x):
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=-1), axis=-1)
    blk, hop = S(0.4), S(0.1)
    ms = []
    for i in range(0, x.shape[1] - blk + 1, hop):
        ms.append(np.sum(np.mean(y[:, i:i + blk] ** 2, axis=1)))
    ms = np.array(ms)
    lk = -0.691 + 10 * np.log10(ms + 1e-12)
    g = ms[lk > -70]
    rel = -0.691 + 10 * np.log10(np.mean(g)) - 10
    g2 = ms[(lk > -70) & (lk > rel)]
    return -0.691 + 10 * np.log10(np.mean(g2))


def true_peak_db(x):
    up = signal.resample_poly(x, 4, 1, axis=-1)
    return 20 * np.log10(np.max(np.abs(up)) + 1e-12)


def limiter(x, ceil_db=-1.6, look_ms=2.0, rel_ms=60.0):
    c = 10 ** (ceil_db / 20)
    up = np.abs(signal.resample_poly(x, 4, 1, axis=-1)).max(axis=0)
    pk = up.reshape(-1, 4).max(axis=1)[:x.shape[1]]
    g = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    L = max(1, int(SR * look_ms / 1000))
    g = minimum_filter1d(g, size=2 * L + 1, origin=0)
    # release: gain may only rise slowly
    a = np.exp(-1 / (SR * rel_ms / 1000))
    out = np.empty_like(g)
    cur = 1.0
    for i in range(len(g)):
        gi = g[i]
        cur = gi if gi < cur else gi + (cur - gi) * a
        out[i] = cur
    out = uniform_filter1d(out, size=L, origin=0)
    out = np.minimum(out, g)
    return x * out


TARGET = -14.0
gain = 1.0
for it in range(6):
    y = limiter(mix * gain)
    l = lufs(y)
    gain *= 10 ** ((TARGET - l) / 20)
y = limiter(mix * gain)

# smooth fade to silence at the very end
fe = np.ones(N)
i0, i1 = S(11.74), S(11.95)
fe[i0:i1] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, i1 - i0))
fe[i1:] = 0
y *= fe
y[:, :1] *= 0

L_final = lufs(y)
TP = true_peak_db(y)
pcm = np.clip(np.round(y.T * 32767), -32768, 32767).astype('<i2')
out = os.path.join(HERE, 'music.wav')
import wave
with wave.open(out, 'wb') as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(pcm.tobytes())
print('wrote %s  %.3f s  integrated %.2f LUFS  true peak %.2f dBTP' % (out, pcm.shape[0] / SR, L_final, TP))

stems = os.environ.get('STEMS')
if stems:
    os.makedirs(stems, exist_ok=True)
    np.save(os.path.join(stems, 'music.npy'), (music_d * gain).astype(np.float32))
    np.save(os.path.join(stems, 'sfx.npy'), (1.15 * sfx_g * gain).astype(np.float32))
