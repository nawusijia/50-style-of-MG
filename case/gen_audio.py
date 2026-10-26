#!/usr/bin/env python3
"""娜乌斯嘉AI · 16 s channel ident — original synthwave score + per-action SFX.

120 BPM, 4/4, A minor, Am–F–C–G (one chord per bar, 8 bars = 16.0 s).
Writes music.wav: 48 kHz, 16-bit, stereo, exactly 16.0 s, ~-14 LUFS integrated, true peak <= -1.2 dBTP.
Every event time below mirrors a visual beat in index.html.
"""
import numpy as np
from scipy import signal
from scipy.ndimage import minimum_filter1d, uniform_filter1d
import wave, os

SR = 48000
DUR = 16.0
N = int(SR * DUR)
rng = np.random.default_rng(20261026)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'music.wav')

# ---------------------------------------------------------------- helpers
def ta(d):
    return np.arange(int(round(d * SR))) / SR

def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12)

NOTE = {'C': 0, 'C#': 1, 'D': 2, 'D#': 3, 'E': 4, 'F': 5, 'F#': 6, 'G': 7, 'G#': 8, 'A': 9, 'A#': 10, 'B': 11}
def nm(s):  # 'A4' -> midi
    n, o = (s[:-1], int(s[-1]))
    return 12 * (o + 1) + NOTE[n]

def fade_in(x, ms=3.0):
    n = max(1, int(SR * ms / 1000))
    n = min(n, len(x))
    x = x.copy()
    r = np.linspace(0, 1, n) ** 1.5
    x[:n] *= r[:, None] if x.ndim == 2 else r
    return x

def fade_out(x, ms=5.0):
    n = max(1, int(SR * ms / 1000))
    n = min(n, len(x))
    x = x.copy()
    r = np.linspace(1, 0, n)
    x[-n:] *= r[:, None] if x.ndim == 2 else r
    return x

def place(bus, sig, t0, pan=0.0, gain=1.0):
    """add mono or stereo sig into stereo bus at time t0 (equal-power pan, -1..1)"""
    i0 = int(round(t0 * SR))
    if sig.ndim == 1:
        a = (pan + 1) * np.pi / 4
        sig = np.stack([sig * np.cos(a), sig * np.sin(a)], 1) * np.sqrt(2)
    if i0 < 0:
        sig = sig[-i0:]; i0 = 0
    n = min(len(sig), N - i0)
    if n > 0:
        bus[i0:i0 + n] += sig[:n] * gain

def place_pan_sweep(bus, sig, t0, p0, p1, gain=1.0):
    pans = np.linspace(p0, p1, len(sig))
    a = (pans + 1) * np.pi / 4
    st = np.stack([sig * np.cos(a), sig * np.sin(a)], 1) * np.sqrt(2)
    place(bus, st, t0, gain=gain)

def sos(kind, f, order=2):
    return signal.butter(order, f, btype=kind, fs=SR, output='sos')

def filt(x, kind, f, order=2):
    return signal.sosfilt(sos(kind, f, order), x, axis=0)

def tv_filter(x, fcs, kind='lowpass', order=2, block=128):
    """time-varying butterworth (block-wise coefficients, state carried)"""
    y = np.zeros_like(x)
    zi = None
    for b in range(0, len(x), block):
        fc = float(np.clip(fcs[min(b, len(fcs) - 1)], 30, SR / 2 * 0.95)) if np.ndim(fcs) else fcs
        if kind == 'bandpass':
            lo, hi = fc / 1.35, min(fc * 1.35, SR / 2 * 0.95)
            s = signal.butter(order, [lo, hi], btype='bandpass', fs=SR, output='sos')
        else:
            s = signal.butter(order, fc, btype=kind, fs=SR, output='sos')
        if zi is None:
            zi = np.zeros((s.shape[0], 2))
        y[b:b + block], zi = signal.sosfilt(s, x[b:b + block], zi=zi)
    return y

def noise(d):
    return rng.standard_normal(int(round(d * SR)))

def saw_blep(freq):
    """band-limited saw from per-sample frequency array (PolyBLEP)"""
    dt = freq / SR
    ph = np.cumsum(dt) % 1.0
    y = 2 * ph - 1
    m = ph < dt
    t = ph[m] / dt[m]
    y[m] -= t + t - t * t - 1
    m2 = ph > 1 - dt
    t2 = (ph[m2] - 1) / dt[m2]
    y[m2] -= t2 * t2 + t2 + t2 + 1
    return y

def square_blep(freq, duty=0.5):
    dt = freq / SR
    ph = np.cumsum(dt) % 1.0
    ph2 = (ph + (1 - duty)) % 1.0
    def s(p):
        y = 2 * p - 1
        m = p < dt; t = p[m] / dt[m]; y[m] -= t + t - t * t - 1
        m2 = p > 1 - dt; t2 = (p[m2] - 1) / dt[m2]; y[m2] -= t2 * t2 + t2 + t2 + 1
        return y
    return (s(ph) - s(ph2)) * 0.5

def env_adsr(n, a, d, s, r, gate):
    t = np.arange(n) / SR
    e = np.where(t < a, t / max(a, 1e-6), s + (1 - s) * np.exp(-(t - a) / max(d, 1e-6)))
    rel = t >= gate
    if rel.any():
        g0 = e[np.argmax(rel)] if rel.any() else s
        e[rel] = g0 * np.exp(-(t[rel] - gate) / max(r, 1e-6))
    return e

def reverb_ir(dur=1.8, decay=0.5, lp=6000, seed=3):
    r = np.random.default_rng(seed)
    t = ta(dur)
    irs = []
    for ch in range(2):
        x = r.standard_normal(len(t)) * np.exp(-t / decay * 3.0)
        x = filt(x, 'lowpass', lp)
        x[:int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))
        irs.append(x)
    ir = np.stack(irs, 1)
    return ir / np.sqrt((ir ** 2).sum(0)).max()

def convolve_st(x, ir):
    if x.ndim == 1:
        x = np.stack([x, x], 1)
    y = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)
    return y

# ---------------------------------------------------------------- timing
BEAT = 0.5
BAR = 2.0
CHORDS = ['Am', 'F', 'C', 'G', 'Am', 'F', 'C', 'G']
PAD_V = {'Am': ['A3', 'C4', 'E4'], 'F': ['F3', 'A3', 'C4'], 'C': ['G3', 'C4', 'E4'], 'G': ['G3', 'B3', 'D4']}
BASS_R = {'Am': 'A1', 'F': 'F1', 'C': 'C2', 'G': 'G1'}
ARP_T = {'Am': ['A4', 'C5', 'E5'], 'F': ['A4', 'C5', 'F5'], 'C': ['G4', 'C5', 'E5'], 'G': ['G4', 'B4', 'D5']}
SIL = [(3.75, 4.0), (14.75, 15.0)]   # the only two silences

def chord_at(t):
    return CHORDS[min(7, int(t // BAR))]

# ---------------------------------------------------------------- MUSIC BUS
music_open = np.zeros((N, 2))
music_muff = np.zeros((N, 2))     # 0-4 s layer (dark / filtered)
drums = np.zeros((N, 2))
verb_send = np.zeros((N, 2))

# --- pad: detuned saws, mid-low voicing
def pad_note(f, dur, rel=0.35):
    n = int((dur + rel) * SR)
    e = env_adsr(n, 0.12, 0.6, 0.75, rel / 2.5, dur)
    out = np.zeros((n, 2))
    for i, cents in enumerate([-7, 0, 6]):
        for c, sgn in enumerate([1, -1]):
            fr = f * 2 ** (sgn * cents / 1200) * np.ones(n)
            out[:, c] += saw_blep(fr) * (0.36 if cents == 0 else 0.3)
    return out * e[:, None]

for b, ch in enumerate(CHORDS):
    t0 = b * BAR
    dur = BAR
    if b == 7:
        dur = 1.0      # G until 15.0, then the final Am tail (below)
    for nn in PAD_V[ch]:
        x = pad_note(midi(nm(nn)), dur)
        tgt = music_muff if t0 < 4 else music_open
        place(tgt, x, t0, gain=0.11)
# final Am tail under the CRT hit (low, warm, decays to nothing)
for nn in ['A2', 'E3', 'A3', 'C4']:
    n = int(0.95 * SR); tt = np.arange(n) / SR
    e = np.minimum(1, tt / 0.004) * np.exp(-tt / 0.28)
    x = np.stack([saw_blep(midi(nm(nn)) * 2 ** (-5 / 1200) * np.ones(n)), saw_blep(midi(nm(nn)) * 2 ** (5 / 1200) * np.ones(n))], 1) * e[:, None]
    place(music_open, x, 15.0, gain=0.12)

# --- arpeggio melody: 16ths, 3-3-3-3-4 synthwave grouping with a moving top line
ARP_RHY = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1]
TOP_MOVE = {'Am': ['E5', 'E5', 'D5', 'E5'], 'F': ['F5', 'E5', 'C5', 'F5'], 'C': ['E5', 'G5', 'E5', 'D5'], 'G': ['D5', 'B4', 'D5', 'G5']}
def pluck(f, dur=0.16, bright=1.0):
    n = int(dur * SR); tt = np.arange(n) / SR
    fr = f * np.ones(n)
    x = 0.65 * saw_blep(fr) + 0.35 * square_blep(fr * 1.002, 0.3)
    fc = 600 + 3800 * bright * np.exp(-tt / 0.05)
    x = tv_filter(x, fc, 'lowpass', 2, 64)
    e = np.minimum(1, tt / 0.003) * np.exp(-tt / 0.085)
    return fade_out(x * e, 8)

for b, ch in enumerate(CHORDS):
    for i in range(16):
        t = b * BAR + i * 0.125
        if t >= 15.0:
            break
        tones = ARP_T[ch]
        idx = ARP_RHY[i]
        note = tones[idx]
        if idx == 2:
            note = TOP_MOVE[ch][i // 4]
        acc = 1.0 if i % 3 == 0 else 0.75
        x = pluck(midi(nm(note)), bright=0.7 + 0.3 * acc)
        pan = -0.35 if i % 2 == 0 else 0.35
        tgt = music_muff if t < 4 else music_open
        place(tgt, x, t, pan=pan, gain=0.085 * acc)

# --- muffle the 0-4 layer, then ping-pong delay on the arp-heavy open layer
music_muff = filt(music_muff, 'lowpass', 520, 2) * 1.25
music_open = filt(music_open, 'lowpass', 3200, 2)
dl = int(0.375 * SR)
delayed = np.zeros_like(music_open)
src = music_open.copy()
for k in range(1, 4):
    g = 0.28 ** k
    sh = dl * k
    if sh < N:
        s = src[:N - sh]
        if k % 2:
            s = s[:, ::-1]
        delayed[sh:] += s * g
music_open += filt(delayed, 'bandpass', [400, 3000], 2) * 0.8

# --- bass: synthwave octave 8ths with glide
bass = np.zeros(N)
bass_notes = []
for b, ch in enumerate(CHORDS):
    if b < 2:
        continue
    for i in range(8):
        t = b * BAR + i * 0.25
        if t >= 14.75:
            break
        m = nm(BASS_R[ch]) + (12 if i % 2 else 0)
        bass_notes.append((t, m))
prev_f = midi(bass_notes[0][1])
for t, m in bass_notes:
    f = midi(m)
    n = int(0.24 * SR); tt = np.arange(n) / SR
    glide = prev_f + (f - prev_f) * (1 - np.exp(-tt / 0.012))
    x = saw_blep(glide) * 0.7 + np.sin(2 * np.pi * np.cumsum(glide) / SR) * 0.6
    fc = 220 + 1400 * np.exp(-tt / 0.06)
    x = tv_filter(x, fc, 'lowpass', 2, 64)
    e = np.minimum(1, tt / 0.003) * np.where(tt < 0.2, 1.0, np.exp(-(tt - 0.2) / 0.012))
    i0 = int(round(t * SR))
    nn_ = min(n, N - i0)
    bass[i0:i0 + nn_] += (x * e)[:nn_]
    prev_f = f
# final low A on the CRT hit
n = int(0.8 * SR); tt = np.arange(n) / SR
x = saw_blep(midi(nm('A1')) * np.ones(n)) * 0.6 + np.sin(2 * np.pi * midi(nm('A1')) * tt) * 0.7
x = filt(x, 'lowpass', 500) * np.minimum(1, tt / 0.004) * np.exp(-tt / 0.22)
bass[int(15.0 * SR):int(15.0 * SR) + n] += x[:N - int(15.0 * SR)]
music_open += np.stack([bass, bass], 1) * 0.17

# --- drums
def kick(heavy=1.0):
    d = 0.5; tt = ta(d)
    f = 46 + (150 + 40 * heavy) * np.exp(-tt / 0.03)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / (0.16 + 0.1 * heavy))
    body = np.tanh(body * (1.6 + heavy))
    click = filt(rng.standard_normal(len(tt)), 'bandpass', [1500, 6000]) * np.exp(-tt / 0.004) * 0.35
    knock = np.sin(2 * np.pi * 180 * tt) * np.exp(-tt / 0.03) * 0.3   # audible on phones
    return fade_in(body + click + knock, 2)

def snare():
    d = 0.35; tt = ta(d)
    tone = (np.sin(2 * np.pi * 190 * tt) + 0.5 * np.sin(2 * np.pi * 330 * tt)) * np.exp(-tt / 0.05)
    nz = filt(rng.standard_normal(len(tt)), 'bandpass', [1400, 9000]) * np.exp(-tt / 0.11)
    return fade_in(0.5 * tone + 0.9 * nz, 2)

def hat(open_=False):
    d = 0.2 if open_ else 0.08; tt = ta(d)
    nz = filt(rng.standard_normal(len(tt)), 'highpass', 7500, 4)
    return fade_in(nz * np.exp(-tt / (0.06 if open_ else 0.018)), 2)

for i in range(32):          # kicks on every beat from 4.0
    t = i * BEAT
    if t < 4.0 or t >= 15.0:
        continue
    if 14.75 <= t < 15.0:
        continue
    place(drums, kick(1.0 if t in (4.0, 8.0, 13.0) else 0.4), t, gain=0.42)
for i in range(32):          # snares on 2 and 4
    t = i * BEAT
    if t < 4.0 or t >= 14.75 or i % 2 == 0:
        continue
    s = snare()
    place(drums, s, t, gain=0.24)
    place(verb_send, s, t, gain=0.16)
for i in range(64):          # 8th hats throughout (0-4: the only drum)
    t = i * 0.25
    if t >= 14.75:
        break
    off = i % 2 == 1
    place(drums, hat(open_=off and (i % 4 == 3)), t, pan=0.25 if off else -0.15, gain=0.12 if off else 0.075)

# ---------------------------------------------------------------- SFX BUS
sfx = np.zeros((N, 2))

def boom(heavy=1.0, d=1.1):
    tt = ta(d)
    f = 38 + 70 * np.exp(-tt / 0.09)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / (0.35 * heavy + 0.1))
    x = np.tanh(x * 2.2) * 0.9
    mid = filt(rng.standard_normal(len(tt)), 'bandpass', [120, 900]) * np.exp(-tt / 0.09) * 0.8
    crack = filt(rng.standard_normal(len(tt)), 'bandpass', [1200, 7000]) * np.exp(-tt / 0.02) * 0.7
    return fade_in(x + mid + crack, 3)

def glass(d=0.9, n=46, seed=5):
    r = np.random.default_rng(seed)
    out = np.zeros((int(d * SR), 2))
    for i in range(n):
        f = r.uniform(2300, 8200)
        t0 = r.uniform(0, 0.16) ** 1.3
        tau = r.uniform(0.04, 0.35)
        L = int(min(d - t0, tau * 6) * SR)
        if L < 64:
            continue
        tt = np.arange(L) / SR
        x = np.sin(2 * np.pi * f * tt + r.uniform(0, 6)) * np.exp(-tt / tau) * r.uniform(0.2, 1)
        x = fade_in(x, 2)
        pan = r.uniform(-0.8, 0.8)
        a = (pan + 1) * np.pi / 4
        i0 = int(t0 * SR)
        out[i0:i0 + L, 0] += x * np.cos(a)
        out[i0:i0 + L, 1] += x * np.sin(a)
    tt = ta(0.05)
    crack = filt(rng.standard_normal(len(tt)), 'highpass', 3000) * np.exp(-tt / 0.012)
    out[:len(tt)] += crack[:, None] * 0.9
    return out * 0.16

def rustle(d=0.8):
    """pixel shards scattering: sparse grains, low-pass swept high->low, gone within 0.8 s"""
    tt = ta(d)
    nz = rng.standard_normal(len(tt))
    grains = np.zeros(len(tt))
    r = np.random.default_rng(9)
    k = 0
    tcur = 0.0
    while tcur < d - 0.01:
        dens = 900 * np.exp(-tcur / 0.25) + 60
        tcur += r.exponential(1 / dens)
        i = int(tcur * SR)
        L = int(r.uniform(0.0015, 0.006) * SR)
        if i + L < len(tt):
            grains[i:i + L] += np.hanning(L) * r.uniform(0.3, 1.0)
        k += 1
    x = nz * grains
    fc = 13000 * np.exp(-tt / 0.22) + 700
    x = tv_filter(x, fc, 'lowpass', 2, 128)
    env = (1 - tt / d) ** 2
    return fade_in(fade_out(x * env, 20), 3)

def whoosh(d=0.35, f0=350, f1=2600, curve=2.0):
    tt = ta(d)
    u = tt / d
    fc = f0 * (f1 / f0) ** (u ** curve)
    x = tv_filter(rng.standard_normal(len(tt)), fc, 'bandpass', 2, 64)
    env = np.sin(np.pi * np.clip(u, 0, 1) ** 0.8) ** 1.5
    return fade_in(fade_out(x * env, 8), 3)

def click(f=3600, amp=1.0):
    tt = ta(0.03)
    x = np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.005) + filt(rng.standard_normal(len(tt)), 'highpass', 3000) * np.exp(-tt / 0.0015) * 0.6
    return fade_in(x * amp, 2)

def snap(f=420):
    tt = ta(0.12)
    x = filt(rng.standard_normal(len(tt)), 'bandpass', [1200, 6000]) * np.exp(-tt / 0.02) * 1.3
    x += np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.02) * 0.8
    x += np.sin(2 * np.pi * 110 * tt) * np.exp(-tt / 0.03) * 0.6
    return fade_in(x, 2)

def bell(f, d=0.7, tau=0.22, bright=0.5):
    tt = ta(d)
    x = np.sin(2 * np.pi * f * tt) + bright * np.sin(2 * np.pi * 2.0 * f * tt) * np.exp(-tt / 0.07) \
        + 0.25 * bright * np.sin(2 * np.pi * 3.01 * f * tt) * np.exp(-tt / 0.04)
    e = np.exp(-tt / tau)
    return fade_out(fade_in(x * e, 3), 30)

def blip8(f, d=0.12):
    tt = ta(d)
    fr = np.where(tt < 0.03, f, f * 1.5)          # 8-bit two-step "coin" blip
    x = square_blep(fr, 0.25)
    x = filt(x, 'lowpass', 9000)
    e = np.where(tt < d - 0.02, 1.0, np.linspace(1, 0, len(tt))) * np.exp(-tt / 0.09)
    return fade_in(x * e, 3)

def thud(f=150):
    tt = ta(0.12)
    ff = f * 0.6 + f * 0.4 * np.exp(-tt / 0.02)
    x = np.sin(2 * np.pi * np.cumsum(ff) / SR) * np.exp(-tt / 0.04)
    x += filt(rng.standard_normal(len(tt)), 'bandpass', [800, 3000]) * np.exp(-tt / 0.006) * 0.4
    return fade_in(x, 2)

def digital_noise(d, seed):
    r = np.random.default_rng(seed)
    n = int(d * SR)
    out = np.zeros(n)
    i = 0
    while i < n:
        L = int(r.uniform(0.008, 0.03) * SR)
        seg = np.zeros(L)
        mode = r.integers(0, 3)
        tt = np.arange(L) / SR
        if mode == 0:     # sample & hold noise, bit-crushed
            hold = int(r.integers(2, 24))
            v = r.uniform(-1, 1, L // hold + 1).repeat(hold)[:L]
            seg = np.round(v * 6) / 6
        elif mode == 1:   # buzzy square
            f = r.uniform(120, 1800)
            seg = np.sign(np.sin(2 * np.pi * f * tt)) * 0.8
        else:             # crushed noise burst
            seg = np.round(r.uniform(-1, 1, L) * 3) / 3
        if r.random() < 0.18:
            seg *= 0.1    # stutter gaps
        seg *= np.minimum(1, np.arange(L) / (0.002 * SR))
        out[i:i + L] = seg[:n - i]
        i += L
    out = filt(out, 'lowpass', 11000)
    return out

def riser(d, f0, f1):
    tt = ta(d)
    u = tt / d
    f = f0 * (f1 / f0) ** (u ** 1.5)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR) * 0.6 + tv_filter(rng.standard_normal(len(tt)), f * 1.5, 'bandpass', 2, 64) * 0.7
    return fade_out(fade_in(x * u ** 1.6, 3), 4)

# 0.0 — opening hook: the camera surges forward
w = whoosh(0.55, 2600, 300, 0.6)
place_pan_sweep(sfx, w, 0.0, 0.0, 0.0, gain=0.55)
place(sfx, thud(110), 0.0, gain=0.35)
# 3.5–3.75 — sun inhales (rising, cut by the silence)
tt = ta(0.25)
inh = tv_filter(rng.standard_normal(len(tt)), 400 * (8 ** (tt / 0.25)), 'bandpass', 2, 64) * (tt / 0.25) ** 1.5
inh += np.sin(2 * np.pi * np.cumsum(220 * (4 ** (tt / 0.25))) / SR) * 0.35 * (tt / 0.25) ** 2
place(sfx, fade_out(fade_in(inh, 3), 3), 3.5, gain=0.5)
# 4.0 — sun explodes: low impact + glass + shard rustle (rustle done by 4.8)
place(sfx, boom(1.0), 4.0, gain=0.95)
place(sfx, glass(), 4.0, gain=1.0)
place_pan_sweep(sfx, rustle(0.8), 4.0, -0.2, 0.2, gain=0.5)
# 4.4–6.6 — vortex suction: ever denser whooshes + soft clicks as cells land
wt = []
t = 4.42
gap = 0.36
while t < 6.45:
    wt.append(t)
    t += gap
    gap = max(0.075, gap * 0.8)
for j, t in enumerate(wt):
    d = 0.34 if j < 4 else 0.24
    place_pan_sweep(sfx, whoosh(d, 300 + 40 * j, 2200 + 120 * j, 2.2), t, -0.7, 0.7, gain=0.42 + 0.02 * min(j, 10))
def invE(x):
    return np.arccos(1 - 2 * np.clip(x, 0, 1)) / np.pi
for j in range(44):
    xn = (j + 0.5) / 44
    tl = 4.95 + 1.55 * invE(xn) * 0.965 + 0.025
    place(sfx, click(3000 + 900 * ((j * 7) % 5) / 4, 1.0), tl, pan=-0.7 + 1.4 * xn, gain=0.13)
# 6.5 / 6.583 — extrusion layers pop out ("咔")
place(sfx, snap(420), 6.5, gain=1.0)
place(sfx, snap(330), 6.5 + 5 / 60, gain=0.95)
# 7.0 — "AI" turns warm yellow: bright chime
ch = bell(midi(nm('E6')), 0.5, 0.12, 0.6) + 0.7 * bell(midi(nm('A6')), 0.5, 0.09, 0.3)
place(sfx, ch, 7.0, pan=0.4, gain=0.22)
place(sfx, snap(600) * 0.5, 7.0, pan=0.4, gain=0.35)
# 7.5–8.0 — glow powers on left->right: mains buzz + per-letter electric ticks
tt = ta(0.5)
hum = saw_blep(np.full(len(tt), 100.0)) * 0.5 + saw_blep(np.full(len(tt), 150.0)) * 0.3
hum = filt(hum, 'bandpass', [180, 2400]) * (0.3 + 0.7 * (tt / 0.5) ** 1.2)
hum *= 1 + 0.4 * np.sin(2 * np.pi * 31 * tt)
place_pan_sweep(sfx, fade_out(fade_in(hum, 3), 6), 7.5, -0.6, 0.6, gain=0.3)
for k in range(6):
    place(sfx, click(2400 + 300 * k, 1.0), 7.5 + 0.075 * k, pan=-0.6 + 0.24 * k, gain=0.55)
    place(sfx, click(2600 + 300 * k, 0.6), 7.5 + 0.075 * k + 0.07, pan=-0.6 + 0.24 * k, gain=0.22)
# 7.2–8.0 — reverse-reverb inhale into the glitch
stab = np.zeros(int(0.3 * SR))
for nn in ['A3', 'C4', 'E4', 'A4']:
    tt = ta(0.3)
    stab += saw_blep(np.full(len(tt), midi(nm(nn)))) * np.exp(-tt / 0.08)
stab += filt(rng.standard_normal(len(stab)), 'bandpass', [800, 6000]) * np.exp(-ta(0.3) / 0.05) * 1.2
wet = convolve_st(np.concatenate([stab, np.zeros(int(2.6 * SR))]), reverb_ir(2.8, 1.1, 5000, 11))
rev = wet[::-1]
L = int(0.8 * SR)
rev = rev[len(rev) - L - 1:len(rev) - 1]
rev = rev / np.abs(rev).max()
rev *= (np.linspace(0, 1, L) ** 2.2)[:, None]
place(sfx, fade_out(rev, 4), 8.0 - 0.8, gain=0.45)
# 8.0 — glitch: full-weight hit + 0.4 s digital noise in two bursts
place(sfx, boom(0.8, 0.8), 8.0, gain=0.9)
dn1 = digital_noise(0.2, 21) * np.linspace(1, 0.75, int(0.2 * SR))
dn2 = digital_noise(0.18, 22) * np.linspace(0.55, 0.4, int(0.18 * SR))
place(sfx, fade_out(fade_in(dn1, 2), 4), 8.0, pan=-0.15, gain=0.38)
place(sfx, fade_out(fade_in(dn2, 2), 6), 8.22, pan=0.15, gain=0.38)
# 8.5 — the red pixel flies over to 娜
place_pan_sweep(sfx, whoosh(0.36, 500, 2800, 1.4), 8.5, 0.6, -0.6, gain=0.3)
place(sfx, blip8(midi(nm('E5')), 0.08), 8.5, pan=0.6, gain=0.28)
# 9.0–11.5 — bounces: 8-bit notes climbing the A-minor pentatonic + a soft thud
SCALE = ['A5', 'C6', 'D6', 'E6', 'G6', 'A6']
for k, t in enumerate([9.0, 9.5, 10.0, 10.5, 11.0, 11.5]):
    pan = -0.55 + 0.22 * k
    place(sfx, blip8(midi(nm(SCALE[k]))), t, pan=pan, gain=0.3)
    place(sfx, thud(160), t, pan=pan, gain=0.5)
# 12.0 — becomes the full stop: crisp "ding" + sonic logo (C5-E5-A5)
LOGO = ['C5', 'E5', 'A5']
place(sfx, bell(1760, 0.6, 0.16, 0.25), 12.0, pan=0.45, gain=0.24)
place(sfx, thud(200), 12.0, pan=0.45, gain=0.3)
def logo(t0, gain):
    for i, nn in enumerate(LOGO):
        b_ = bell(midi(nm(nn)), 0.6 if i < 2 else 0.75, 0.16 if i < 2 else 0.22, 0.55)
        place(sfx, b_, t0 + i * 0.125, pan=-0.2 + 0.2 * i, gain=gain)
logo(12.0, 0.3)
# 12.0–13.0 — light sweep across the title
sw = whoosh(0.95, 1800, 6500, 1.0) * 0.6
place_pan_sweep(sfx, sw, 12.05, -0.7, 0.7, gain=0.22)
# 12.75–13.0 — short upward riser into the strong beat
place(sfx, riser(0.25, 300, 1600), 12.75, gain=0.35)
# 13.0 — strong beat: title lifts, subtitle appears
place(sfx, boom(0.6, 0.6), 13.0, gain=0.75)
place(sfx, snap(250), 13.0, gain=0.7)
# 13.0–13.55 — scramble chatter + a light "嗒" as each character locks
for j in range(11):
    t = 13.0 + j * 0.05
    place(sfx, blip8(900 + 700 * ((j * 5) % 7) / 6, 0.025) * 0.5, t, pan=-0.5 + 0.1 * j, gain=0.06)
for k in range(11):
    place(sfx, click(2200 + 120 * k, 1.0), 13.1 + 0.044 * k, pan=-0.6 + 0.12 * k, gain=0.5)
    place(sfx, thud(260) * 0.5, 13.1 + 0.044 * k, pan=-0.6 + 0.12 * k, gain=0.2)
# 13.7 — subtitle settles: sonic logo again
logo(13.7, 0.3)
# 14.0–14.75 — full stop breathes (soft low swell, no high tones)
tt = ta(0.75)
br = filt(rng.standard_normal(len(tt)), 'bandpass', [300, 1200]) * np.sin(np.pi * tt / 0.75) ** 2
place(sfx, fade_out(fade_in(br, 3), 10), 14.0, gain=0.08)
# 15.0 — CRT power-off: the heaviest hit + electric "噗"
place(sfx, boom(1.2, 1.0), 15.0, gain=1.0)
tt = ta(0.4)
pu = tv_filter(rng.standard_normal(len(tt)), 5000 * np.exp(-tt / 0.08) + 150, 'lowpass', 2, 64) * np.exp(-tt / 0.12)
whine = np.sin(2 * np.pi * np.cumsum(3200 * np.exp(-tt / 0.07) + 120) / SR) * np.exp(-tt / 0.09) * 0.35
place(sfx, fade_in(pu * 1.3 + whine, 3), 15.0, gain=0.55)
# 15.5 — pixel flashes at centre
place(sfx, bell(midi(nm('E5')), 0.25, 0.06, 0.2), 15.5, gain=0.12)
# 15.7 — goes out with a short "哔"
tt = ta(0.075)
beep = square_blep(np.full(len(tt), 1000.0), 0.5)
beep = filt(beep, 'lowpass', 5000)
place(sfx, fade_out(fade_in(beep, 3), 5), 15.7, gain=0.2)

# ---------------------------------------------------------------- MIX
ir = reverb_ir(1.6, 0.55, 6000, 4)
music = music_open + music_muff + drums + convolve_st(verb_send, ir) * 0.5
sfx_wet = convolve_st(sfx, ir) * 0.12
sfx_all = sfx + sfx_wet

# ducking: music makes room whenever an effect sounds
env = np.abs(sfx).max(1)
env = uniform_filter1d(env, int(0.004 * SR))
att, rel = np.exp(-1 / (0.003 * SR)), np.exp(-1 / (0.14 * SR))
g = np.zeros(N)
lvl = 0.0
for i in range(0, N, 32):            # block-wise envelope follower
    v = env[i:i + 32].max()
    lvl = v if v > lvl else lvl * rel ** 32
    g[i:i + 32] = lvl
duck = 1 - 0.65 * np.clip(g / 0.15, 0, 1)
mix = music * duck[:, None] + sfx_all
if os.environ.get('DIAG'):
    bp = signal.butter(4, [1000, 6000], 'bandpass', fs=SR, output='sos')
    mb = signal.sosfilt(bp, (music * duck[:, None]).mean(1)); sb = signal.sosfilt(bp, sfx_all.mean(1))
    mf = (music * duck[:, None]).mean(1); sf = sfx_all.mean(1)
    def r(v): return 10 * np.log10(np.mean(v ** 2) + 1e-14)
    for t in [0.0, 3.55, 4.0, 4.42, 4.95, 5.6, 6.5, 6.583, 7.0, 7.5, 7.6, 8.0, 8.22, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5, 12.0, 12.125, 12.25, 12.4, 12.85, 13.0, 13.1, 13.276, 13.54, 13.7, 14.3, 15.0, 15.5, 15.7]:
        i = int(t * SR); j = i + int(0.06 * SR)
        print('t=%6.3f  sfx-vs-music  band1-6k %+5.1f dB   full %+5.1f dB' % (t, r(sb[i:j]) - r(mb[i:j]), r(sf[i:j]) - r(mf[i:j])))

# master EQ: high-pass below 40 Hz, presence lift 4–8 kHz
mix = signal.sosfilt(signal.butter(2, 40, 'highpass', fs=SR, output='sos'), mix, axis=0)
def peaking(f0, gain_db, q):
    A = 10 ** (gain_db / 40); w0 = 2 * np.pi * f0 / SR; al = np.sin(w0) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w0), 1 - al * A]); a = np.array([1 + al / A, -2 * np.cos(w0), 1 - al / A])
    return b / a[0], a / a[0]
b_, a_ = peaking(6000, 2.5, 0.7)
mix = signal.lfilter(b_, a_, mix, axis=0)

# ---------------------------------------------------------------- loudness / true peak
def lufs(x):
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]; a1 = [1, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]; a2 = [1, -1.99004745483398, 0.99007225036621]
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=0), axis=0)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    ms = np.array([(y[i:i + blk] ** 2).mean(0).sum() for i in range(0, len(y) - blk + 1, hop)])
    l = -0.691 + 10 * np.log10(ms + 1e-20)
    ms = ms[l > -70]
    rel_ = -0.691 + 10 * np.log10(ms.mean()) - 10
    ms2 = ms[(-0.691 + 10 * np.log10(ms)) > rel_]
    return -0.691 + 10 * np.log10(ms2.mean())

def true_peak(x):
    up = signal.resample_poly(x, 4, 1, axis=0)
    return 20 * np.log10(np.abs(up).max() + 1e-12)

def limiter(x, ceil_db=-1.35):
    thr = 10 ** (ceil_db / 20)
    up = np.abs(signal.resample_poly(x, 4, 1, axis=0)).max(1)
    pk = up[:len(x) * 4].reshape(-1, 4).max(1)[:len(x)]
    gt = np.minimum(1.0, thr / (pk + 1e-12))
    la = int(0.0025 * SR)
    gm = minimum_filter1d(gt, 2 * la + 1)
    gs = uniform_filter1d(gm, la)
    gs = np.minimum(gs, gt)
    # release smoothing (one-pole on gain reduction, block-wise)
    out = gs.copy()
    rel_c = np.exp(-1 / (0.06 * SR))
    cur = 1.0
    for i in range(0, len(out), 16):
        v = out[i:i + 16].min()
        cur = v if v < cur else 1 - (1 - cur) * rel_c ** 16
        out[i:i + 16] = np.minimum(out[i:i + 16], cur)
    return x * out[:, None]

def gate_silences(x):
    g = np.ones(len(x))
    tt = np.arange(len(x)) / SR
    for a, b in SIL:
        r = 0.004
        m = (tt >= a) & (tt < b)
        g[m] = 0
        i_a = int(a * SR); n_r = int(r * SR)
        g[i_a - n_r:i_a] = np.minimum(g[i_a - n_r:i_a], np.linspace(1, 0, n_r))
        i_b = int(b * SR)
        g[i_b:i_b + n_r] = np.minimum(g[i_b:i_b + n_r], np.linspace(0, 1, n_r))
    # final smooth decay to silence
    i0, i1 = int(15.8 * SR), int(15.97 * SR)
    g[i0:i1] *= np.linspace(1, 0, i1 - i0) ** 2
    g[i1:] = 0
    return x * g[:, None]

TARGET = -14.0
x = gate_silences(mix)
for it in range(6):
    l = lufs(x)
    x = x * 10 ** ((TARGET - l) / 20)
    x = limiter(x, -1.4)
    x = gate_silences(x)
l = lufs(x)
x = x * 10 ** ((TARGET - l) / 20)
if true_peak(x) > -1.25:
    x = limiter(x, -1.4)
x = gate_silences(x)
print('integrated LUFS %.2f  true peak %.2f dBTP' % (lufs(x), true_peak(x)))

pcm = np.clip(np.round(x * 32767), -32768, 32767).astype('<i2')
assert len(pcm) == N
with wave.open(OUT, 'wb') as wf:
    wf.setnchannels(2); wf.setsampwidth(2); wf.setframerate(SR)
    wf.writeframes(pcm.tobytes())
print('wrote', OUT, len(pcm) / SR, 's')
