#!/usr/bin/env python3
"""Sound for the 9 s making-of (the pixel 克肆 builds the ident). 120 BPM, A minor.
Every effect sits on the frame of its action in index.html (constants mirrored below)."""
import os, wave
import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, minimum_filter1d, uniform_filter1d

SR = 48000; DUR = 9.0; N = int(SR * DUR)
HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261028)
DROPS = [0.55, 1.0, 1.45, 1.95, 2.45, 2.95, 3.45, 4.45, 4.88]   # clips thrown into the timeline
T_MUSIC = 5.12
T_VER0, T_VERD, NVER = 5.3, 0.09, 10
T_CHEER, T_KICK, T_DASH0, T_PUSH0, T_PUSH1 = 5.96, 6.2, 6.28, 6.62, 7.62
T_REND0, T_REND1, T_ZOOM0 = 7.62, 8.02, 8.1

S = lambda t: int(round(t * SR))
tt = lambda n: np.arange(n) / SR
mtof = lambda m: 440.0 * 2 ** ((np.asarray(m, float) - 69) / 12)
def sine(f, n, ph=0.0):
    f = np.broadcast_to(np.asarray(f, float), (n,)); return np.sin(2 * np.pi * np.cumsum(f) / SR + ph)
def saw(f, n, ph=0.0):
    f = np.broadcast_to(np.asarray(f, float), (n,)); p = (np.cumsum(f) / SR + ph) % 1.0; return 2 * p - 1
def square(f, n, duty=0.5):
    f = np.broadcast_to(np.asarray(f, float), (n,)); p = (np.cumsum(f) / SR) % 1.0; return np.where(p < duty, 1.0, -1.0)
def noise(n): return rng.standard_normal(n)
def filt(x, kind, fc, order=2):
    b, a = signal.butter(order, np.asarray(fc) / (SR / 2), btype={'low': 'low', 'high': 'high', 'band': 'band'}[kind]); return signal.lfilter(b, a, x, axis=-1)
def svf_bp(x, fc, q=1.5):
    y = np.zeros_like(x); lp = bp = 0.0; fc = np.broadcast_to(np.asarray(fc, float), x.shape)
    for i in range(len(x)):
        f = 2 * np.sin(np.pi * min(fc[i], SR / 6) / SR); hp = x[i] - lp - bp / q; bp += f * hp; lp += f * bp; y[i] = bp
    return y
def fade_in(x, ms=3.0):
    x = np.array(x, float); r = min(S(ms / 1000), x.shape[-1]); x[..., :r] *= np.linspace(0, 1, r); return x
def fade_out(x, ms=6.0):
    x = np.array(x, float); r = min(S(ms / 1000), x.shape[-1]); x[..., -r:] *= np.linspace(1, 0, r); return x
def st(x, pan=0.0):
    if x.ndim == 2: return x
    l = np.cos((pan + 1) * np.pi / 4); r = np.sin((pan + 1) * np.pi / 4); return np.stack([x * l * 1.414, x * r * 1.414])
def add(buf, x, t0, g=1.0, pan=0.0):
    x = st(np.asarray(x, float), pan); i0 = S(t0)
    if i0 < 0: x = x[:, -i0:]; i0 = 0
    n = min(x.shape[1], N - i0)
    if n > 0: buf[:, i0:i0 + n] += g * x[:, :n]
def env_ad(n, a, d):
    t = tt(n); e = np.exp(-t / d); r = S(a); e[:r] *= np.linspace(0, 1, r); return e
def click(f=3200, tau=0.006, nz=0.5, dur=0.03):
    n = S(dur); t = tt(n); return fade_in((sine(f, n) * (1 - nz) + filt(noise(n), 'high', 2000) * nz) * np.exp(-t / tau), 1)
def whoosh(dur, f0, f1, q=1.6, peak=0.5):
    n = S(dur); t = tt(n); fc = f0 * (f1 / f0) ** (t / dur); x = svf_bp(noise(n), fc, q); u = t / dur
    e = np.where(u < peak, np.sin(np.pi / 2 * u / peak) ** 2, np.cos(np.pi / 2 * (u - peak) / (1 - peak)) ** 2); return x * e
def blip(f0, f1, dur=0.07, duty=0.25):
    n = S(dur); t = tt(n); f = f0 * (f1 / f0) ** (t / dur); return fade_out(fade_in(square(f, n, duty) * np.exp(-t / (dur * 0.6)), 2), 4)

# ------------------------------------------------------------------ music bed (Am F C G, 0.5 → 8.0, then drops out for the riser)
mus = np.zeros((2, N))
PROG = ['Am', 'F', 'C', 'G']
ROOT = {'Am': 33, 'F': 29, 'C': 36, 'G': 31}
PAD = {'Am': [57, 60, 64], 'F': [57, 60, 65], 'C': [55, 60, 64], 'G': [55, 59, 62]}
ARP = {'Am': [69, 72, 76, 81], 'F': [69, 72, 77, 81], 'C': [67, 72, 76, 79], 'G': [67, 71, 74, 79]}
END = 8.0
def kick():
    n = S(0.4); t = tt(n); f = 46 + 120 * np.exp(-t / 0.03)
    return fade_in(np.tanh(1.8 * (sine(f, n) * np.exp(-t / 0.2) + filt(noise(n), 'high', 1800) * np.exp(-t / 0.004) * 0.3)), 2)
def hat(op=False):
    n = S(0.2 if op else 0.06); t = tt(n); return fade_in(filt(noise(n), 'high', 7500, 4) * np.exp(-t / (0.06 if op else 0.016)), 2)
def snare():
    n = S(0.3); t = tt(n); return fade_in(np.tanh(1.3 * (filt(noise(n), 'band', [1300, 7000]) * np.exp(-t / 0.1) * 0.9 + sine(190 - 25 * t, n) * np.exp(-t / 0.05) * 0.8)), 2)
for k in range(1, 16):            # kicks 0.5 … 7.5
    add(mus, kick(), k * 0.5, 0.9)
for b in range(1, 4):
    for off in (0.5, 1.5): add(mus, snare(), 2.0 * b + off, 0.5, 0.05)
for k in range(2, 32):            # 8th hats
    add(mus, hat(k % 4 == 3), k * 0.25, 0.12 if k % 2 else 0.08, 0.3)
for b in range(4):
    ch = PROG[b]
    for e8 in range(8):
        t0 = 2.0 * b + 0.25 * e8
        if t0 < 0.5 or t0 >= END: continue
        n = S(0.22); f = mtof(ROOT[ch] + (12 if e8 % 3 == 2 else 0))
        v = (0.7 * saw(f, n) + 0.5 * sine(f, n)) * env_ad(n, 0.004, 0.12)
        add(mus, np.tanh(1.5 * filt(v, 'low', 700)), t0, 0.55)
    n = S(2.0); x = np.zeros(n)
    for m in PAD[ch]: x += saw(mtof(m) * 1.002, n, rng.random()) + saw(mtof(m) * 0.998, n, rng.random())
    e = np.ones(n); e[:S(0.05)] = np.linspace(0, 1, S(0.05)); e[-S(0.2):] = np.linspace(1, 0, S(0.2))
    add(mus, filt(x * e, 'low', 1200) * 0.09, 2.0 * b if b else 0.5)
    for s in range(16):           # arp from 2 s on
        t0 = 2.0 * b + 0.125 * s
        if t0 < 2.0 or t0 >= END: continue
        n = S(0.12); m = ARP[ch][[0, 2, 3, 2, 1, 2, 3, 2][s % 8]]
        add(mus, filt(square(mtof(m), n, 0.3) * env_ad(n, 0.003, 0.05), 'low', 3500), t0, 0.12 if s % 2 else 0.17, -0.3 if s % 2 else 0.3)

# ------------------------------------------------------------------ effects
sfx = np.zeros((2, N))
# power-on + panels snapping in, with a soft room tone so the first beats are never empty
n = S(0.8); add(sfx, (0.5 * sine(55, n) + 0.4 * filt(noise(n), 'low', 400)) * np.minimum(1, tt(n) / 0.02) * np.exp(-tt(n) / 0.5), 0.0, 0.12)
add(sfx, blip(220, 880, 0.12, 0.5), 0.0, 0.35)
for i in range(5): add(sfx, click(1800 + 400 * i, 0.01, 0.6, 0.05), 0.06 + 0.04 * i, 0.25, -0.6 + 0.3 * i)
# keyboard: typing ticks all the way (faster while the whole file flies past)
t = 0.18
while t < T_PUSH1:
    rate = 38 if T_PUSH0 <= t < T_PUSH1 else 20
    if not (T_VER0 <= t < T_VER0 + NVER * T_VERD):
        add(sfx, click(rng.uniform(2400, 4200), 0.004, 0.75, 0.02), t, rng.uniform(0.05, 0.1), rng.uniform(-0.5, -0.1))
    t += rng.uniform(0.6, 1.4) / rate
# every clip: pick-up pop, throw whoosh, landing thunk + a chime climbing the scale
CH = [69, 72, 74, 76, 79, 81, 84, 86, 88]
for j, D in enumerate(DROPS):
    add(sfx, blip(500, 1400, 0.06, 0.3), D - 0.36, 0.22, 0.2)
    n = S(0.2); add(sfx, filt(noise(n), 'band', [500, 1800]) * np.sin(np.pi * tt(n) / 0.2) ** 2, D - 0.34, 0.07, 0.1)   # drag scrape
    add(sfx, whoosh(0.15, 3200, 700, 1.4, 0.4), D - 0.15, 0.4, -0.2)
    n = S(0.12); th = np.tanh(2.2 * sine(150 - 500 * tt(n), n) * np.exp(-tt(n) / 0.04))
    add(sfx, fade_in(th, 2), D, 0.75)
    n = S(0.35); f = mtof(CH[j]); x = (0.6 * sine(f, n) + 0.3 * square(f, n, 0.25) * np.exp(-tt(n) / 0.03) + 0.25 * sine(2 * f, n)) * np.exp(-tt(n) / 0.11)
    add(sfx, fade_in(filt(x, 'low', 6000), 2), D + 0.01, 0.4, -0.4 + 0.1 * j)
# zoom slider drag (rising zip) + the timeline stretching
n = S(0.5); add(sfx, svf_bp(noise(n), 600 * (5 ** (tt(n) / 0.5)), 3.0) * np.sin(np.pi * tt(n) / 0.5) ** 2, 3.72, 0.35, 0.5)
add(sfx, whoosh(0.6, 300, 1500, 1.2, 0.5), 3.62, 0.25)
# music clip: glasses glint "ting" + the waveform drawing in
add(sfx, fade_in(sine(3520, S(0.4)) * np.exp(-tt(S(0.4)) / 0.09), 2), 5.04, 0.3, 0.3)
n = S(0.35); add(sfx, svf_bp(noise(n), 300 * (12 ** (tt(n) / 0.35)), 2.0) * np.exp(-tt(n) / 0.2), T_MUSIC, 0.4)
# version flip v1 … v9: a shutter click each, a little success chord on v9
for i in range(NVER):
    tv = T_VER0 + T_VERD * i
    add(sfx, click(2200 + 160 * i, 0.008, 0.85, 0.04), tv, 0.8, -0.3 + 0.07 * i)
    add(sfx, blip(660 * 2 ** (2 * i / 12), 660 * 2 ** (2 * i / 12), 0.05, 0.5), tv, 0.3)
for k, m in enumerate((81, 85, 88)):
    n = S(0.5); add(sfx, fade_in(sine(mtof(m), n) * np.exp(-tt(n) / 0.18), 2), T_VER0 + T_VERD * (NVER - 1) + 0.03 * k, 0.28)
# cheer jump (boing up), kick (thud + spring back)
n = S(0.2); add(sfx, fade_in(square(300 * 2 ** (tt(n) / 0.2 * 1.5), n, 0.3) * np.exp(-tt(n) / 0.1), 2), T_CHEER, 0.3)
n = S(0.12); add(sfx, fade_in(np.tanh(2 * sine(120 - 400 * tt(n), n) * np.exp(-tt(n) / 0.05)), 2), T_KICK, 0.8)
n = S(0.3); add(sfx, svf_bp(noise(n), 3000 * (0.15 ** (tt(n) / 0.3)), 3.0) * np.exp(-tt(n) / 0.15), T_KICK + 0.02, 0.4, 0.4)
add(sfx, whoosh(0.34, 2500, 600, 1.5, 0.3), T_DASH0, 0.45, 0.4)
# fast-forward: the ident's own music, 10 s squeezed into 1 s (grains keep the pitch), plus a tape whirr
fn = os.path.join(HERE, '..', 'case', 'music.wav')
with wave.open(fn) as w:
    raw = np.frombuffer(w.readframes(w.getnframes()), '<i2').reshape(-1, 2).T / 32768.0
G = S(0.03); hop_out = S(0.012); span = T_PUSH1 - T_PUSH0
nout = S(span); ff = np.zeros((2, nout + G)); win = np.hanning(G)
for i in range(0, nout, hop_out):
    u = i / nout; src = int(np.clip(6.4 * u * SR, 0, raw.shape[1] - G - 1))
    ff[:, i:i + G] += raw[:, src:src + G] * win
ff = ff[:, :nout] * 0.45
ff[:, :S(0.02)] *= np.linspace(0, 1, S(0.02)); ff[:, -S(0.05):] *= np.linspace(1, 0, S(0.05))
add(sfx, ff, T_PUSH0, 1.0)
n = S(span); add(sfx, filt(noise(n), 'band', [900, 2400]) * (0.5 + 0.5 * np.sin(2 * np.pi * 34 * tt(n))) * np.sin(np.pi * tt(n) / span) * 0.6, T_PUSH0, 0.12)
# render: rising sweep with ticks, then a two-note "done"
n = S(T_REND1 - T_REND0); u = tt(n) / (T_REND1 - T_REND0)
add(sfx, sine(300 * 6 ** u, n) * u ** 1.5 * 0.4 + svf_bp(noise(n), 800 * 6 ** u, 2.0) * u, T_REND0, 0.3)
for i in range(10): add(sfx, click(3000, 0.004, 0.6, 0.02), T_REND0 + i * 0.04, 0.15)
for k, m in enumerate((88, 93)):
    n = S(0.4); add(sfx, fade_in(0.7 * sine(mtof(m), n) + 0.3 * square(mtof(m), n, 0.5) * np.exp(-tt(n) / 0.02), 2) * np.exp(-tt(n) / 0.12), T_REND1 + 0.07 * k, 0.35)
# the push into the monitor: a riser that ends exactly where the ident starts
n = S(DUR - T_ZOOM0); u = tt(n) / (DUR - T_ZOOM0)
ris = svf_bp(noise(n), 400 * 14 ** u, 2.2) * u ** 1.8 + 0.35 * sine(110 * 4 ** u, n) * u ** 2
ris = fade_out(ris, 8)
add(sfx, ris, T_ZOOM0, 0.55)

# victory jump after the render, glasses glint before the push
n = S(0.2); add(sfx, fade_in(square(330 * 2 ** (tt(n) / 0.2 * 1.5), n, 0.3) * np.exp(-tt(n) / 0.1), 2), 8.02, 0.28)
add(sfx, fade_in(sine(3520, S(0.35)) * np.exp(-tt(S(0.35)) / 0.08), 2), 8.36, 0.22, 0.3)

# ------------------------------------------------------------------ mix
fe = np.clip((8.08 - tt(N)) / 0.22, 0, 1) ** 1.5      # the bed fades out under the riser instead of stopping dead
# dynamics: the bed sits back during the version flip and takes a breath before the ×10 run
tN = tt(N); dyn = 1 - 0.45 * np.clip((tN - 5.2) / 0.1, 0, 1) * np.clip((6.2 - tN) / 0.1, 0, 1) - 0.7 * np.clip((tN - 6.22) / 0.04, 0, 1) * np.clip((T_PUSH0 - tN) / 0.05, 0, 1)
mus *= fe * dyn
env = maximum_filter1d(np.max(np.abs(sfx), axis=0), S(0.01))
a = np.exp(-1 / (SR * 0.1)); env = signal.lfilter([1 - a], [1, -a], env)
duck = 10 ** (-8.0 * np.clip(env / 0.3, 0, 1) / 20)
mix = mus * duck * 0.9 + sfx * 1.1
mix = filt(filt(mix, 'high', 40, 4), 'low', 14500, 2)

def lufs(x):
    b1, a1 = [1.53512485958697, -2.69169618940638, 1.19839281085285], [1.0, -1.69065929318241, 0.73248077421585]
    b2, a2 = [1.0, -2.0, 1.0], [1.0, -1.99004745483398, 0.99007225036621]
    y = signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=-1), axis=-1)
    blk, hop = S(0.4), S(0.1); ms = np.array([np.sum(np.mean(y[:, i:i + blk] ** 2, axis=1)) for i in range(0, x.shape[1] - blk + 1, hop)])
    lk = -0.691 + 10 * np.log10(ms + 1e-12); g = ms[lk > -70]; rel = -0.691 + 10 * np.log10(np.mean(g)) - 10
    return -0.691 + 10 * np.log10(np.mean(ms[(lk > -70) & (lk > rel)]))
def tp_db(x): return 20 * np.log10(np.max(np.abs(signal.resample_poly(x, 4, 1, axis=-1))) + 1e-12)
def limiter(x, ceil_db=-2.2):
    c = 10 ** (ceil_db / 20); up = np.abs(signal.resample_poly(x, 4, 1, axis=-1)).max(axis=0)
    pk = up.reshape(-1, 4).max(axis=1)[:x.shape[1]]; g = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    L = int(SR * 0.002); g = minimum_filter1d(g, 2 * L + 1); al = np.exp(-1 / (SR * 0.06)); out = np.empty_like(g); cur = 1.0
    for i in range(len(g)):
        gi = g[i]; cur = gi if gi < cur else gi + (cur - gi) * al; out[i] = cur
    return x * np.minimum(uniform_filter1d(out, L), g)
gain = 1.0
for _ in range(5): gain *= 10 ** ((-16.0 - lufs(limiter(mix * gain))) / 20)
y = limiter(mix * gain)
y[:, -S(0.004):] *= np.linspace(1, 0, S(0.004))
pcm = np.clip(np.round(y.T * 32767), -32768, 32767).astype('<i2')
with wave.open(os.path.join(HERE, 'making.wav'), 'wb') as w:
    w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm.tobytes())
print(f'wrote making.wav {DUR:.3f} s  {lufs(y):.2f} LUFS  TP {tp_db(y):.2f} dBTP')
