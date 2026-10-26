#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
娜乌斯嘉AI · 16 秒频道片头 —— 配乐与音效（纯 NumPy / SciPy 合成）
输出 music.wav：48 kHz / 16-bit / 立体声 / 正好 16.0 s，约 -14 LUFS，真峰值 <= -1.2 dBTP

120 BPM，4/4，一拍 0.5 s，一小节 2 s，共 8 小节。A 小调，Am–F–C–G。
所有时间点与 index.html 的画面一一对应（见 EVENTS 注释）。
"""
import numpy as np
from scipy import signal
import wave, os, sys

SR = 48000
DUR = 16.0
N = int(SR * DUR)
rng = np.random.default_rng(20261026)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'music.wav')

BEAT = 0.5
SIL = [(3.75, 4.0), (14.75, 15.0)]          # 两次静音留白

# ---------------------------------------------------------------- 基础工具
def mtof(m):
    return 440.0 * 2 ** ((np.asarray(m, dtype=float) - 69) / 12)

def zeros():
    return np.zeros((2, N))

def place(buf, sig, t0, gain=1.0, pan=0.0):
    """把单声道或立体声信号放到 t0（秒），等功率声像。"""
    i0 = int(round(t0 * SR))
    if sig.ndim == 1:
        a = np.pi / 4 * (pan + 1)
        st = np.vstack([sig * np.cos(a), sig * np.sin(a)]) * np.sqrt(2)
    else:
        st = sig
    if i0 < 0:
        st = st[:, -i0:]; i0 = 0
    n = min(st.shape[1], N - i0)
    if n > 0:
        buf[:, i0:i0 + n] += st[:, :n] * gain

def fade_in(x, ms=3.0):
    n = max(1, int(SR * ms / 1000))
    n = min(n, x.shape[-1])
    w = np.sin(np.linspace(0, np.pi / 2, n)) ** 2
    x = x.copy()
    x[..., :n] *= w
    return x

def fade_out(x, ms=5.0):
    n = max(1, int(SR * ms / 1000))
    n = min(n, x.shape[-1])
    w = np.cos(np.linspace(0, np.pi / 2, n)) ** 2
    x = x.copy()
    x[..., -n:] *= w
    return x

def tt(dur):
    return np.arange(int(SR * dur)) / SR

def blep(ph, dt):
    """polyBLEP 修正（相位 0..1，dt 每采样相位增量）。"""
    y = np.zeros_like(ph)
    m = ph < dt
    t = ph[m] / dt[m]
    y[m] = t + t - t * t - 1
    m2 = ph > 1 - dt
    t = (ph[m2] - 1) / dt[m2]
    y[m2] = t * t + t + t + 1
    return y

def saw_from_freq(freq, ph0=0.0):
    freq = np.asarray(freq, float)
    dt = freq / SR
    ph = (ph0 + np.cumsum(dt)) % 1.0
    return 2 * ph - 1 - blep(ph, dt)

def pulse_from_freq(freq, duty=0.5, ph0=0.0):
    dt = freq / SR
    ph = (ph0 + np.cumsum(dt)) % 1.0
    ph2 = (ph + 1 - duty) % 1.0
    s1 = 2 * ph - 1 - blep(ph, dt)
    s2 = 2 * ph2 - 1 - blep(ph2, dt)
    return 0.5 * (s1 - s2)

def lp(x, fc, order=2):
    sos = signal.butter(order, min(fc, SR * 0.45), 'low', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=-1)

def hp(x, fc, order=2):
    sos = signal.butter(order, fc, 'high', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=-1)

def bp(x, f1, f2, order=2):
    sos = signal.butter(order, [f1, min(f2, SR * 0.45)], 'band', fs=SR, output='sos')
    return signal.sosfilt(sos, x, axis=-1)

def sweep_filter(x, f_start, f_end, kind='low', q=0.9, block=128, curve='exp'):
    """时变滤波：逐块更新二阶滤波器系数并延续状态（用于扫频）。"""
    y = np.zeros_like(x)
    nb = int(np.ceil(len(x) / block))
    zi = np.zeros(2)
    for b in range(nb):
        u = b / max(1, nb - 1)
        fc = f_start * (f_end / f_start) ** u if curve == 'exp' else f_start + (f_end - f_start) * u
        fc = float(np.clip(fc, 30, SR * 0.45))
        w0 = 2 * np.pi * fc / SR
        al = np.sin(w0) / (2 * q)
        cw = np.cos(w0)
        if kind == 'low':
            bb = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        elif kind == 'high':
            bb = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        else:  # band (constant peak gain)
            bb = [al, 0, -al]
        aa = [1 + al, -2 * cw, 1 - al]
        bb = np.array(bb) / aa[0]; aa = np.array(aa) / aa[0]
        seg = x[b * block:(b + 1) * block]
        out, zi = signal.lfilter(bb, aa, seg, zi=zi)
        y[b * block:(b + 1) * block] = out
    return y

def peaking(x, f0, gain_db, q):
    A = 10 ** (gain_db / 40); w0 = 2 * np.pi * f0 / SR; al = np.sin(w0) / (2 * q); cw = np.cos(w0)
    b = np.array([1 + al * A, -2 * cw, 1 - al * A]); a = np.array([1 + al / A, -2 * cw, 1 - al / A])
    return signal.lfilter(b / a[0], a / a[0], x, axis=-1)

def highshelf(x, f0, gain_db, s=0.8):
    A = 10 ** (gain_db / 40); w0 = 2 * np.pi * f0 / SR; cw = np.cos(w0); sw = np.sin(w0)
    al = sw / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    b0 = A * ((A + 1) + (A - 1) * cw + 2 * np.sqrt(A) * al)
    b1 = -2 * A * ((A - 1) + (A + 1) * cw)
    b2 = A * ((A + 1) + (A - 1) * cw - 2 * np.sqrt(A) * al)
    a0 = (A + 1) - (A - 1) * cw + 2 * np.sqrt(A) * al
    a1 = 2 * ((A - 1) - (A + 1) * cw)
    a2 = (A + 1) - (A - 1) * cw - 2 * np.sqrt(A) * al
    return signal.lfilter(np.array([b0, b1, b2]) / a0, np.array([1, a1 / a0, a2 / a0]), x, axis=-1)

def make_ir(dur=1.4, decay=0.45, lpf=6000, seed=3):
    r = np.random.default_rng(seed)
    n = int(SR * dur); t = np.arange(n) / SR
    env = np.exp(-t / decay * 3.0) * (1 - np.exp(-t / 0.004))
    ir = np.vstack([r.standard_normal(n), r.standard_normal(n)]) * env
    ir = lp(ir, lpf)
    ir[:, :int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))  # 预延迟感
    ir /= np.sqrt(np.sum(ir ** 2, axis=1, keepdims=True))
    return ir

def reverb(st, ir, wet):
    out = np.vstack([signal.fftconvolve(st[0], ir[0])[:N], signal.fftconvolve(st[1], ir[1])[:N]])
    return out * wet

def env_ad(n, att=0.003, dec=0.2, shape=1.0):
    t = np.arange(n) / SR
    a = np.clip(t / att, 0, 1) ** 2 if att > 0 else np.ones(n)
    return a * np.exp(-t / dec) ** shape

def silence_gate():
    """两次留白的整体门：3.75/14.75 前 6 ms 收，4.0/15.0 后 3 ms 开。"""
    g = np.ones(N)
    t = np.arange(N) / SR
    for a, b in SIL:
        g = np.where((t >= a) & (t < b), 0.0, g)
        m = (t >= a - 0.006) & (t < a)
        g[m] = np.minimum(g[m], np.cos((t[m] - (a - 0.006)) / 0.006 * np.pi / 2) ** 2)
        m = (t >= b) & (t < b + 0.003)
        g[m] = np.minimum(g[m], np.sin((t[m] - b) / 0.003 * np.pi / 2) ** 2)
    return g

def in_sil(t0):
    return any(a - 1e-9 <= t0 < b for a, b in SIL)

# ---------------------------------------------------------------- 和声
#            Am             F              C              G
CHORDS = [[57, 60, 64], [57, 60, 65], [55, 60, 64], [55, 59, 62]]
ROOTS = [33, 29, 36, 31]                       # A1 F1 C2 G1
ARP_TONES = [[69, 72, 76], [65, 69, 72], [67, 72, 76], [67, 71, 74]]
MELODY = [[81, 76, 81, 79],                    # 旋律（每拍第一个十六分音符）：A–E–A 动机
          [77, 76, 72, 69],
          [79, 76, 84, 83],
          [74, 71, 74, 79]]
LOGO = [81, 88, 93]                            # 声音标识：A5–E6–A6（与旋律动机同源）

def chord_idx(t):
    return int(t // 2) % 4

# ---------------------------------------------------------------- 乐器
def pad_layer():
    """合成器和弦铺底：去谐锯齿 ×3 声部，暗 / 亮两版按段落能量交叉。"""
    out = zeros()
    bars = [(0, 2, 0), (2, 4, 1), (4, 6, 2), (6, 8, 3), (8, 10, 0), (10, 12, 1), (12, 14, 2), (14, 15, 3)]
    for (a, b, ci) in bars:
        for m in CHORDS[ci]:
            for ch in range(2):
                for dc in (-9, 0, 9):
                    det = dc + (4 if ch else -4)
                    f = mtof(m) * 2 ** (det / 1200)
                    dur = (b - a) + 0.25
                    n = int(SR * dur)
                    s = saw_from_freq(np.full(n, f), ph0=rng.random())
                    t = np.arange(n) / SR
                    env = np.clip(t / 0.06, 0, 1) * np.clip((b - a + 0.0 - t) / 0.12 + 1, 0, 1)
                    seg = s * env
                    tg = np.zeros((2, n)); tg[ch] = seg
                    place(out, tg, a, gain=0.055)
    # 15 s 终止和弦：Am 长衰减
    for m in [45, 57, 60, 64, 69]:
        for ch in range(2):
            for dc in (-8, 0, 8):
                f = mtof(m) * 2 ** ((dc + (3 if ch else -3)) / 1200)
                n = int(SR * 1.0)
                s = saw_from_freq(np.full(n, f), ph0=rng.random())
                env = env_ad(n, 0.003, 0.42)
                tg = np.zeros((2, n)); tg[ch] = s * env
                place(out, tg, 15.0, gain=0.07)
    dark = lp(out, 900, 2)
    bright = lp(out, 3200, 2)
    t = np.arange(N) / SR
    energy = np.interp(t, [0, 3.7, 4.0, 7.9, 8.0, 12.9, 13.0, 16], [0.15, 0.4, 0.55, 0.62, 0.75, 0.8, 0.9, 0.9])
    return dark * (1 - energy) + bright * energy

def arp_layer():
    """十六分音符琶音旋律：每拍第一音是旋律音（重音），后三音是和弦音。"""
    dry = zeros()
    for b in range(30):                       # 0..15 s 的拍
        tb = b * BEAT
        if tb >= 15.0:
            break
        ci = chord_idx(tb)
        mel = MELODY[ci][b % 4]
        tones = ARP_TONES[ci]
        notes = [mel, tones[0], tones[1], tones[2]]
        for k, m in enumerate(notes):
            t0 = tb + k * 0.125
            if in_sil(t0):
                continue
            vel = 1.0 if k == 0 else 0.55
            intro = 0.72 if t0 < 4 else 1.0
            f = float(mtof(m))
            n = int(SR * 0.32)
            t = np.arange(n) / SR
            s = np.zeros(n)
            for h in range(1, 18):
                if f * h > 15000:
                    break
                amp = (1.0 / h) * (1.0 if h % 2 else 0.45)
                bright = (0.7 if t0 < 4 else 1.0)
                s += amp * np.sin(2 * np.pi * f * h * t + rng.random() * 0.3) * np.exp(-t * (6 + h * 2.2 / bright))
            s *= env_ad(n, 0.003, 0.16 if k else 0.22)
            s = fade_out(s, 20)
            pan = [-0.0, -0.35, 0.35, 0.0][k]
            place(dry, s, t0, gain=0.2 * vel * intro, pan=pan)
    # 附点八分乒乓延迟
    d = int(0.375 * SR)
    wetL = np.zeros(N); wetR = np.zeros(N)
    src = lp(dry.sum(0) * 0.5, 4500, 1)
    fb = 0.38
    l = np.zeros(N); r = np.zeros(N)
    # 递推（向量化成几次叠加）
    tap = src.copy()
    for i in range(1, 6):
        sh = np.zeros(N); sh[d * i:] = tap[:N - d * i]
        g = fb ** (i - 1) * 0.42
        if i % 2:
            l += sh * g
        else:
            r += sh * g
    wet = np.vstack([l, r])
    return dry + wet

def bass_layer():
    """复古 Synthwave 八分音符贝斯（低 / 高八度交替），音高之间带滑音。"""
    t = np.arange(N) / SR
    eighth = 0.25
    tgt = np.zeros(N)
    gate = np.zeros(N)
    tn = np.zeros(N)
    for e in range(60):
        t0 = e * eighth
        if t0 >= 15.0:
            break
        i0, i1 = int(t0 * SR), int((t0 + eighth) * SR)
        ci = chord_idx(t0)
        root = ROOTS[ci]
        m = root if e % 2 == 0 else root + 12
        tgt[i0:i1] = m
        tn[i0:i1] = (np.arange(i1 - i0)) / SR
        if in_sil(t0):
            continue
        g = np.zeros(i1 - i0)
        L = int(0.205 * SR)
        g[:L] = 1.0
        g[:int(0.003 * SR)] = np.linspace(0, 1, int(0.003 * SR))
        g[L - int(0.012 * SR):L] *= np.linspace(1, 0, int(0.012 * SR))
        gate[i0:i1] = g
    # 15 s：终止长音 A1
    i0 = int(15.0 * SR)
    tgt[i0:] = 33
    tn[i0:] = (np.arange(N - i0)) / SR
    g = np.exp(-(np.arange(N - i0) / SR) / 0.5)
    g[:int(0.003 * SR)] *= np.linspace(0, 1, int(0.003 * SR))
    gate[i0:] = g
    tgt[tgt == 0] = 33
    # 滑音：对 MIDI 音高做 22 ms 一阶平滑
    a = np.exp(-1 / (SR * 0.022))
    pitch = signal.lfilter([1 - a], [1, -a], tgt, zi=[tgt[0] * a])[0]
    f = mtof(pitch)
    ph = np.cumsum(f / SR)
    s = np.zeros(N)
    for h in range(1, 26):
        w = np.exp(-tn * (3.0 + 2.4 * h)) * 0.8 + 0.2 * np.exp(-h * 0.35)
        w = np.where(f * h < 12000, w, 0)
        s += np.sin(2 * np.pi * h * ph) / h * w
    s += 0.55 * np.sin(2 * np.pi * ph)            # 次低音
    s = lp(s, 1400, 2)
    s *= gate
    return np.vstack([s, s]) * 0.135

def kick(heavy=1.0):
    n = int(SR * 0.6); t = np.arange(n) / SR
    f = 46 + 110 * np.exp(-t / 0.042) * heavy ** 0.3
    ph = np.cumsum(f / SR)
    body = np.sin(2 * np.pi * ph) * np.exp(-t / (0.30 * heavy ** 0.5))
    click = bp(rng.standard_normal(n), 1800, 7000) * np.exp(-t / 0.006) * 0.5
    knock = np.sin(2 * np.pi * 180 * t) * np.exp(-t / 0.025) * 0.35
    return fade_in(body + click + knock, 2)

def snare():
    n = int(SR * 0.45); t = np.arange(n) / SR
    body = (np.sin(2 * np.pi * 188 * t) * 0.7 + np.sin(2 * np.pi * 330 * t) * 0.35) * np.exp(-t / 0.08)
    nz = bp(rng.standard_normal(n), 1300, 9500) * np.exp(-t / 0.15)
    return fade_in(body * 0.8 + nz * 0.9, 2)

def hat(open_=False):
    n = int(SR * (0.2 if open_ else 0.09)); t = np.arange(n) / SR
    nz = hp(rng.standard_normal(n), 7200, 4)
    return fade_in(nz * np.exp(-t / (0.06 if open_ else 0.028)), 2)

def drums():
    out = zeros(); snr = zeros()
    t = 0.0
    for b in range(31):
        tb = b * BEAT
        if tb >= 4.0 - 1e-9 and tb <= 15.0 + 1e-9 and not in_sil(tb):
            heavy = 1.0
            if abs(tb - 15.0) < 1e-6: heavy = 1.6
            place(out, kick(heavy), tb, gain=0.40 * (1.3 if heavy > 1 else 1.0))
            if (tb - 4.0) % 1.0 > 0.49 and tb < 14.75:
                place(snr, snare(), tb, gain=0.30)
    for e in range(60):
        te = e * 0.25
        if te >= 14.75 or in_sil(te):
            continue
        off = e % 2 == 1
        place(out, hat(open_=False), te, gain=(0.075 if off else 0.045) * (0.85 if te < 4 else 1.0), pan=0.25 if off else -0.15)
    return out, snr

# ---------------------------------------------------------------- 音效
def bell(f, dur=1.2, bright=1.0):
    n = int(SR * dur); t = np.arange(n) / SR
    s = np.zeros(n)
    for r, a, d in [(1, 1, 0.9), (2.0, 0.45, 0.5), (3.01, 0.22, 0.32), (4.18, 0.14 * bright, 0.18), (5.43, 0.08 * bright, 0.12)]:
        if f * r < 16000:
            s += a * np.sin(2 * np.pi * f * r * t) * np.exp(-t / d)
    return fade_in(s, 2)

def logo(t0, out, gain=0.21):
    for i, m in enumerate(LOGO):
        s = bell(float(mtof(m)), 1.4)
        place(out, s, t0 + i * 0.125, gain=gain * [0.9, 0.95, 1.1][i], pan=[-0.25, 0.25, 0.0][i])

def sfx_bus():
    sfx = zeros()
    rev_send = zeros()

    # ---- 4.0 太阳炸开：低频冲击 + 玻璃碎裂 + 像素四散沙沙
    n = int(SR * 1.2); t = np.arange(n) / SR
    f = 38 + 70 * np.exp(-t / 0.09)
    boom = np.sin(2 * np.pi * np.cumsum(f / SR)) * np.exp(-t / 0.42)
    punch = lp(rng.standard_normal(n), 1800, 2) * np.exp(-t / 0.05) * 0.9
    crack = bp(rng.standard_normal(n), 900, 5000) * np.exp(-t / 0.018) * 0.7
    place(sfx, fade_in(boom * 1.0 + punch + crack, 2), 4.0, gain=0.62)
    # 玻璃：随机高频非谐分音，前 0.3 s 内密集出现
    for k in range(46):
        tk = 4.0 + 0.32 * (rng.random() ** 1.6)
        fk = 2200 + 6500 * rng.random()
        nk = int(SR * 0.4); tt_ = np.arange(nk) / SR
        d = 0.03 + 0.12 * rng.random()
        s = (np.sin(2 * np.pi * fk * tt_) + 0.5 * np.sin(2 * np.pi * fk * 1.53 * tt_)) * np.exp(-tt_ / d)
        place(sfx, fade_in(s, 2), tk, gain=0.05 * (1.2 - (tk - 4.0) * 2), pan=rng.uniform(-0.8, 0.8))
    gl = hp(rng.standard_normal(int(SR * 0.25)), 3500, 2) * np.exp(-np.arange(int(SR * 0.25)) / SR / 0.06)
    place(sfx, fade_in(gl, 2), 4.0, gain=0.22)
    # 沙沙：稀疏颗粒 + 低通从 12 kHz 扫到 900 Hz，0.8 s 收住
    n = int(SR * 0.8); t = np.arange(n) / SR
    grains = (rng.random(n) < 0.06).astype(float) * rng.standard_normal(n)
    grains = grains + 0.25 * rng.standard_normal(n)
    sand = sweep_filter(grains, 12000, 900, 'low', 0.8)
    env = np.clip(t / 0.01, 0, 1) * (1 - t / 0.8) ** 1.6
    sandL = sand * env
    sandR = sweep_filter(np.roll(grains, 917), 12000, 900, 'low', 0.8) * env
    place(sfx, fade_in(np.vstack([sandL, sandR]), 3), 4.02, gain=0.30)

    # ---- 4.4–6.5 吸入“嗖嗖”：越来越密，从左往右
    times = []
    tk = 4.42
    while tk < 6.42:
        times.append(tk)
        u = (tk - 4.42) / 2.0
        tk += 0.30 * (1 - u) + 0.055 * u
    for i, tk in enumerate(times):
        u = (tk - 4.42) / 2.0
        dur = 0.24 - 0.08 * u
        n = int(SR * dur); t = np.arange(n) / SR
        nz = rng.standard_normal(n)
        s = sweep_filter(nz, 700 + 500 * u, 4200 + 1500 * u, 'band', 2.2)
        env = np.sin(np.pi * np.clip(t / dur, 0, 1) ** 0.6) ** 2
        s = s * env
        pan = -0.7 + 1.4 * ((i * 0.37) % 1.0)
        place(sfx, fade_in(s, 3), tk, gain=0.20 + 0.16 * u, pan=pan)
    # 底下一层持续变密的吸气床
    n = int(SR * 2.1); t = np.arange(n) / SR
    bed = sweep_filter(rng.standard_normal(n), 500, 3000, 'band', 1.2)
    bed *= (t / 2.1) ** 1.5 * np.clip((2.1 - t) / 0.05, 0, 1)
    place(sfx, fade_in(bed, 5), 4.42, gain=0.10)
    # 每批像素落定：轻“咔哒”（5.0 起每 1/32 音符一批，共 25 批）
    for g in range(25):
        tg = 5.0 + 0.0625 * g
        n = int(SR * 0.04); t = np.arange(n) / SR
        c = np.sin(2 * np.pi * (3000 + 400 * ((g * 7) % 5)) * t) * np.exp(-t / 0.006)
        c += bp(rng.standard_normal(n), 2500, 9000) * np.exp(-t / 0.003) * 0.6
        on_beat = abs((tg * 2) - round(tg * 2)) < 1e-6
        x = 368 + (g + 0.5) / 25 * 1164
        place(sfx, fade_in(c, 2), tg, gain=0.40 * (1.5 if on_beat else 1.0), pan=(x - 960) / 960 * 0.8)

    # ---- 7.0–8.0 反向混响“吸气”
    src = zeros()
    n = int(SR * 0.35); t = np.arange(n) / SR
    chord = sum(saw_from_freq(np.full(n, float(mtof(m)))) for m in [57, 64, 69, 72, 76]) * np.exp(-t / 0.15)
    chord = lp(chord, 3000) + 0.6 * hp(rng.standard_normal(n), 2000) * np.exp(-t / 0.08)
    s0 = np.zeros(int(SR * 1.6)); s0[:n] = chord
    ir = make_ir(1.5, 0.9, 7000, seed=11)
    wl = signal.fftconvolve(s0, ir[0])[:int(SR * 1.6)]
    wr = signal.fftconvolve(s0, ir[1])[:int(SR * 1.6)]
    rv = np.vstack([wl, wr])[:, ::-1]
    L = int(SR * 1.0)
    rv = rv[:, -L:]
    rv = rv / np.max(np.abs(rv))
    rv *= np.linspace(0, 1, L) ** 1.4
    rv = fade_out(fade_in(rv, 20), 4)
    place(sfx, rv, 7.0, gain=0.6)

    # ---- 8.0 故障：重击 + 两下数字噪声（第二下更弱）
    n = int(SR * 0.9); t = np.arange(n) / SR
    f = 38 + 85 * np.exp(-t / 0.07)
    hit = np.sin(2 * np.pi * np.cumsum(f / SR)) * np.exp(-t / 0.33)
    hit += lp(rng.standard_normal(n), 2500) * np.exp(-t / 0.04) * 0.9
    hit += bp(rng.standard_normal(n), 1000, 6000) * np.exp(-t / 0.015) * 0.8
    place(sfx, fade_in(hit, 2), 8.0, gain=0.62)
    def digital(dur, seed):
        r = np.random.default_rng(seed)
        n = int(SR * dur); out = np.zeros(n)
        seg = int(SR / 30)
        for i0 in range(0, n, seg):
            m = min(seg, n - i0)
            kind = r.integers(3)
            if kind == 0:   # 采样保持噪声
                hold = int(r.integers(3, 40))
                v = np.repeat(r.uniform(-1, 1, m // hold + 2), hold)[:m]
            elif kind == 1: # 方波蜂鸣
                fq = r.uniform(90, 1600)
                v = np.sign(np.sin(2 * np.pi * fq * np.arange(m) / SR)) * 0.8
            else:
                v = r.uniform(-1, 1, m)
            out[i0:i0 + m] = v * r.uniform(0.6, 1.0)
        out = np.round(out * 6) / 6                  # 降位深
        out = bp(out, 300, 9000)
        return out
    g1 = digital(0.183, 81); g2 = digital(0.167, 82)
    g1 *= np.linspace(1, 0.8, len(g1)); g2 *= np.linspace(1, 0.8, len(g2))
    place(sfx, fade_out(fade_in(np.vstack([g1, np.roll(g1, 37)]), 2), 6), 8.0, gain=0.34)
    place(sfx, fade_out(fade_in(np.vstack([np.roll(g2, 51), g2]), 2), 6), 8.2333, gain=0.2)
    n = int(SR * 0.12); t = np.arange(n) / SR
    tk2 = bp(rng.standard_normal(n), 800, 5000) * np.exp(-t / 0.02)
    place(sfx, fade_in(tk2, 2), 8.2333, gain=0.25)

    # ---- 8.1–8.5 主角飞向“娜”上方 + 8.5 放大
    n = int(SR * 0.4); t = np.arange(n) / SR
    zip_ = sweep_filter(rng.standard_normal(n), 1200, 5000, 'band', 2.5) * np.sin(np.pi * t / 0.4) ** 2
    place(sfx, fade_in(zip_, 3), 8.1, gain=0.12, pan=-0.3)
    n = int(SR * 0.26); t = np.arange(n) / SR
    fq = 300 * 2 ** (t / 0.26 * 2.0)
    grow = pulse_from_freq(fq, 0.25) * np.exp(-t / 0.18)
    place(sfx, fade_in(lp(grow, 6000), 3), 8.5, gain=0.5, pan=-0.45)

    # ---- 9.0–11.5 每落一个字一声 8-bit，音阶上行；12.0 变句号“叮”
    steps = [72, 74, 76, 79, 81, 84]                 # C D E G A C（五声音阶上行）
    for k, m in enumerate(steps):
        tk = 9.0 + 0.5 * k
        n = int(SR * 0.18); t = np.arange(n) / SR
        f0 = float(mtof(m))
        fq = f0 * 2 ** (-2 / 12 * np.exp(-t / 0.012))
        s = pulse_from_freq(fq, 0.25) * np.exp(-t / 0.075)
        s = lp(s, 9000)
        thump = np.sin(2 * np.pi * np.cumsum((90 + 120 * np.exp(-t / 0.02)) / SR)) * np.exp(-t / 0.06)
        x = [479, 719, 959, 1199, 1400, 1520][k]
        place(sfx, fade_in(s, 2), tk, gain=0.42, pan=(x - 960) / 960 * 0.7)
        place(sfx, fade_in(thump, 2), tk, gain=0.22)
    ding = bell(2637.0, 1.0, 1.4)
    n = len(ding); t = np.arange(n) / SR
    ding += 0.4 * np.sin(2 * np.pi * 5274 * t) * np.exp(-t / 0.08)
    place(sfx, ding, 12.0, gain=0.15, pan=0.5)
    logo(12.0, sfx, 0.17)

    # ---- 12.05–12.9 光扫：一层很轻的“闪”从左到右
    n = int(SR * 0.85); t = np.arange(n) / SR
    sh = sweep_filter(rng.standard_normal(n), 3000, 9000, 'band', 3.0) * np.sin(np.pi * t / 0.85) ** 2
    panv = np.linspace(-0.8, 0.8, n); a = np.pi / 4 * (panv + 1)
    place(sfx, fade_in(np.vstack([sh * np.cos(a), sh * np.sin(a)]) * 1.4, 4), 12.05, gain=0.07)

    # ---- 12.62–13.0 短促上扬
    n = int(SR * 0.38); t = np.arange(n) / SR
    fq = 220 * 2 ** (t / 0.38 * 3.0)
    rise = saw_from_freq(fq) * 0.5 + sweep_filter(rng.standard_normal(n), 800, 8000, 'band', 2.0)
    rise = lp(rise, 7000) * (t / 0.38) ** 2
    place(sfx, fade_out(fade_in(rise, 4), 4), 12.62, gain=0.16)

    # ---- 13.0 强拍：标题上移 + 副标题出现
    n = int(SR * 0.6); t = np.arange(n) / SR
    imp = np.sin(2 * np.pi * np.cumsum((45 + 90 * np.exp(-t / 0.05)) / SR)) * np.exp(-t / 0.25)
    imp += bp(rng.standard_normal(n), 600, 7000) * np.exp(-t / 0.05) * 0.7
    place(sfx, fade_in(imp, 2), 13.0, gain=0.42)
    # 乱码闪动的细碎数据声
    for i in range(13):
        tk = 13.0 + 0.05 * i
        n = int(SR * 0.025); t = np.arange(n) / SR
        fq = 1500 + 2500 * rng.random()
        s = pulse_from_freq(np.full(n, fq), 0.5) * np.exp(-t / 0.008)
        place(sfx, fade_in(s, 2), tk, gain=0.018, pan=rng.uniform(-0.6, 0.6))
    # 每锁定一个字：轻“嗒”
    for j in range(11):
        tk = 13.2 + 0.045 * j
        n = int(SR * 0.05); t = np.arange(n) / SR
        s = np.sin(2 * np.pi * 1900 * t) * np.exp(-t / 0.01) + bp(rng.standard_normal(n), 2000, 8000) * np.exp(-t / 0.004) * 0.7
        place(sfx, fade_in(s, 2), tk, gain=0.24, pan=-0.6 + 1.2 * j / 10)
    logo(13.7, sfx, 0.17)

    # ---- 14.0–14.75 句点呼吸
    n = int(SR * 0.75); t = np.arange(n) / SR
    br = lp(rng.standard_normal(n), 1800) * 0.5 + np.sin(2 * np.pi * 880 * t) * 0.25
    br *= np.sin(np.pi * t / 0.75) ** 2
    place(sfx, fade_out(fade_in(br, 5), 6), 14.0, gain=0.09, pan=0.5)

    # ---- 15.0 显像管关机“噗”
    n = int(SR * 0.9); t = np.arange(n) / SR
    puh = np.sin(2 * np.pi * np.cumsum((32 + 100 * np.exp(-t / 0.05)) / SR)) * np.exp(-t / 0.22)
    puh += lp(rng.standard_normal(n), 1200) * np.exp(-t / 0.07) * 0.9
    stat = bp(rng.standard_normal(n) * (rng.random(n) < 0.3), 1500, 7000) * np.exp(-t / 0.18) * 0.6
    whine = np.sin(2 * np.pi * np.cumsum((5200 * np.exp(-t / 0.14) + 180) / SR)) * np.exp(-t / 0.2) * 0.22
    place(sfx, fade_in(puh + stat + whine, 2), 15.0, gain=0.6)
    # 15.5 句点闪一下
    n = int(SR * 0.3); t = np.arange(n) / SR
    bl = np.sin(2 * np.pi * 2637 * t) * np.exp(-t / 0.07) + 0.4 * np.sin(2 * np.pi * 3951 * t) * np.exp(-t / 0.04)
    place(sfx, fade_in(bl, 2), 15.5, gain=0.08, pan=0.5)
    return sfx

# ---------------------------------------------------------------- 响度
def k_weight(x):
    b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]; a1 = [1.0, -1.69065929318241, 0.73248077421585]
    b2 = [1.0, -2.0, 1.0]; a2 = [1.0, -1.99004745483398, 0.99007225036621]
    return signal.lfilter(b2, a2, signal.lfilter(b1, a1, x, axis=-1), axis=-1)

def lufs(x):
    y = k_weight(x)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    z = []
    for i in range(0, x.shape[1] - blk + 1, hop):
        z.append(np.sum(np.mean(y[:, i:i + blk] ** 2, axis=1)))
    z = np.array(z)
    L = -0.691 + 10 * np.log10(z + 1e-20)
    z1 = z[L > -70]
    Lr = -0.691 + 10 * np.log10(np.mean(z1)) - 10
    z2 = z[(L > -70) & (L > Lr)]
    return -0.691 + 10 * np.log10(np.mean(z2))

def true_peak(x):
    up = signal.resample_poly(x, 4, 1, axis=-1)
    return 20 * np.log10(np.max(np.abs(up)) + 1e-20)

def limiter(x, ceil_db=-1.6, look=0.002, rel=0.08):
    c = 10 ** (ceil_db / 20)
    up = signal.resample_poly(x, 4, 1, axis=-1)
    pk = np.max(np.abs(up), axis=0).reshape(-1, 4).max(axis=1)[:x.shape[1]]
    need = np.minimum(1.0, c / np.maximum(pk, 1e-9))
    la = int(look * SR)
    # 前视：窗口最小值
    from scipy.ndimage import minimum_filter1d
    g = minimum_filter1d(need, size=2 * la + 1, mode='nearest')
    # 释放平滑（只在增益回升时慢）
    a = np.exp(-1 / (rel * SR))
    out = np.empty_like(g); cur = 1.0
    for i in range(len(g)):
        v = g[i]
        cur = v if v < cur else v + (cur - v) * a
        out[i] = cur
    # 平滑起始段，避免突变
    out = signal.filtfilt(*signal.butter(1, 400, fs=SR), out)
    out = np.minimum(out, minimum_filter1d(need, size=2 * la + 1, mode='nearest') * 1.0 + 0.0)
    return x * out

# ---------------------------------------------------------------- 混音
def render():
    """返回 (music, sfx, mix)；music / sfx 为让位后、母带前的两路。"""
    print('synth music ...', flush=True)
    pad = pad_layer()
    arp = arp_layer()
    bass = bass_layer()
    dr, snr = drums()
    t = np.arange(N) / SR
    # 侧链：4 s 起底鼓压一下 pad / 贝斯
    sc = np.ones(N)
    for b in range(8, 31):
        tb = b * BEAT
        if in_sil(tb) or tb > 15.0:
            continue
        m = t >= tb
        sc[m] *= 1 - 0.38 * np.exp(-(t[m] - tb) / 0.11)
    music = pad * sc + arp * (0.6 + 0.4 * sc) + bass * (0.55 + 0.45 * sc) + dr + snr
    ir = make_ir(1.3, 0.6, 6500, seed=5)
    music = music + reverb(snr * 0.8 + pad * 0.25 + arp * 0.3, ir, 0.22)

    print('synth sfx ...', flush=True)
    sfx = sfx_bus()
    ir2 = make_ir(1.0, 0.45, 8000, seed=9)
    sfx = sfx + reverb(sfx, ir2, 0.12)

    # 让位：音效响起时音乐压低（最多约 -7 dB）
    env = np.abs(sfx).max(axis=0)
    a_att = np.exp(-1 / (0.004 * SR)); a_rel = np.exp(-1 / (0.16 * SR))
    e = np.zeros(N); cur = 0.0
    for i in range(N):
        v = env[i]
        cur = v + (cur - v) * (a_att if v > cur else a_rel)
        e[i] = cur
    la = int(0.006 * SR)                      # 6 ms 前视：音效起音前音乐先让开
    e = np.concatenate([e[la:], np.full(la, e[-1])])
    duck = 1 - 0.6 * np.clip(e / 0.12, 0, 1)
    music = music * duck

    return music, sfx, music + sfx

def main():
    music, sfx, mix = render()
    t = np.arange(N) / SR
    mix = mix * silence_gate()
    # 结尾平滑衰减到静音
    tail = np.ones(N)
    m = t >= 15.35
    tail[m] = np.cos(np.clip((t[m] - 15.35) / (15.97 - 15.35), 0, 1) * np.pi / 2) ** 2
    mix *= tail
    # 母带：40 Hz 以下衰减；4–8 kHz 补亮度
    mix = hp(mix, 40, 4)
    mix = peaking(mix, 5800, 2.5, 0.7)
    mix = highshelf(mix, 9000, 1.0)
    mix *= silence_gate()          # 滤波器尾巴也清干净
    mix *= tail

    # 响度 -14 LUFS + 真峰值限制
    target = -14.0
    for it in range(6):
        L = lufs(mix)
        mix *= 10 ** ((target - L) / 20)
        tp = true_peak(mix)
        if tp <= -1.35:
            break
        mix = limiter(mix, -1.6)
    L = lufs(mix); tp = true_peak(mix)
    if tp > -1.25:
        mix *= 10 ** ((-1.3 - tp) / 20)
    mix[:, -int(0.03 * SR):] = 0.0
    L = lufs(mix); tp = true_peak(mix)
    print(f'integrated loudness {L:.2f} LUFS, true peak {tp:.2f} dBTP', flush=True)

    pcm = np.clip(np.round(mix.T * 32767), -32768, 32767).astype('<i2')
    with wave.open(OUT, 'wb') as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    print('wrote', OUT, pcm.shape[0] / SR, 's')

if __name__ == '__main__':
    main()
