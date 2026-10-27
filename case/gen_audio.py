#!/usr/bin/env python3
"""娜乌斯嘉AI · 16 秒频道片头 —— 配乐与音效（NumPy / SciPy 合成）

输出 music.wav：48 kHz、16-bit、立体声、正好 16.0 秒，约 -14 LUFS，真峰值 <= -1.2 dBTP。
120 BPM，4/4，A 小调 Am–F–C–G。所有时间点与 index.html 的画面动作一一对应。
"""
import os
import sys
import numpy as np
from scipy import signal

SR = 48000
DUR = 16.0
N = int(SR * DUR)
BEAT = 0.5
RNG = np.random.default_rng(20261027)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'music.wav')

SILENCES = [(3.75, 4.0), (14.75, 15.0)]          # 全片仅有的两次静音留白


# ------------------------------------------------------------------ 基础工具
def midi(m):
    return 440.0 * 2 ** ((m - 69) / 12.0)


def tarr(n):
    return np.arange(n) / SR


def fade_edges(x, a_ms=3.0, r_ms=4.0):
    """所有声音的起音 2–5 ms 淡入、尾部淡出，杜绝咔哒声"""
    n = len(x)
    a = min(n, max(1, int(SR * a_ms / 1000)))
    r = min(n, max(1, int(SR * r_ms / 1000)))
    x = x.copy()
    x[:a] *= np.sin(np.linspace(0, np.pi / 2, a)) ** 2
    x[n - r:] *= np.cos(np.linspace(0, np.pi / 2, r)) ** 2
    return x


def place(bus, x, t0, gain=1.0, pan=0.0):
    """把单声道 / 立体声片段放到总线上（等功率声像）"""
    i0 = int(round(t0 * SR))
    if x.ndim == 1:
        th = (pan + 1) * np.pi / 4
        x = np.stack([x * np.cos(th), x * np.sin(th)], axis=1) * np.sqrt(2)
    if i0 < 0:
        x = x[-i0:]
        i0 = 0
    n = min(len(x), N - i0)
    if n > 0:
        bus[i0:i0 + n] += x[:n] * gain


def blep_osc(freq, kind='saw', phase0=0.0):
    """带 PolyBLEP 抗混叠的锯齿 / 方波，freq 可以是逐采样数组"""
    freq = np.broadcast_to(np.asarray(freq, dtype=float), np.shape(freq) if np.ndim(freq) else (1,)).astype(float)
    dt = freq / SR
    ph = (np.cumsum(dt) + phase0) % 1.0

    def blep(p, d):
        y = np.zeros_like(p)
        m = p < d
        tt = p[m] / d[m]
        y[m] = tt + tt - tt * tt - 1
        m2 = p > 1 - d
        tt = (p[m2] - 1) / d[m2]
        y[m2] = tt * tt + tt + tt + 1
        return y

    saw = 2 * ph - 1 - blep(ph, dt)
    if kind == 'saw':
        return saw
    ph2 = (ph + 0.5) % 1.0
    saw2 = 2 * ph2 - 1 - blep(ph2, dt)
    return saw - saw2  # 方波


def sine_sweep(f, n=None):
    f = np.asarray(f, dtype=float)
    return np.sin(2 * np.pi * np.cumsum(f) / SR)


def sos_lp(f, order=2):
    return signal.butter(order, min(f, SR * 0.45), 'low', fs=SR, output='sos')


def sos_hp(f, order=2):
    return signal.butter(order, f, 'high', fs=SR, output='sos')


def sos_bp(lo, hi, order=2):
    return signal.butter(order, [lo, min(hi, SR * 0.45)], 'band', fs=SR, output='sos')


def filt(sos, x):
    return signal.sosfilt(sos, x, axis=0)


def sweep_lp(x, f_start, f_end, block=256, curve='exp'):
    """时变低通（分块更新系数，保持状态）"""
    y = np.zeros_like(x)
    nb = int(np.ceil(len(x) / block))
    zi = None
    for b in range(nb):
        u = b / max(1, nb - 1)
        fc = f_start * (f_end / f_start) ** u if curve == 'exp' else f_start + (f_end - f_start) * u
        sos = sos_lp(fc, 2)
        if zi is None:
            zi = np.zeros((sos.shape[0], 2))
        seg = x[b * block:(b + 1) * block]
        y[b * block:(b + 1) * block], zi = signal.sosfilt(sos, seg, zi=zi)
    return y


def sweep_bp(x, f_start, f_end, q=1.2, block=256):
    y = np.zeros_like(x)
    nb = int(np.ceil(len(x) / block))
    zi = np.zeros((1, 2))
    for b in range(nb):
        u = b / max(1, nb - 1)
        fc = f_start * (f_end / f_start) ** u
        bw = fc / q
        sos = signal.butter(1, [max(20, fc - bw / 2), min(SR * 0.45, fc + bw / 2)], 'band', fs=SR, output='sos')
        seg = x[b * block:(b + 1) * block]
        y[b * block:(b + 1) * block], zi = signal.sosfilt(sos, seg, zi=zi)
    return y


def env_exp(n, tau, attack=0.003):
    tt = tarr(n)
    e = np.exp(-tt / tau)
    a = int(attack * SR)
    if a > 0:
        e[:a] *= np.linspace(0, 1, a)
    return e


def norm(x, peak=1.0):
    m = np.max(np.abs(x))
    return x * (peak / m) if m > 0 else x


def make_ir(sec=1.6, tau=0.45, lp_end=2500, seed=3):
    r = np.random.default_rng(seed)
    n = int(sec * SR)
    tt = tarr(n)
    ir = r.standard_normal((n, 2)) * np.exp(-tt / tau)[:, None]
    ir[:, 0] = sweep_lp(ir[:, 0], 9000, lp_end, block=1024)
    ir[:, 1] = sweep_lp(ir[:, 1], 9000, lp_end, block=1024)
    ir[: int(0.012 * SR)] *= np.linspace(0, 1, int(0.012 * SR))[:, None]
    return ir / np.sqrt(np.sum(ir ** 2) / 2)


IR = make_ir()


def reverb(x, wet=0.2, ir=IR):
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    y = np.stack([signal.fftconvolve(x[:, c], ir[:, c])[: len(x)] for c in range(2)], axis=1)
    return x + y * wet


# ------------------------------------------------------------------ 和声
BARS = ['Am', 'F', 'C', 'G', 'Am', 'F', 'C', 'G']
ROOT = {'Am': 33, 'F': 29, 'C': 36, 'G': 31}               # 贝斯根音（A1 F1 C2 G1）
PAD = {'Am': [57, 60, 64, 69], 'F': [53, 57, 60, 64], 'C': [55, 60, 64, 67], 'G': [55, 59, 62, 67]}
ARPT = {'Am': [57, 60, 64, 69, 72], 'F': [53, 57, 60, 65, 69], 'C': [55, 60, 64, 67, 72], 'G': [55, 59, 62, 67, 71]}


def music_stop(t):
    """音乐声部在静音段与 15 秒后的可用性"""
    for a, b in SILENCES:
        if a <= t < b:
            return False
    return t < 14.75


# ------------------------------------------------------------------ 音乐声部
def make_pad():
    out = np.zeros((N, 2))
    for b, ch in enumerate(BARS):
        t0 = b * 2.0
        if t0 >= 14.75:
            break
        dur = min(2.0, 14.75 - t0) + 0.35
        if t0 < 3.75 <= t0 + dur:
            dur = 3.75 - t0
        n = int(dur * SR)
        tt = tarr(n)
        att = 0.05 if t0 in (4.0, 8.0) else 0.22
        e = np.minimum(1, tt / att) * np.minimum(1, np.maximum(0, (dur - tt) / 0.3))
        e = e ** 1.5
        sig = np.zeros((n, 2))
        for k, m in enumerate(PAD[ch]):
            for d, pan in ((-0.09, -0.7), (0.0, 0.0), (0.085, 0.7)):
                f = midi(m) * 2 ** (d / 12) * (1 + 0.0015 * np.sin(2 * np.pi * (0.3 + 0.1 * k) * tt))
                s = blep_osc(f, 'saw', phase0=RNG.random())
                th = (pan + 1) * np.pi / 4
                sig[:, 0] += s * np.cos(th)
                sig[:, 1] += s * np.sin(th)
        sig *= e[:, None]
        place(out, sig, t0)
    muff = filt(sos_lp(420, 2), out)
    opn = filt(sos_lp(1500, 2), out)
    tt = tarr(N)
    k = (tt >= 4.0).astype(float)
    pad = muff * (1 - k)[:, None] * 0.7 + opn * k[:, None]
    # 侧链“泵”感：4 秒后跟着底鼓呼吸
    pump = np.ones(N)
    for i in range(32):
        tb = i * BEAT
        if tb < 4.0 or tb >= 14.75:
            continue
        i0 = int(tb * SR)
        n = int(0.42 * SR)
        x = np.arange(n) / SR
        pump[i0:i0 + n] = np.minimum(pump[i0:i0 + n], 1 - 0.45 * np.exp(-x / 0.12))
    pad *= pump[:, None]
    return pad


ARP_PAT = [0, 2, 3, 4, 2, 3, 1, 2, 0, 2, 3, 4, 3, 2, 1, 2]


def make_arp():
    out = np.zeros((N, 2))
    step = BEAT / 2 / 2  # 十六分音符 0.125 s
    for b, ch in enumerate(BARS):
        for s in range(16):
            t0 = b * 2.0 + s * step
            if not music_stop(t0):
                continue
            tone = ARPT[ch][ARP_PAT[s]]
            acc = 1.0 if s in (3, 11) else (0.8 if s % 4 == 0 else 0.62)
            n = int(0.24 * SR)
            if t0 < 3.75 and t0 + 0.24 > 3.75:
                n = int((3.75 - t0) * SR)
            if t0 + n / SR > 14.75:
                n = int((14.75 - t0) * SR)
            if n < 64:
                continue
            tt = tarr(n)
            f = midi(tone)
            s1 = blep_osc(np.full(n, f), 'sq') * 0.55 + blep_osc(np.full(n, f * 1.004), 'saw') * 0.45
            fc_env = 700 + 3800 * np.exp(-tt / 0.06)
            # 简易滤波包络：两段固定滤波按包络混合
            bright = filt(sos_lp(4200, 2), s1)
            dark = filt(sos_lp(800, 2), s1)
            w = np.clip((fc_env - 800) / 3400, 0, 1)
            x = bright * w + dark * (1 - w)
            x *= np.exp(-tt / 0.16) * acc
            x = fade_edges(x, 2.5, 6)
            place(out, x, t0, 1.0, pan=0.25 * np.sin(s * 0.7 + b))
    # 0–4 秒压低做闷
    tt = tarr(N)
    k = (tt >= 4.0).astype(float)
    muff = filt(sos_lp(900, 2), out) * 0.55
    arp = muff * (1 - k)[:, None] + out * k[:, None]
    # 乒乓延迟（附点八分）
    d = int(0.375 * SR)
    wet = np.zeros_like(arp)
    fb = 0.3
    src = arp.copy()
    for rep in range(1, 5):
        sh = d * rep
        g = fb ** (rep - 1) * 0.32
        ch = rep % 2
        wet[sh:, ch] += src[:-sh, 0 if ch else 1] * g if sh < N else 0
    wet = filt(sos_lp(3000, 1), wet)
    arp = arp + wet
    return arp


def make_bass():
    out = np.zeros(N)
    # 每个八分音符一个音；根音为主，偶尔跳八度；换和弦时带 45 ms 滑音
    seq = []
    for b, ch in enumerate(BARS):
        for e8 in range(8):
            t0 = b * 2.0 + e8 * 0.25
            if t0 < 4.0 or t0 >= 14.75:
                continue
            m = ROOT[ch] + (12 if e8 in (3, 7) else 0)
            seq.append((t0, m))
    freq = np.zeros(N)
    gate = np.zeros(N)
    for i, (t0, m) in enumerate(seq):
        i0 = int(t0 * SR)
        i1 = int(min(t0 + 0.25, 14.75) * SR)
        f1 = midi(m)
        f0 = midi(seq[i - 1][1]) if i > 0 else f1
        n = i1 - i0
        x = np.arange(n) / SR
        glide = 0.045 if (i > 0 and seq[i - 1][1] % 12 != m % 12) else 0.018
        fr = f1 * (f0 / f1) ** np.exp(-x / glide * 3) if i > 0 else np.full(n, f1)
        freq[i0:i1] = fr
        ge = np.minimum(1, x / 0.004) * np.minimum(1, np.maximum(0, (0.235 - x) / 0.02))
        gate[i0:i1] = ge * (0.85 + 0.15 * np.exp(-x / 0.05))
    freq[freq == 0] = 55.0
    s = blep_osc(freq, 'saw') * 0.7 + blep_osc(freq * 0.5, 'sq') * 0.35
    s = filt(sos_lp(520, 2), s)
    s = np.tanh(s * 1.6) / np.tanh(1.6)
    s = filt(sos_lp(1800, 2), s)
    s *= gate
    return np.stack([s, s], axis=1)


def kick(strength=1.0, dur=0.42):
    n = int(dur * SR)
    tt = tarr(n)
    f = 46 + 120 * np.exp(-tt / 0.035) + 40 * np.exp(-tt / 0.006)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-tt / (0.17 * strength ** 0.5))
    click = filt(sos_bp(1800, 6000), RNG.standard_normal(n)) * np.exp(-tt / 0.004) * 0.35
    x = np.tanh((body + click) * (1.4 + 0.6 * strength))
    return fade_edges(x, 2.0, 8)


def snare():
    n = int(0.28 * SR)
    tt = tarr(n)
    tone = np.sin(2 * np.pi * np.cumsum(185 + 60 * np.exp(-tt / 0.02)) / SR) * np.exp(-tt / 0.06)
    nz = filt(sos_bp(1400, 8000), RNG.standard_normal(n)) * np.exp(-tt / 0.09)
    x = tone * 0.6 + nz * 1.1
    return fade_edges(x, 2.0, 8)


def hat(open_=False):
    n = int((0.22 if open_ else 0.07) * SR)
    tt = tarr(n)
    nz = filt(sos_hp(7000, 2), RNG.standard_normal(n))
    x = nz * np.exp(-tt / (0.09 if open_ else 0.018))
    return fade_edges(x, 2.0, 5)


def make_drums():
    out = np.zeros((N, 2))
    for i in range(32):
        tb = i * BEAT
        if 4.0 <= tb < 14.75 and tb not in (8.0, 13.0):
            place(out, kick(1.0), tb, 0.95)
        if 4.0 <= tb < 14.75 and i % 2 == 1:
            place(out, snare(), tb, 0.42, pan=0.05)
    for i in range(64):
        th = i * 0.25
        if not music_stop(th) or th + 0.07 > 14.75 and th < 14.75 and False:
            continue
        if 3.75 - 0.07 < th < 3.75:
            continue
        op = (i % 4 == 2) and th >= 4.0 and th < 14.0
        g = (0.20 if i % 2 else 0.13) * (0.75 if th < 4.0 else 1.0)
        x = hat(op)
        if th < 3.75:
            x = x[: max(64, min(len(x), int((3.75 - th) * SR)))]
        if th < 14.75 and th + len(x) / SR > 14.75:
            x = fade_edges(x[: int((14.75 - th) * SR)], 2, 4)
        place(out, x, th, g * (1.4 if op else 1.0), pan=0.25 if i % 2 else -0.15)
    return out


# ------------------------------------------------------------------ 音效
def noise(n):
    return RNG.standard_normal(n)


def sfx_open_whoosh():
    n = int(0.6 * SR)
    tt = tarr(n)
    x = sweep_bp(noise(n), 300, 2600, q=1.1)
    x2 = sweep_bp(noise(n), 2600, 500, q=1.1)
    w = np.clip(tt / 0.15, 0, 1)
    x = x * (1 - w) + x2 * w
    e = np.minimum(1, tt / 0.035) * np.exp(-np.maximum(0, tt - 0.1) / 0.13)
    rum = np.sin(2 * np.pi * np.cumsum(60 + 30 * np.exp(-tt / 0.1)) / SR) * np.exp(-tt / 0.18)
    y = norm(x * e) * 0.9 + rum * 0.5
    return fade_edges(y, 3, 10)


def sfx_road():
    """0–3.75 路面风噪：随速度加强，每个弯道带一阵左右摆动的风声"""
    n = int(3.75 * SR)
    tt = tarr(n)
    br = np.cumsum(noise(n))
    br = filt(sos_hp(40, 2), br)
    br = norm(filt(sos_lp(380, 2), br))
    wind = norm(filt(sos_bp(500, 2400), noise(n)))
    v = 0.25 + 0.75 * np.clip((tt - 0.6) / 2.9, 0, 1) ** 1.8
    # 弯道方向 → 声像（与画面 HEAD 的弯一致：右左右左右）
    curves = [(0.62, 0.52, 1), (1.42, 0.44, -1), (2.12, 0.36, 1), (2.72, 0.30, -1), (3.18, 0.24, 1)]
    pan = np.zeros(n)
    swell = np.zeros(n)
    for c, hw, s in curves:
        x = (tt - c) / hw
        m = np.abs(x) < 1
        pan[m] += s * 0.6 * np.cos(x[m] * np.pi / 2) ** 2
        swell[m] += np.cos(x[m] * np.pi / 2) ** 2
    L = (br * 0.55 + wind * (0.25 + 0.5 * swell)) * v * (1 - pan * 0.5)
    R = (br * 0.55 + wind * (0.25 + 0.5 * swell)) * v * (1 + pan * 0.5)
    y = np.stack([L, R], axis=1)
    y[:, 0] = fade_edges(y[:, 0], 30, 4)
    y[:, 1] = fade_edges(y[:, 1], 30, 4)
    return y


def sfx_inhale():
    """3.45–3.75 太阳吸气：上扬的吸入声，3.75 戛然而止"""
    n = int(0.3 * SR)
    tt = tarr(n)
    nz = sweep_bp(noise(n), 400, 5000, q=2.0)
    tone = sine_sweep(180 * (1200 / 180) ** (tt / 0.3)) * 0.35
    e = (tt / 0.3) ** 2.2
    y = (norm(nz) * 0.8 + tone) * e
    return fade_edges(y, 3, 4)


def sfx_explosion():
    """4.0 太阳炸开：低频冲击 + 玻璃碎裂 + 0.8 秒内收住的像素沙沙声"""
    n = int(1.4 * SR)
    tt = tarr(n)
    boom = np.sin(2 * np.pi * np.cumsum(44 + 70 * np.exp(-tt / 0.08)) / SR) * np.exp(-tt / 0.42)
    thump = filt(sos_lp(260, 2), noise(n)) * np.exp(-tt / 0.07)
    low = np.tanh((boom * 1.0 + norm(thump) * 0.6) * 2.0)
    # 玻璃碎裂：大量高频衰减正弦 + 亮噪声
    glass = np.zeros(n)
    for k in range(70):
        f = RNG.uniform(2400, 10500)
        t0 = RNG.uniform(0, 0.14) ** 1.3
        i0 = int(t0 * SR)
        m = int(RNG.uniform(0.05, 0.4) * SR)
        x = np.arange(m) / SR
        g = np.sin(2 * np.pi * f * x + RNG.random() * 6) * np.exp(-x / RNG.uniform(0.02, 0.12))
        g = fade_edges(g, 1.0, 3)
        glass[i0:i0 + m] += g[: n - i0] * RNG.uniform(0.3, 1.0)
    crack = filt(sos_hp(3000, 2), noise(n)) * np.exp(-tt / 0.03)
    glass = norm(glass) * 0.75 + norm(crack) * 0.5
    # 像素四散的沙沙声：细碎颗粒 + 低通从高往低扫，0.8 s 收住
    m = int(0.8 * SR)
    grains = np.zeros(m)
    nt = 520
    pos = np.sort(RNG.uniform(0, 0.78, nt) ** 1.4 * 0.78 / 0.78 ** 1.4)
    for p in pos:
        i0 = int(p * SR)
        L = int(RNG.uniform(0.002, 0.006) * SR)
        g = noise(L) * np.hanning(L) * RNG.uniform(0.3, 1)
        grains[i0:i0 + L] += g[: m - i0]
    rustle = sweep_lp(grains + noise(m) * 0.15, 12000, 700, block=256)
    re = np.exp(-tarr(m) / 0.22) * np.clip((0.8 - tarr(m)) / 0.12, 0, 1)
    rustle = norm(rustle * re)
    y = np.zeros((n, 2))
    y[:, 0] = low * 1.0 + glass * 0.55
    y[:, 1] = low * 1.0 + glass * 0.55
    y[:m, 0] += rustle * 0.55
    y[:m, 1] += np.roll(rustle, 90) * 0.55
    y[:, 0] = fade_edges(y[:, 0], 2.0, 30)
    y[:, 1] = fade_edges(y[:, 1], 2.0, 30)
    return y


CPL = [4.70 + 0.18 * k + 1.05 for k in range(5)]   # 每个字成形时刻（与画面一致）
S_UNIT = [4.70 + 0.18 * k for k in range(5)]


def sfx_suction():
    """4.4–6.47 越来越密的“嗖嗖”吸入声"""
    t_a, t_b = 4.4, 6.47
    n = int((t_b - t_a + 0.3) * SR)
    out = np.zeros((n, 2))
    t = 0.0
    k = 0
    while True:
        u = t / (t_b - t_a)
        if u >= 1:
            break
        rate = 3.5 + 13 * u ** 1.4
        L = int(RNG.uniform(0.14, 0.24) * SR)
        x = sweep_bp(noise(L), RNG.uniform(500, 900), RNG.uniform(2500, 5200), q=1.6)
        e = np.sin(np.linspace(0, np.pi, L)) ** 2 * np.linspace(0.6, 1.0, L)
        g = norm(x * e) * (0.45 + 0.55 * u)
        pan = -0.8 + 1.6 * u + RNG.uniform(-0.25, 0.25)
        th = (np.clip(pan, -1, 1) + 1) * np.pi / 4
        i0 = int(t * SR)
        out[i0:i0 + L, 0] += g[: n - i0] * np.cos(th)
        out[i0:i0 + L, 1] += g[: n - i0] * np.sin(th)
        t += 1.0 / rate
        k += 1
    out[:, 0] = fade_edges(out[:, 0], 5, 40)
    out[:, 1] = fade_edges(out[:, 1], 5, 40)
    return out


def click(f=4200, dur=0.012, tau=0.0025):
    n = int(dur * SR)
    tt = tarr(n)
    x = np.sin(2 * np.pi * f * tt) * np.exp(-tt / tau) + filt(sos_hp(2500, 1), noise(n)) * np.exp(-tt / (tau * 0.6)) * 0.4
    return fade_edges(x, 0.6, 2)


def sfx_cell_clicks(bus):
    """每个像素格落定时的轻“咔哒”"""
    for k in range(5):
        a0, a1 = S_UNIT[k] + 0.3, CPL[k]
        cnt = 34 if k < 4 else 40
        ts = np.sort(a0 + (a1 - a0) * RNG.beta(2.2, 1.6, cnt))
        for t0 in ts:
            place(bus, click(RNG.uniform(3000, 6000)), t0, RNG.uniform(0.10, 0.2), pan=-0.7 + 0.35 * k + RNG.uniform(-.1, .1))
        # 字成形（亮一下变实心）
        n = int(0.16 * SR)
        tt = tarr(n)
        f = midi(76 + [0, 2, 3, 5, 7][k])
        tink = (np.sin(2 * np.pi * f * 2 * tt) * 0.6 + np.sin(2 * np.pi * f * 5.4 * tt) * 0.25) * np.exp(-tt / 0.045)
        place(bus, fade_edges(tink, 1.5, 6), CPL[k], 0.3, pan=-0.7 + 0.35 * k)


def sfx_pop(f0=180):
    """6.5 挤出厚度“咔”"""
    n = int(0.12 * SR)
    tt = tarr(n)
    snap = filt(sos_bp(1500, 5000), noise(n)) * np.exp(-tt / 0.012)
    thunk = np.sin(2 * np.pi * np.cumsum(f0 * 0.5 + f0 * np.exp(-tt / 0.015)) / SR) * np.exp(-tt / 0.04)
    return fade_edges(norm(snap) * 0.8 + thunk * 0.7, 1.5, 6)


def sfx_blip_ai():
    """7.0 “AI” 变暖黄：两音上行的亮色短音"""
    y = np.zeros(int(0.32 * SR))
    for i, (m, d) in enumerate(((76, 0.07), (83, 0.24))):
        n = int(d * SR)
        tt = tarr(n)
        s = blep_osc(np.full(n, midi(m)), 'sq') * 0.5 + np.sin(2 * np.pi * midi(m + 12) * tt) * 0.3
        s = filt(sos_lp(5000, 2), s) * np.exp(-tt / (0.05 if i == 0 else 0.09))
        i0 = int(i * 0.065 * SR)
        y[i0:i0 + n] += fade_edges(s, 1.5, 5)[: len(y) - i0]
    return y


def sfx_power_on():
    """7.5 光晕从左到右通电：电流嗡声 + 每个字一次通电“滋”"""
    n = int(0.48 * SR)
    tt = tarr(n)
    hum = blep_osc(np.full(n, 100.0), 'saw') * 0.6 + blep_osc(np.full(n, 150.0), 'sq') * 0.25
    hum = filt(sos_bp(150, 2600), hum)
    flick = np.zeros(n)
    for g in range(6):
        on = g * 0.065
        for j, v in enumerate([1, 0.2, 1, 0.35, 1]):
            a = int((on + j * 0.02) * SR)
            b = int((on + (j + 1) * 0.02) * SR)
            flick[a:b] = np.maximum(flick[a:b], v)
        flick[int((on + 0.1) * SR):] = np.maximum(flick[int((on + 0.1) * SR):], 0.55)
    flick = filt(sos_lp(300, 1), flick)
    e = flick * np.clip((0.48 - tt) / 0.15, 0, 1)
    zz = filt(sos_bp(2500, 9000), noise(n)) * flick * 0.25
    y = hum * e + zz
    pan = np.linspace(-0.8, 0.8, n)
    out = np.stack([y * np.cos((pan + 1) * np.pi / 4), y * np.sin((pan + 1) * np.pi / 4)], axis=1) * 1.4
    out[:, 0] = fade_edges(out[:, 0], 3, 20)
    out[:, 1] = fade_edges(out[:, 1], 3, 20)
    return out


def sfx_reverse_swell():
    """7.0–8.0 反向混响的“吸气”，正好在 8.0 断开"""
    n = int(0.25 * SR)
    tt = tarr(n)
    src = np.zeros(n)
    for m in (57, 60, 64, 69, 76):
        src += blep_osc(np.full(n, midi(m)), 'saw')
    src = filt(sos_lp(3000, 2), src) * np.exp(-tt / 0.06) + filt(sos_hp(2000, 2), noise(n)) * np.exp(-tt / 0.03) * 2
    long_ir = make_ir(sec=1.4, tau=0.5, lp_end=4000, seed=9)
    wet = np.stack([signal.fftconvolve(src, long_ir[:, c]) for c in range(2)], axis=1)
    wet = wet[::-1][-int(1.0 * SR):]
    wet = norm(wet)
    e = np.linspace(0, 1, len(wet)) ** 1.8
    wet *= e[:, None]
    wet[:, 0] = fade_edges(wet[:, 0], 5, 3)
    wet[:, 1] = fade_edges(wet[:, 1], 5, 3)
    return wet


def sfx_impact(strength=1.0):
    """重拍冲击：底鼓 + 次低频 + 碎裂噪声"""
    n = int(0.9 * SR)
    tt = tarr(n)
    sub = np.sin(2 * np.pi * np.cumsum(48 + 90 * np.exp(-tt / 0.05)) / SR) * np.exp(-tt / (0.28 * strength))
    crack = filt(sos_bp(900, 7000), noise(n)) * np.exp(-tt / 0.05)
    x = np.tanh((sub * 1.2 + norm(crack) * 0.55) * 2.2)
    return fade_edges(x, 2.0, 30)


def sfx_glitch():
    """8.0–8.4 数字噪声“滋啦”，两下，第二下更弱"""
    n = int(0.42 * SR)
    tt = tarr(n)
    y = np.zeros(n)
    i = 0
    while i < n:
        L = int(RNG.uniform(0.006, 0.022) * SR)
        f = RNG.choice([180, 330, 520, 760, 1200, 2100, 3400, 5200]) * RNG.uniform(0.9, 1.1)
        kind = RNG.integers(0, 3)
        x = np.arange(L) / SR
        if kind == 0:
            seg = np.sign(np.sin(2 * np.pi * f * x))
        elif kind == 1:
            seg = np.repeat(RNG.uniform(-1, 1, L // 24 + 1), 24)[:L]
        else:
            seg = np.round(noise(L) * 3) / 3
        y[i:i + L] = seg[: n - i] * RNG.uniform(0.5, 1.0)
        i += L
    y = filt(sos_hp(150, 2), y)
    e = np.where(tt < 0.2, 1.0 - 0.3 * tt / 0.2, 0.0) + np.where((tt >= 0.24) & (tt < 0.42), 0.5 * (1 - 0.4 * (tt - 0.24) / 0.18), 0)
    e = filt(sos_lp(400, 1), e)
    y = y * e
    st = np.stack([y, np.roll(y, 37) * 0.9], axis=1)
    st[:, 0] = fade_edges(st[:, 0], 2, 5)
    st[:, 1] = fade_edges(st[:, 1], 2, 5)
    return st


def sfx_zip():
    n = int(0.3 * SR)
    tt = tarr(n)
    x = sweep_bp(noise(n), 900, 6000, q=2.5)
    tone = sine_sweep(400 * (1600 / 400) ** (tt / 0.3)) * 0.2
    e = np.sin(np.pi * np.clip(tt / 0.3, 0, 1)) ** 1.5
    return fade_edges((norm(x) * 0.8 + tone) * e, 3, 6)


def sfx_8bit(m):
    """小球落字：8-bit 短音（方波 + 音高下滑）+ 轻轻一声闷响"""
    n = int(0.16 * SR)
    tt = tarr(n)
    f = midi(m) * (1 + 0.5 * np.exp(-tt / 0.012))
    sq = blep_osc(f, 'sq') * np.where(tt < 0.1, 1, np.exp(-(tt - 0.1) / 0.02))
    sq = np.round(sq * 8) / 8                      # 轻微比特化
    thud = np.sin(2 * np.pi * np.cumsum(110 + 120 * np.exp(-tt / 0.01)) / SR) * np.exp(-tt / 0.035)
    return fade_edges(sq * 0.55 + thud * 0.6, 1.5, 6)


def sfx_ding():
    """12.0 句号落定：清脆的“叮”"""
    n = int(0.9 * SR)
    tt = tarr(n)
    fc = 2093.0
    mod = np.sin(2 * np.pi * fc * 3.5 * tt) * 2.2 * np.exp(-tt / 0.08)
    x = np.sin(2 * np.pi * fc * tt + mod) * np.exp(-tt / 0.28)
    x += np.sin(2 * np.pi * fc * 2.76 * tt) * np.exp(-tt / 0.07) * 0.3
    return fade_edges(x, 1.5, 40)


def sfx_logo():
    """声音标识：同一个 3 音动机 C5–E5–A5（短促、尾音自然衰减，不留长高音）"""
    notes = [(72, 0.0, 0.16), (76, 0.12, 0.16), (81, 0.24, 0.55)]
    y = np.zeros(int(0.95 * SR))
    for m, t0, d in notes:
        n = int(d * SR)
        tt = tarr(n)
        f = midi(m)
        s = blep_osc(np.full(n, f), 'sq') * 0.35 + blep_osc(np.full(n, f * 1.003), 'saw') * 0.35
        s = filt(sos_lp(3200, 2), s)
        bell = np.sin(2 * np.pi * f * 2 * tt + 1.5 * np.sin(2 * np.pi * f * 2 * tt) * np.exp(-tt / 0.05)) * 0.35
        e = np.exp(-tt / (0.09 if d < 0.3 else 0.22))
        i0 = int(t0 * SR)
        y[i0:i0 + n] += fade_edges((s + bell) * e, 2, 10)
    return reverb(y, 0.18)


def sfx_shimmer():
    """12.1–12.95 扫光：从左到右的细亮沙声"""
    n = int(0.85 * SR)
    tt = tarr(n)
    x = sweep_bp(noise(n), 2500, 9000, q=2.0)
    e = np.sin(np.pi * tt / 0.85) ** 2
    y = norm(x * e)
    pan = np.linspace(-0.85, 0.85, n)
    out = np.stack([y * np.cos((pan + 1) * np.pi / 4), y * np.sin((pan + 1) * np.pi / 4)], axis=1) * 1.4
    return out


def sfx_riser():
    """12.75–13.0 短促上扬"""
    n = int(0.25 * SR)
    tt = tarr(n)
    f = 220 * (1760 / 220) ** (tt / 0.25) ** 1.5
    s = blep_osc(f, 'saw') * 0.4 + sweep_bp(noise(n), 800, 7000, q=1.5) * 1.2
    e = (tt / 0.25) ** 1.6
    return fade_edges(norm(filt(sos_lp(6000, 2), s)) * e, 3, 3)


def sfx_decode(bus):
    """乱码解码：随机字闪动的细碎“滴答” + 每锁定一个字一声“嗒”"""
    for i in range(12):
        t0 = 13.0 + i * 0.05
        place(bus, click(RNG.uniform(5000, 8000), 0.008, 0.0012), t0 + 0.002, 0.09, pan=RNG.uniform(-.5, .5))
    for n_ in range(11):
        t0 = 13.08 + n_ * 0.048
        n = int(0.05 * SR)
        tt = tarr(n)
        x = (np.sin(2 * np.pi * 1550 * tt) * 0.7 + np.sin(2 * np.pi * 3100 * tt) * 0.25) * np.exp(-tt / 0.008)
        x += filt(sos_bp(1500, 6000), noise(n)) * np.exp(-tt / 0.003) * 0.5
        place(bus, fade_edges(x, 1, 4), t0, 0.32, pan=-0.6 + 0.12 * n_)


def sfx_breath():
    """14.0–14.75 句点轻轻呼吸发光"""
    n = int(0.75 * SR)
    tt = tarr(n)
    x = filt(sos_bp(300, 1800), noise(n))
    tone = np.sin(2 * np.pi * 220 * tt) * 0.25 + np.sin(2 * np.pi * 330 * tt) * 0.12
    e = np.sin(np.pi * tt / 0.75) ** 2
    return fade_edges((norm(x) * 0.5 + tone) * e, 5, 10)


def sfx_crt():
    """15.0 显像管关机：“噗”的电流声（全片最重的一拍）"""
    n = int(1.0 * SR)
    tt = tarr(n)
    thump = np.sin(2 * np.pi * np.cumsum(46 + 110 * np.exp(-tt / 0.045)) / SR) * np.exp(-tt / 0.32)
    puff = filt(sos_lp(700, 2), noise(n)) * np.exp(-tt / 0.06)
    whine = sine_sweep(9000 * (250 / 9000) ** np.clip(tt / 0.16, 0, 1)) * np.exp(-tt / 0.09) * 0.25
    crackle = np.zeros(n)
    for k in range(40):
        i0 = int(RNG.uniform(0, 0.18) ** 1.5 * SR)
        L = int(0.002 * SR)
        crackle[i0:i0 + L] += noise(L) * np.hanning(L) * RNG.uniform(0.4, 1)
    x = np.tanh((thump * 1.3 + norm(puff) * 0.7) * 2.2) + whine + filt(sos_hp(2000, 1), crackle) * 0.8
    return fade_edges(x, 2.0, 40)


def sfx_beep():
    """15.7 熄灭：短促的“哔”"""
    n = int(0.085 * SR)
    tt = tarr(n)
    x = np.sin(2 * np.pi * 1320 * tt) * 0.8 + blep_osc(np.full(n, 1320.0), 'sq') * 0.12
    return fade_edges(x, 3, 12)


def sfx_sparkle():
    n = int(0.3 * SR)
    tt = tarr(n)
    x = np.sin(2 * np.pi * 2637 * tt) * np.exp(-tt / 0.05) + np.sin(2 * np.pi * 3951 * tt) * np.exp(-tt / 0.03) * 0.5
    return fade_edges(x, 1.5, 10)


def final_chord():
    """15.0 最后一拍：低沉的 Am 落定，自然衰减到静音"""
    n = int(1.0 * SR)
    tt = tarr(n)
    y = np.zeros(n)
    for m in (45, 52, 57, 60, 64):
        y += blep_osc(np.full(n, midi(m)), 'saw') * (1.2 if m < 50 else 0.7)
    y = filt(sos_lp(900, 2), y) * np.exp(-tt / 0.33)
    return reverb(fade_edges(y, 3, 60), 0.25)


# ------------------------------------------------------------------ 混音与母带
def biquad_peak(f0, gain_db, q):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / SR
    al = np.sin(w0) / (2 * q)
    b = np.array([1 + al * A, -2 * np.cos(w0), 1 - al * A])
    a = np.array([1 + al / A, -2 * np.cos(w0), 1 - al / A])
    return np.concatenate([b / a[0], a / a[0]])[None, :]


def lufs(x):
    sos = np.array([[1.53512485958697, -2.69169618940638, 1.19839281085285, 1.0, -1.69065929318241, 0.73248077421585],
                    [1.0, -2.0, 1.0, 1.0, -1.99004745483398, 0.99007225036621]])
    y = signal.sosfilt(sos, x, axis=0)
    blk, hop = int(0.4 * SR), int(0.1 * SR)
    ms = []
    for i in range(0, len(y) - blk + 1, hop):
        ms.append(np.mean(y[i:i + blk] ** 2, axis=0).sum())
    ms = np.array(ms)
    l = -0.691 + 10 * np.log10(ms + 1e-20)
    g = ms[l > -70]
    rel = -0.691 + 10 * np.log10(g.mean()) - 10
    g2 = g[(-0.691 + 10 * np.log10(g)) > rel]
    return -0.691 + 10 * np.log10(g2.mean())


def true_peak_db(x):
    up = signal.resample_poly(x, 4, 1, axis=0)
    return 20 * np.log10(np.max(np.abs(up)) + 1e-12)


def tp_limiter(x, ceil_db=-1.35, look_ms=2.0, rel_ms=80):
    """前瞻式真峰值限幅：4 倍过采样检测，平滑增益"""
    ceil = 10 ** (ceil_db / 20)
    up = signal.resample_poly(x, 4, 1, axis=0)
    pk = np.max(np.abs(up), axis=1).reshape(-1, 4).max(axis=1)[: len(x)]
    if len(pk) < len(x):
        pk = np.pad(pk, (0, len(x) - len(pk)))
    need = np.minimum(1.0, ceil / np.maximum(pk, 1e-9))
    la = int(look_ms / 1000 * SR)
    # 向前取最小（前瞻）
    from scipy.ndimage import minimum_filter1d
    g = minimum_filter1d(need, size=2 * la + 1, mode='nearest')
    # 释放平滑（单极点，只在增益回升时平滑）
    a = np.exp(-1 / (rel_ms / 1000 * SR))
    out = np.empty_like(g)
    cur = 1.0
    for i in range(len(g)):
        v = g[i]
        cur = v if v < cur else a * cur + (1 - a) * v
        out[i] = cur
    out = minimum_filter1d(out, size=la + 1, mode='nearest')
    sm = signal.lfilter([1 / (la + 1)] * (la + 1), [1], out)
    sm = np.minimum(sm, out)
    return x * sm[:, None]


def silence_mask():
    tt = tarr(N)
    m = np.ones(N)
    r = int(0.004 * SR)
    for a, b in SILENCES:
        ia, ib = int(round(a * SR)), int(round(b * SR))
        m[ia:ib] = 0
        m[ia - r:ia] *= np.cos(np.linspace(0, np.pi / 2, r)) ** 2
    # 结尾平滑衰减到静音
    fe = (tt >= 15.8)
    m[fe] *= np.cos(np.clip((tt[fe] - 15.8) / 0.2, 0, 1) * np.pi / 2) ** 2
    m[-1] = 0
    return m


def build(stems=False):
    pad = make_pad()
    arp = make_arp()
    bass = make_bass()
    drums = make_drums()
    music = pad * 0.12 + arp * 0.085 + bass * 0.24 + drums * 0.34
    music = reverb(music, 0.06)
    # 15.0 最后一拍的和弦
    place(music, final_chord(), 15.0, 0.22)

    sfx = np.zeros((N, 2))
    place(sfx, sfx_open_whoosh(), 0.0, 0.55)
    place(sfx, sfx_road(), 0.0, 0.16)
    place(sfx, sfx_inhale(), 3.45, 0.55)
    place(sfx, sfx_explosion(), 4.0, 0.95)
    place(sfx, kick(1.6), 4.0, 0.7)
    place(sfx, sfx_suction(), 4.4, 1.7)
    sfx_cell_clicks(sfx)
    place(sfx, sfx_pop(170), 6.5, 1.1, pan=-0.1)
    place(sfx, sfx_pop(230), 6.5 + 4 / 60, 1.0, pan=0.1)
    place(sfx, sfx_blip_ai(), 7.0, 0.95, pan=0.45)
    place(sfx, sfx_power_on(), 7.5, 1.2)
    place(sfx, sfx_reverse_swell(), 7.0, 0.8)
    place(sfx, sfx_impact(1.0), 8.0, 0.95)
    place(sfx, kick(1.6), 8.0, 0.6)
    place(sfx, sfx_glitch(), 8.0, 0.36)
    place(sfx, sfx_zip(), 8.5, 0.85, pan=-0.3)
    for k, m in enumerate([69, 72, 74, 76, 79, 81]):
        place(sfx, sfx_8bit(m), 9.0 + 0.5 * k, 0.8, pan=-0.6 + 0.24 * k)
    place(sfx, sfx_ding(), 12.0, 0.42, pan=0.5)
    place(sfx, sfx_logo(), 12.0, 0.75)
    place(sfx, sfx_shimmer(), 12.1, 1.05)
    place(sfx, sfx_riser(), 12.75, 1.5)
    place(sfx, sfx_impact(0.8), 13.0, 0.75)
    place(sfx, kick(1.3), 13.0, 0.55)
    sfx_decode(sfx)
    place(sfx, sfx_logo(), 13.7, 0.75)
    place(sfx, sfx_breath(), 14.0, 1.2, pan=0.45)
    place(sfx, sfx_crt(), 15.0, 1.0)
    place(sfx, sfx_sparkle(), 15.5, 0.16)
    place(sfx, sfx_beep(), 15.7, 0.32)

    # 音效响起时音乐让一让（侧链闪避）
    env = np.max(np.abs(sfx), axis=1)
    att, rel = np.exp(-1 / (0.003 * SR)), np.exp(-1 / (0.14 * SR))
    from scipy.signal import lfilter
    e = lfilter([1 - rel], [1, -rel], env)
    e = np.maximum(e, lfilter([1 - att], [1, -att], env))
    duck = 1 / (1 + 3.2 * np.clip(e, 0, 1.5))
    music_d = music * duck[:, None]

    mix = music_d + sfx
    # 母带：40 Hz 以下衰减、4–8 kHz 补亮
    mix = filt(sos_hp(40, 4), mix)
    mix = signal.sosfilt(biquad_peak(5600, 2.8, 0.7), mix, axis=0)
    mask = silence_mask()
    mix *= mask[:, None]
    for _ in range(4):
        L = lufs(mix)
        mix *= 10 ** ((-14.0 - L) / 20)
        mix = tp_limiter(mix, -1.35)
        mix *= mask[:, None]
    L = lufs(mix)
    mix *= 10 ** ((-14.0 - L) / 20)
    tp = true_peak_db(mix)
    if tp > -1.25:
        mix *= 10 ** ((-1.25 - tp) / 20)
    mix *= mask[:, None]
    if stems:
        return mix, music_d, sfx, mask
    return mix


def write_wav(path, x):
    from scipy.io import wavfile
    y = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
    assert y.shape == (N, 2)
    wavfile.write(path, SR, y)


if __name__ == '__main__':
    mix = build()
    write_wav(OUT, mix)
    print(f'wrote {OUT}: {len(mix) / SR:.3f}s  LUFS={lufs(mix):.2f}  TP={true_peak_db(mix):.2f} dBTP')
