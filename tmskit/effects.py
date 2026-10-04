"""时频处理：混响、延迟、合唱、EQ、侧链、限幅、声场。全部为纯数组函数，无状态。"""

from __future__ import annotations

import numpy as np

from .voices import db_to_gain


# ---------------------------------------------------------------- 频域滤波


def _response(f: np.ndarray, hp: float | None, lp: float | None, order: int) -> np.ndarray:
    g = np.ones_like(f)
    if hp:
        g = g * (f**order) / (f**order + hp**order)
    if lp:
        g = g * (lp**order) / (f**order + lp**order)
    return g


def fft_filter(x: np.ndarray, sr: int, hp: float | None = None, lp: float | None = None, order: int = 2) -> np.ndarray:
    """单声道/多声道（最后一维为通道）频域滤波。"""
    n = x.shape[0]
    spec = np.fft.rfft(x, axis=0)
    g = _response(np.fft.rfftfreq(n, 1.0 / sr), hp, lp, order)
    if x.ndim == 2:
        g = g[:, None]
    return np.fft.irfft(spec * g, n=n, axis=0)


def peaking_eq(x: np.ndarray, sr: int, freq: float, gain_db: float, q: float = 1.0) -> np.ndarray:
    """单峰 bell EQ（频域实现）。"""
    if abs(gain_db) < 1e-3:
        return x
    n = x.shape[0]
    spec = np.fft.rfft(x, axis=0)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    bw = freq / max(q, 0.1)
    shape = 1.0 / (1.0 + ((f - freq) / bw) ** 2)
    g = 1.0 + (db_to_gain(gain_db) - 1.0) * shape
    if x.ndim == 2:
        g = g[:, None]
    return np.fft.irfft(spec * g, n=n, axis=0)


# ---------------------------------------------------------------- 卷积混响


def make_impulse_response(
    sr: int,
    decay: float = 2.6,
    predelay: float = 0.030,
    brightness: float = 0.55,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """合成混响 IR：分频段不同衰减时间 + 早反射。返回 (n, 2)。"""
    rng = rng or np.random.default_rng(7)
    n = int((predelay + decay + 0.3) * sr)
    t = np.arange(n) / sr - predelay
    t = np.clip(t, 0.0, None)
    bands = ((80.0, 300.0, 1.15), (300.0, 1200.0, 1.0), (1200.0, 4500.0, 0.62), (4500.0, 16000.0, 0.32))
    out = np.zeros((n, 2))
    for ch in range(2):
        tail = np.zeros(n)
        for lo, hi, rel in bands:
            noise = rng.standard_normal(n)
            band = fft_filter(noise, sr, hp=lo, lp=hi)
            tau = decay * rel
            tail += band * np.exp(-t / tau) * (1.0 + 0.15 * brightness)
        # 早反射：一串稀疏脉冲，给空间尺度感
        for _ in range(9):
            pos = int(rng.uniform(0.004, 0.055) * sr)
            if pos < n:
                tail[pos] += rng.uniform(-0.6, 0.6) * 0.8
        fade = np.clip(t / 0.002, 0.0, 1.0)
        out[:, ch] = tail * fade
    out /= max(float(np.max(np.abs(out))), 1e-9)
    # 两声道做部分相关：完全独立的 IR 会让整曲声道相关度趋近 0，单声道下掉内容
    mid = out.mean(axis=1, keepdims=True)
    out = mid + (out - mid) * 0.55
    return out


def fft_convolve(x: np.ndarray, h: np.ndarray, block: int = 1 << 19) -> np.ndarray:
    """重叠相加 FFT 卷积；x 单声道，h 为 (m,) 或 (m,2)，返回同通道数。"""
    n_h = h.shape[0]
    n_out = x.shape[0] + n_h - 1
    n_fft = 1 << int(np.ceil(np.log2(block + n_h - 1)))
    step = n_fft - n_h + 1
    if h.ndim == 1:
        y = np.zeros(n_out)
        H = np.fft.rfft(h, n=n_fft)
        for start in range(0, x.shape[0], step):
            seg = x[start : start + step]
            part = np.fft.irfft(np.fft.rfft(seg, n=n_fft) * H, n=n_fft)
            end = min(start + n_fft, n_out)
            y[start:end] += part[: end - start]
        return y
    y = np.zeros((n_out, h.shape[1]))
    H = np.fft.rfft(h, n=n_fft, axis=0)
    for start in range(0, x.shape[0], step):
        seg = x[start : start + step]
        part = np.fft.irfft(np.fft.rfft(seg, n=n_fft)[:, None] * H, n=n_fft, axis=0)
        end = min(start + n_fft, n_out)
        y[start:end] += part[: end - start]
    return y


# ---------------------------------------------------------------- 延迟 / 合唱


def feedback_delay(x: np.ndarray, sr: int, time_s: float, feedback: float, wet: float, pingpong: bool = True) -> np.ndarray:
    """反馈延迟（展开成 FIR 形式，完全矢量化）。输入单声道，返回立体声。"""
    d = max(int(time_s * sr), 1)
    taps = int(np.ceil(np.log(1e-3) / np.log(max(feedback, 1e-3)))) if feedback > 0 else 0
    left = x.copy()
    right = x.copy()
    for k in range(1, taps + 1):
        g = feedback**k
        shift = k * d
        if shift >= x.shape[0]:
            break
        if pingpong:
            if k % 2 == 1:
                left[shift:] += g * x[:-shift]
            else:
                right[shift:] += g * x[:-shift]
        else:
            left[shift:] += g * x[:-shift]
            right[shift:] += g * x[:-shift]
    return np.stack([np.clip(left, -4, 4) * wet, np.clip(right, -4, 4) * wet], axis=1)


def chorus(x: np.ndarray, sr: int, rate: float = 0.35, depth_ms: float = 9.0, mix: float = 0.35, voices: int = 2) -> np.ndarray:
    """轻度合唱/加宽：返回立体声。"""
    n = x.shape[0]
    t = np.arange(n) / sr
    idx = np.arange(n, dtype=np.float64)
    out = np.zeros((n, 2))
    for ch in range(2):
        for v in range(voices):
            lfo = np.sin(2.0 * np.pi * rate * (1.0 + 0.13 * v) * t + ch * 1.7 + v * 2.1)
            delay = (depth_ms * (v + 1) / voices) * 1e-3 * sr
            pos = idx - delay * (0.5 + 0.5 * lfo)
            out[:, ch] += np.interp(pos, idx, x, left=0.0, right=0.0)
    out /= max(voices, 1)
    return (1.0 - mix) * np.stack([x, x], axis=1) + mix * out


# ---------------------------------------------------------------- 动态处理


def _block_peak(x: np.ndarray, sr: int, ms: float = 1.0) -> tuple[np.ndarray, int]:
    mag = np.abs(x).max(axis=1) if x.ndim == 2 else np.abs(x)
    b = max(int(sr * ms / 1000.0), 1)
    usable = (mag.shape[0] // b) * b
    if usable == 0:
        return mag.reshape(-1), 1
    peak = mag[:usable].reshape(-1, b).max(axis=1)
    if usable < mag.shape[0]:
        peak = np.append(peak, mag[usable:].max())
    return peak, b


def _dilate(m: np.ndarray, width: int) -> np.ndarray:
    """长度为 width 的滑动最大值（对数次 roll+max）。"""
    out = m.copy()
    k = 1
    while k < width:
        out = np.maximum(out, np.roll(out, k))
        out = np.maximum(out, np.roll(out, -k))
        k *= 2
    return out


def _smooth(m: np.ndarray, width: int) -> np.ndarray:
    width = max(int(width), 1)
    kernel = np.ones(width) / width
    return np.convolve(m, kernel, mode="same")


def limiter(x: np.ndarray, sr: int, ceiling_db: float = -1.0, lookahead_ms: float = 6.0, release_ms: float = 90.0) -> np.ndarray:
    """前视限幅：包络取块峰值 → 前视膨胀 → 平滑 → 增益包络。"""
    ceiling = db_to_gain(ceiling_db)
    peak, block = _block_peak(x, sr, 1.0)
    look_blocks = max(int(lookahead_ms / 1.0), 1)
    rel_blocks = max(int(release_ms / 1.0), 1)
    env = _dilate(peak, look_blocks)
    env = _smooth(env, rel_blocks)
    gain_blocks = np.minimum(1.0, ceiling / np.maximum(env, 1e-9))
    idx = np.arange(x.shape[0]) / block
    gain = np.interp(idx, np.arange(gain_blocks.shape[0]), gain_blocks, left=gain_blocks[0], right=gain_blocks[-1])
    if x.ndim == 2:
        gain = gain[:, None]
    return x * gain


def soft_clip(x: np.ndarray, drive: float = 1.2) -> np.ndarray:
    return np.tanh(x * drive) / np.tanh(drive)


def compress(x: np.ndarray, sr: int, threshold_db: float = -14.0, ratio: float = 2.0,
             attack_ms: float = 20.0, release_ms: float = 160.0, makeup_db: float = 0.0) -> np.ndarray:
    """总线胶水压缩：块峰值包络 → 增益衰减（软拐点）。"""
    peak, block = _block_peak(x, sr, 1.0)
    env = _smooth(_dilate(peak, max(int(attack_ms), 1)), max(int(release_ms), 1))
    env_db = 20.0 * np.log10(np.maximum(env, 1e-9))
    over = np.maximum(env_db - threshold_db, 0.0)
    reduction = over * (1.0 - 1.0 / max(ratio, 1.0))
    gain_blocks = db_to_gain(-reduction + makeup_db)
    idx = np.arange(x.shape[0]) / block
    gain = np.interp(idx, np.arange(gain_blocks.shape[0]), gain_blocks, left=gain_blocks[0], right=gain_blocks[-1])
    if x.ndim == 2:
        gain = gain[:, None]
    return x * gain


def rms_db(x: np.ndarray) -> float:
    r = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20.0 * np.log10(max(r, 1e-12))


def true_peak_db(x: np.ndarray, sr: int) -> float:
    """4 倍过采样后的峰值估计，接近真峰值。"""
    if x.size == 0:
        return -120.0
    chans = [x] if x.ndim == 1 else [x[:, c] for c in range(x.shape[1])]
    best = 0.0
    for ch in chans:
        up = np.interp(np.arange(0, ch.shape[0] - 1, 0.25), np.arange(ch.shape[0]), ch)
        best = max(best, float(np.max(np.abs(up))))
    return 20.0 * np.log10(max(best, 1e-12))


def normalize_rms(x: np.ndarray, target_db: float) -> np.ndarray:
    cur = rms_db(x)
    return x * db_to_gain(target_db - cur)


# ---------------------------------------------------------------- 声场


def mono_bass(x: np.ndarray, sr: int, fc: float = 120.0) -> np.ndarray:
    """120 Hz 以下强制单声道，避免低频相位抵消。"""
    low = fft_filter(x, sr, lp=fc, order=4)
    mid = low.mean(axis=1, keepdims=True)
    return x - low + np.repeat(mid, 2, axis=1)


def sidechain_duck(x: np.ndarray, key: np.ndarray, sr: int, depth_db: float = -4.0, attack_ms: float = 8.0, release_ms: float = 110.0) -> np.ndarray:
    """按 key 轨的包络衰减 x（kick 触发 sub/pad 让位）。"""
    if key.size == 0 or x.size == 0:
        return x
    n = min(x.shape[0], key.shape[0])
    peak, block = _block_peak(key[:n], sr, 1.0)
    peak = peak / max(float(np.max(peak)), 1e-9)
    env = _dilate(peak, max(int(attack_ms), 1))
    env = _smooth(env, max(int(release_ms), 1))
    depth = 1.0 - db_to_gain(depth_db)
    gain_blocks = 1.0 - depth * np.clip(env, 0.0, 1.0)
    idx = np.arange(x.shape[0]) / block
    gain = np.interp(idx, np.arange(gain_blocks.shape[0]), gain_blocks, left=gain_blocks[0], right=gain_blocks[-1])
    if x.ndim == 2:
        gain = gain[:, None]
    return x * gain


def stereo_width(x: np.ndarray, width: float = 1.0) -> np.ndarray:
    """width>1 加宽，<1 收窄（中侧处理）。"""
    mid = x.mean(axis=1, keepdims=True)
    side = (x - mid) * width
    return mid + side
