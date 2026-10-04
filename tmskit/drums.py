"""无音高打击乐：全部由正弦/噪声 + 包络 + 频域滤波合成，无采样。

每个鼓件自带绝对电平（`_level` 的 dB 目标），相对配比参考：
kick −12、snare −14、clap −16、tom −18、rim −24、ride −22、crash −18、
hat −24、ohat −23、shaker −30、riser/downlift −20（dBFS 峰值 @ vel=1）。
"kick 最响、hat 最轻"这一相对关系比绝对数值重要，母带会用 RMS 归一化统一抬到目标响度。
"""

from __future__ import annotations

import numpy as np

from .voices import PatchSpec, db_to_gain


def _level(x: np.ndarray, target_db: float) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1e-12:
        return x
    return x * (db_to_gain(target_db) / peak)


def _fft_filter(x: np.ndarray, sr: int, hp: float | None = None, lp: float | None = None, order: int = 2) -> np.ndarray:
    n = x.size
    spec = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1.0 / sr)
    g = np.ones_like(f)
    if hp:
        g = g * (f**order) / (f**order + hp**order)
    if lp:
        g = g * (lp**order) / (f**order + lp**order)
    return np.fft.irfft(spec * g, n=n)


def _noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


# ---------------------------------------------------------------- 鼓件


def _kick(vel, sr, rng, dur=None):
    n = int(0.9 * sr)
    t = np.arange(n) / sr
    f = 45.0 + 95.0 * np.exp(-t / 0.030)
    phase = 2.0 * np.pi * np.cumsum(f) / sr
    body = np.sin(phase) * np.exp(-t / 0.155)
    click = _fft_filter(_noise(n, rng), sr, hp=1400.0) * np.exp(-t / 0.0035) * 0.45
    x = np.tanh((body + click) * 1.35)
    x *= np.exp(-t / 0.34)
    return _level(x, -12.0)


def _snare(vel, sr, rng, dur=None):
    n = int(0.42 * sr)
    t = np.arange(n) / sr
    tone = np.sin(2.0 * np.pi * 196.0 * t) * np.exp(-t / 0.048) * 0.7
    tone += np.sin(2.0 * np.pi * 286.0 * t) * np.exp(-t / 0.032) * 0.35
    nz = _fft_filter(_noise(n, rng), sr, hp=1250.0, lp=5600.0) * np.exp(-t / 0.115)
    return _level(tone + nz * 0.95, -13.0)


def _hat(vel, sr, rng, dur=None, tau=0.028, level=-21.0, length=0.14):
    n = int(length * sr)
    t = np.arange(n) / sr
    nz = _fft_filter(_noise(n, rng), sr, hp=7600.0, lp=15000.0)
    metal = np.sin(2.0 * np.pi * 9200.0 * t) * 0.25 + np.sin(2.0 * np.pi * 12300.0 * t) * 0.18
    env = np.exp(-t / tau)
    env *= np.clip(t / 0.0006, 0.0, 1.0)
    return _level((nz + metal) * env, level)


def _ohat(vel, sr, rng, dur=None):
    return _hat(vel, sr, rng, tau=0.30, level=-20.0, length=0.85)


def _clap(vel, sr, rng, dur=None):
    n = int(0.5 * sr)
    t = np.arange(n) / sr
    nz = _fft_filter(_noise(n, rng), sr, hp=950.0, lp=2600.0, order=3)
    env = np.zeros(n)
    for off in (0.0, 0.009, 0.018, 0.027):
        env += np.exp(-np.clip(t - off, 0, None) / 0.011) * (t >= off)
    tail = np.exp(-np.clip(t - 0.027, 0, None) / 0.19) * (t >= 0.027)
    return _level(nz * (0.55 * env + 0.85 * tail), -15.0)


def _ride(vel, sr, rng, dur=None):
    n = int(1.9 * sr)
    t = np.arange(n) / sr
    base = 340.0
    x = np.zeros(n)
    for k, ratio in enumerate((1.0, 1.47, 1.93, 2.51, 3.17, 3.83)):
        tau = 1.4 / (1.0 + 0.35 * k)
        x += np.sin(2.0 * np.pi * base * ratio * t + k * 1.1) * np.exp(-t / tau) / (1.0 + 0.6 * k)
    nz = _fft_filter(_noise(n, rng), sr, hp=5200.0, lp=13000.0) * np.exp(-t / 0.75) * 0.55
    env = np.clip(t / 0.0015, 0.0, 1.0)
    return _level((x * 0.8 + nz) * env, -18.0)


def _crash(vel, sr, rng, dur=None):
    n = int(2.6 * sr)
    t = np.arange(n) / sr
    nz = _fft_filter(_noise(n, rng), sr, hp=2800.0, lp=16000.0) * np.exp(-t / 1.05)
    metal = np.zeros(n)
    for k, ratio in enumerate((1.0, 1.41, 1.86, 2.37, 3.02, 3.71, 4.55)):
        metal += np.sin(2.0 * np.pi * 520.0 * ratio * t + k * 2.1) * np.exp(-t / (1.5 / (1 + 0.3 * k))) / (1 + k)
    env = np.clip(t / 0.002, 0.0, 1.0)
    shimmer = 1.0 + 0.12 * np.sin(2.0 * np.pi * 6.5 * t)
    return _level((nz * 0.9 + metal * 0.7) * env * shimmer, -15.0)


def _rim(vel, sr, rng, dur=None):
    n = int(0.13 * sr)
    t = np.arange(n) / sr
    tone = np.sin(2.0 * np.pi * 430.0 * t) * np.exp(-t / 0.018)
    nz = _fft_filter(_noise(n, rng), sr, hp=2200.0) * np.exp(-t / 0.010)
    return _level(tone * 0.8 + nz * 0.6, -22.0)


def _tom(vel, sr, rng, dur=None):
    n = int(0.6 * sr)
    t = np.arange(n) / sr
    f = 132.0 + 88.0 * np.exp(-t / 0.06)
    phase = 2.0 * np.pi * np.cumsum(f) / sr
    x = np.sin(phase) * np.exp(-t / 0.24)
    nz = _fft_filter(_noise(n, rng), sr, hp=900.0, lp=3800.0) * np.exp(-t / 0.03) * 0.25
    return _level(x + nz, -17.0)


def _shaker(vel, sr, rng, dur=None):
    n = int(0.22 * sr)
    t = np.arange(n) / sr
    nz = _fft_filter(_noise(n, rng), sr, hp=5200.0, lp=14000.0)
    env = np.clip(t / 0.012, 0.0, 1.0) * np.exp(-t / 0.045)
    return _level(nz * env, -26.0)


def _riser(vel, sr, rng, dur=None):
    """上行噪声：一组错开的带通噪声层，频段与包络随时间上移。"""
    span = max(float(dur or 2.0), 0.4)
    n = int((span + 0.25) * sr)
    t = np.arange(n) / sr
    bands = (400.0, 700.0, 1100.0, 1800.0, 2900.0, 4600.0, 7200.0)
    x = np.zeros(n)
    for i, fc in enumerate(bands):
        centre = span * (i + 0.5) / len(bands)
        width = span * 0.55 / len(bands) + 0.08
        layer = _fft_filter(_noise(n, rng), sr, hp=fc * 0.7, lp=fc * 2.2)
        x += layer * np.exp(-((t - centre) / width) ** 2)
    ramp = np.clip(t / span, 0.0, 1.0) ** 1.6
    tone = np.sin(2.0 * np.pi * 220.0 * (1.0 + 4.0 * ramp) * t) * ramp * 0.06
    return _level((x * 0.8 + tone) * ramp, -20.0)


def _downlift(vel, sr, rng, dur=None):
    """下行 whoosh：宽噪声从高频落向低频，尾端快速收掉。"""
    span = max(float(dur or 1.5), 0.4)
    n = int((span + 0.4) * sr)
    t = np.arange(n) / sr
    bands = (7200.0, 5000.0, 3300.0, 2100.0, 1300.0, 800.0, 450.0)
    x = np.zeros(n)
    for i, fc in enumerate(bands):
        centre = span * (i + 0.5) / len(bands)
        width = span * 0.55 / len(bands) + 0.08
        layer = _fft_filter(_noise(n, rng), sr, hp=fc * 0.7, lp=fc * 2.2)
        x += layer * np.exp(-((t - centre) / width) ** 2)
    env = np.clip(t / (span * 0.35), 0.0, 1.0) * np.exp(-np.clip(t - span * 0.5, 0, None) / (span * 0.4))
    return _level(x * env, -20.0)


_DISPATCH = {
    "kick": _kick,
    "snare": _snare,
    "hat": _hat,
    "ohat": _ohat,
    "clap": _clap,
    "ride": _ride,
    "crash": _crash,
    "rim": _rim,
    "tom": _tom,
    "shaker": _shaker,
    "riser": _riser,
    "downlift": _downlift,
}


def gen_drum(token: str, dur: float, vel: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    fn = _DISPATCH.get(token)
    if fn is None:
        raise ValueError(f"未知鼓件: {token!r}")
    return fn(vel, sr, rng, dur)


DRUMKIT = PatchSpec(
    name="drumkit",
    kind="drum",
    low=0,
    high=0,
    nominal_db=0.0,
    desc="鼓组：kick/snare/hat/ohat/clap/ride/crash/rim/tom/shaker/riser/downlift",
    gen=gen_drum,
)
