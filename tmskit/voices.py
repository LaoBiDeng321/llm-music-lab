"""有音高音色：加法合成 + FM。每个 gen 返回峰值已归一化到 nominal_db 的单声道 float64 波形。

约定
----
- gen(freqs, dur, vel, sr, rng) -> np.ndarray
  freqs: 本次触发同时发声的频率列表（单音为长度 1）
  dur:   门限时长（秒），返回值可以更长（自带释放尾巴）
- gen 内部不做力度与声道处理；vel 只用于「音色本身对力度的响应」（如滤波亮度）。
- 所有随机必须来自传入的 rng，保证同 seed 逐字节可复现。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from . import theory

SR = 44100

# ---------------------------------------------------------------- 基础工具


def db_to_gain(db):
    """dB → 线性增益；标量或数组皆可。"""
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 20.0)


def adsr(n: int, sr: int, attack: float, decay: float, sustain: float, release: float, hold: float) -> np.ndarray:
    """门限 hold 秒、总长 hold+release 秒的 ADSR 包络，长度 n。"""
    a = max(attack, 1e-4)
    d = max(decay, 1e-4)
    r = max(release, 1e-3)
    knee = max(hold, a + d)
    xs = np.array([0.0, a, a + d, knee, knee + r])
    ys = np.array([0.0, 1.0, max(sustain, 0.0), max(sustain, 0.0), 0.0])
    t = np.arange(n) / sr
    return np.interp(t, xs, ys)


def exp_decay(n: int, sr: int, attack: float, tau: float, floor: float = 0.0) -> np.ndarray:
    t = np.arange(n) / sr
    env = np.exp(-t / max(tau, 1e-3))
    if attack > 0:
        env *= np.clip(t / attack, 0.0, 1.0)
    return floor + (1.0 - floor) * env


def gate_fade(n: int, sr: int, hold: float, release: float) -> np.ndarray:
    """门限后线性收尾：t < hold 时为 1，之后在 release 秒内降到 0。

    拨弦类音色用它替代 ADSR，避免「衰减到很小后还拖一段几乎无声的尾巴」
    （那会白白拉长渲染时间，也让有效 RMS 归一化失真）。
    """
    t = np.arange(n) / sr
    return np.clip((hold + release - t) / max(release, 1e-3), 0.0, 1.0)


def one_pole_smooth(x: np.ndarray, tau_s: float, sr: int) -> np.ndarray:
    """对慢变控制信号做单极点平滑（块级近似，控制信号本身点数很少时使用）。"""
    alpha = float(np.exp(-1.0 / max(tau_s, 1e-6) / sr))
    out = np.empty_like(x)
    acc = x[0]
    for i, v in enumerate(x):
        acc = alpha * acc + (1.0 - alpha) * v
        out[i] = acc
    return out


def _partials_saw(max_k: int, detunes_cents: tuple[float, ...] = (0.0,), tilt: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """返回 (ratio, amp) 数组：tilt=1 锯齿，tilt=2 三角（只取奇次）。"""
    ratios: list[float] = []
    amps: list[float] = []
    ks = range(1, max_k + 1) if tilt < 1.5 else range(1, max_k + 1, 2)
    for cents in detunes_cents:
        mul = 2.0 ** (cents / 1200.0)
        for k in ks:
            ratios.append(k * mul)
            amps.append(1.0 / (k ** tilt))
    return np.array(ratios), np.array(amps)


def _norm_peak(x: np.ndarray, target_db: float) -> np.ndarray:
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if peak < 1e-12:
        return x
    return x * (db_to_gain(target_db) / peak)


def _effective_rms(x: np.ndarray, sr: int, window_ms: float = 40.0) -> float:
    """滑动 RMS 的高分位值：代表「持续段的有效电平」，不被起音/尾巴拉低。"""
    if x.size == 0:
        return 0.0
    w = max(int(sr * window_ms / 1000.0), 8)
    if x.shape[0] < w:
        return float(np.sqrt(np.mean(np.square(x))))
    csum = np.cumsum(np.square(x))
    csum = np.concatenate([[0.0], csum])
    seg = csum[w:] - csum[:-w]
    rms = np.sqrt(seg / w)
    return float(np.percentile(rms, 88))


def _norm_rms(x: np.ndarray, target_db: float, sr: int) -> np.ndarray:
    """按有效 RMS 归一化（适合有稳定延音的铺底类音色），峰值 0.9 兜底。"""
    ref = _effective_rms(x, sr)
    peak = float(np.max(np.abs(x))) if x.size else 0.0
    if ref < 1e-12 or peak < 1e-12:
        return x
    gain = float(db_to_gain(target_db)) / ref
    gain = min(gain, 0.9 / peak)
    return x * gain


def additive_layer(
    freqs: list[float],
    n: int,
    sr: int,
    partial_ratios: np.ndarray,
    partial_amps: np.ndarray,
    amp_env: np.ndarray,
    cutoff_env: np.ndarray,
    rng: np.random.Generator,
    vib_depth: float = 0.0,
    vib_rate: float = 5.0,
    phase_mod: np.ndarray | None = None,
) -> np.ndarray:
    """把一组基频用同一谱型叠加。cutoff_env 为随时间变化的低通截止（两极点，-12dB/oct）。"""
    out = np.zeros(n)
    if not freqs:
        return out
    fc2 = np.square(cutoff_env)
    chunk = 16384
    for f0 in freqs:
        mask = partial_ratios * f0 < sr * 0.45
        ratios = partial_ratios[mask]
        amps = partial_amps[mask]
        if ratios.size == 0:
            continue
        fk = ratios * f0
        phi = rng.uniform(0.0, 2.0 * np.pi, size=ratios.size)
        spread = np.sqrt(fk / max(f0, 1e-6))  # 高次分音揉弦幅度更大
        for s in range(0, n, chunk):
            e = min(n, s + chunk)
            t = np.arange(s, e) / sr
            phase = 2.0 * np.pi * fk[:, None] * t[None, :] + phi[:, None]
            if vib_depth > 0.0:
                phase += (vib_depth * spread)[:, None] * np.sin(2.0 * np.pi * vib_rate * t)[None, :]
            if phase_mod is not None:
                phase += phase_mod[s:e][None, :]
            gain = (fc2[s:e] / (fc2[s:e] + np.square(fk)[:, None])) * amps[:, None]
            gain *= amp_env[s:e][None, :]
            out[s:e] += np.einsum("km,km->m", np.sin(phase), gain, optimize=True)
    return out


def fm_voice(
    freq: float,
    n: int,
    sr: int,
    ratio: float,
    index_env: np.ndarray,
    amp_env: np.ndarray,
    rng: np.random.Generator,
    carrier_ratio: float = 1.0,
    detune_cents: float = 0.0,
) -> np.ndarray:
    """经典 2 算子 FM。index_env 为调制指数随时间变化曲线。"""
    t = np.arange(n) / sr
    f = freq * 2.0 ** (detune_cents / 1200.0)
    phase = rng.uniform(0.0, 2.0 * np.pi)
    mod = index_env * np.sin(2.0 * np.pi * f * ratio * t + phase)
    return amp_env * np.sin(2.0 * np.pi * f * carrier_ratio * t + mod)


def noise(n: int, rng: np.random.Generator, color: str = "white") -> np.ndarray:
    """白噪；`pink` 用频域整形实现（矢量化，避免逐样本循环）。"""
    x = rng.standard_normal(n)
    if color != "pink":
        return x
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, d=1.0 / SR)
    scale = np.ones_like(freqs)
    scale[1:] = 1.0 / np.sqrt(freqs[1:])
    scale /= np.max(scale[1:]) if scale.size > 1 else 1.0
    return np.fft.irfft(spec * scale, n=n)


# ---------------------------------------------------------------- 包络 / 滤波控制


def _cutoff_curve(n: int, sr: int, start: float, end: float, tau: float) -> np.ndarray:
    t = np.arange(n) / sr
    return end + (start - end) * np.exp(-t / max(tau, 1e-3))


def _hold_len(dur: float, release: float, sr: int, cap: float | None = None) -> int:
    span = dur + release
    if cap is not None:
        span = min(span, cap)
    return max(int(span * sr) + 1, int(0.01 * sr))


# ---------------------------------------------------------------- 音色生成器


def gen_saw_pad(freqs, dur, vel, sr, rng):
    """暖 Pad：3 层失谐锯齿 + 慢起音，低通 1.2k→2.4k。"""
    rel = 2.6
    n = _hold_len(dur, rel, sr)
    ratios, amps = _partials_saw(26, (-9.0, 0.0, 9.0), tilt=1.0)
    amps = amps / 3.0
    env = adsr(n, sr, attack=0.85, decay=1.6, sustain=0.82, release=rel, hold=dur)
    fc = _cutoff_curve(n, sr, 900.0, 2500.0, 1.6)
    x = additive_layer(freqs, n, sr, ratios, amps, env, fc, rng, vib_depth=0.0)
    return x


def gen_glass_pad(freqs, dur, vel, sr, rng):
    """玻璃感 Pad：三角波族 + 慢起音 + 更亮的截止，用于高潮铺底。"""
    rel = 3.0
    n = _hold_len(dur, rel, sr)
    ratios, amps = _partials_saw(20, (-12.0, 0.0, 12.0), tilt=1.7)
    amps = amps / 3.0
    env = adsr(n, sr, attack=1.1, decay=1.8, sustain=0.85, release=rel, hold=dur)
    fc = _cutoff_curve(n, sr, 1400.0, 4200.0, 2.0)
    x = additive_layer(freqs, n, sr, ratios, amps, env, fc, rng, vib_depth=0.004, vib_rate=0.22)
    return x


def gen_choir_pad(freqs, dur, vel, sr, rng):
    """人声感 Pad：共振峰加权（约 600 / 1150 / 2900 Hz）+ 轻揉弦。"""
    rel = 2.8
    n = _hold_len(dur, rel, sr)
    ratios, amps = _partials_saw(24, (-7.0, 0.0, 7.0), tilt=1.0)
    amps = amps / 3.0
    env = adsr(n, sr, attack=1.3, decay=2.0, sustain=0.9, release=rel, hold=dur)
    fc = _cutoff_curve(n, sr, 1800.0, 3200.0, 3.0)
    x = additive_layer(freqs, n, sr, ratios, amps, env, fc, rng, vib_depth=0.006, vib_rate=4.6)
    return x


def gen_epiano(freqs, dur, vel, sr, rng):
    """电钢琴 FM：比 1:1，调制指数 3.4→0.5（400ms），带 tine 泛音瞬态。"""
    rel = 0.9
    n = _hold_len(dur, rel, sr)
    t = np.arange(n) / sr
    bright = 0.55 + 0.45 * vel
    index = (3.4 * bright) * np.exp(-t / 0.34) + 0.35
    env = adsr(n, sr, attack=0.004, decay=1.15, sustain=0.30, release=rel, hold=dur)
    out = np.zeros(n)
    for f in freqs:
        idx = index * min(1.0, 700.0 / max(f, 1.0)) ** 0.5
        out += fm_voice(f, n, sr, ratio=1.0, index_env=idx, amp_env=env, rng=rng)
        out += 0.22 * fm_voice(f, n, sr, ratio=1.0, index_env=idx, amp_env=env, rng=rng, detune_cents=4.0)
        # tine 敲击瞬态：高八度以上、极快衰减的细正弦
        tine = exp_decay(n, sr, 0.0008, 0.075)
        out += 0.14 * np.sin(2.0 * np.pi * f * 9.0 * t) * tine
    return out


def gen_bell(freqs, dur, vel, sr, rng):
    """钟琴：FM 比 1:3.5，长衰减。"""
    rel = 2.2
    n = _hold_len(dur, rel, sr)
    t = np.arange(n) / sr
    index = 2.6 * np.exp(-t / 0.9) + 0.15
    env = exp_decay(n, sr, 0.004, 1.5, floor=0.0)
    env *= adsr(n, sr, attack=0.002, decay=0.5, sustain=0.95, release=rel, hold=dur + 0.6)
    out = np.zeros(n)
    for f in freqs:
        out += fm_voice(f, n, sr, ratio=3.5, index_env=index * min(1.0, 900.0 / max(f, 1.0)), amp_env=env, rng=rng)
        out += 0.25 * np.sin(2.0 * np.pi * f * 2.0 * t) * exp_decay(n, sr, 0.002, 0.6)
    return out


def gen_music_box(freqs, dur, vel, sr, rng):
    """八音盒：高比例 FM，极短起音，清脆。"""
    rel = 1.4
    n = _hold_len(dur, rel, sr)
    t = np.arange(n) / sr
    index = 1.9 * np.exp(-t / 0.35) + 0.1
    env = exp_decay(n, sr, 0.002, 0.85)
    out = np.zeros(n)
    for f in freqs:
        out += fm_voice(f, n, sr, ratio=7.0, index_env=index, amp_env=env, rng=rng)
        out += 0.3 * np.sin(2.0 * np.pi * f * t) * exp_decay(n, sr, 0.002, 0.5)
    return out


def gen_pluck(freqs, dur, vel, sr, rng):
    """拨弦：锯齿族，截止 4.2k→700（150ms），指数衰减 0.32s，最长 1.5s。"""
    rel = 0.30
    n = _hold_len(dur, rel, sr, cap=1.5)
    bright = 0.4 + 0.6 * vel
    span = n / sr
    env = exp_decay(n, sr, 0.004, 0.32) * gate_fade(n, sr, max(span - rel, 0.05), rel)
    fc = _cutoff_curve(n, sr, 4200.0 * bright, 700.0, 0.11)
    ratios, amps = _partials_saw(24, (-4.0, 4.0), tilt=1.0)
    amps = amps / 2.0
    return additive_layer(freqs, n, sr, ratios, amps, env, fc, rng)


def gen_nylon(freqs, dur, vel, sr, rng):
    """尼龙弦拨奏：更柔的谱型，衰减 0.55s，最长 2.0s。"""
    rel = 0.35
    n = _hold_len(dur, rel, sr, cap=2.0)
    span = n / sr
    env = exp_decay(n, sr, 0.006, 0.55) * gate_fade(n, sr, max(span - rel, 0.05), rel)
    fc = _cutoff_curve(n, sr, 2600.0, 500.0, 0.2)
    ratios, amps = _partials_saw(20, (0.0,), tilt=1.6)
    return additive_layer(freqs, n, sr, ratios, amps, env, fc, rng)


def gen_lead_hybrid(freqs, dur, vel, sr, rng):
    """主奏：有延音的锯条主体 + 短促拨弦起音层。

    单纯拨弦（`pluck`）的长音在 0.3 秒内就衰减完了，旋律「唱不住」；
    单纯锯齿又会失去咬字的起音。两层相加：起音给穿透力，主体把长音托住。
    """
    rel = 0.55
    n = _hold_len(dur, rel, sr)
    body_ratios, body_amps = _partials_saw(22, (-5.0, 5.0), tilt=1.0)
    body_amps = body_amps / 2.0
    env_body = adsr(n, sr, attack=0.035, decay=0.55, sustain=0.74, release=rel, hold=dur)
    fc_body = _cutoff_curve(n, sr, 3200.0, 1900.0, 0.6)
    body = additive_layer(
        freqs, n, sr, body_ratios, body_amps, env_body, fc_body, rng, vib_depth=0.005, vib_rate=4.5
    )
    plk_ratios, plk_amps = _partials_saw(18, (0.0,), tilt=1.0)
    env_plk = exp_decay(n, sr, 0.004, 0.26) * gate_fade(n, sr, min(dur, 0.55), 0.18)
    fc_plk = _cutoff_curve(n, sr, 4600.0 * (0.45 + 0.55 * vel), 900.0, 0.10)
    plk = additive_layer(freqs, n, sr, plk_ratios, plk_amps, env_plk, fc_plk, rng)
    return 0.78 * body + 0.52 * plk


def gen_sub_bass(freqs, dur, vel, sr, rng):
    """Sub Bass：正弦为主 + 半频 Sub 层 + 少量谐波（轻饱和感），无滤波器动作。

    ratio 0.5 的次八度让 65–123 Hz 的贝斯音也能填满 30–60 Hz 的 Sub 频段。
    """
    rel = 0.18
    n = _hold_len(dur, rel, sr)
    env = adsr(n, sr, attack=0.012, decay=0.12, sustain=0.92, release=rel, hold=dur)
    ratios = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
    amps = np.array([0.50, 1.0, 0.16, 0.06, 0.02])
    fc = np.full(n, 900.0)
    return additive_layer(freqs, n, sr, ratios, amps, env, fc, rng)


def gen_strings(freqs, dur, vel, sr, rng):
    """弦乐衬底：锯齿 2 层（±6 cents）+ 揉弦，起音 0.22s。"""
    rel = 1.0
    n = _hold_len(dur, rel, sr)
    ratios, amps = _partials_saw(22, (-6.0, 6.0), tilt=1.0)
    amps = amps / 2.0
    env = adsr(n, sr, attack=0.22, decay=0.5, sustain=0.88, release=rel, hold=dur)
    fc = _cutoff_curve(n, sr, 1500.0, 3000.0, 0.9)
    return additive_layer(freqs, n, sr, ratios, amps, env, fc, rng, vib_depth=0.008, vib_rate=5.2)


def gen_swell_strings(freqs, dur, vel, sr, rng):
    """长起音弦乐，用于段落推进。"""
    rel = 1.6
    n = _hold_len(dur, rel, sr)
    ratios, amps = _partials_saw(22, (-8.0, 8.0), tilt=1.0)
    amps = amps / 2.0
    env = adsr(n, sr, attack=1.1, decay=1.2, sustain=0.95, release=rel, hold=dur)
    fc = _cutoff_curve(n, sr, 1100.0, 2800.0, 1.4)
    return additive_layer(freqs, n, sr, ratios, amps, env, fc, rng, vib_depth=0.007, vib_rate=4.8)


def gen_shimmer(freqs, dur, vel, sr, rng):
    """高频空气层：高八度三角族 + 很慢起音，供高潮提亮。"""
    rel = 3.4
    n = _hold_len(dur, rel, sr)
    ratios, amps = _partials_saw(8, (-14.0, 0.0, 14.0, 27.0), tilt=2.0)
    amps = amps / 4.0
    env = adsr(n, sr, attack=1.6, decay=2.0, sustain=0.9, release=rel, hold=dur)
    fc = np.full(n, 8000.0)
    return additive_layer([f * 2.0 for f in freqs], n, sr, ratios, amps, env, fc, rng)


# ---------------------------------------------------------------- patch 注册表


@dataclass(frozen=True)
class PatchSpec:
    name: str
    kind: str  # "pitched" | "drum"
    low: int
    high: int
    nominal_db: float  # 归一化目标（见 norm 的解释）
    desc: str
    gen: Callable
    norm: str = "rms"  # "rms" = 有延音的铺底类；"peak" = 一次性击发类（拨弦/电钢/钟琴）


PITCHED_PATCHES: dict[str, PatchSpec] = {
    "saw-pad": PatchSpec("saw-pad", "pitched", 36, 76, -24.0, "暖 Pad：3 层失谐锯齿 + 慢起音", gen_saw_pad, "rms"),
    "glass-pad": PatchSpec("glass-pad", "pitched", 40, 88, -25.0, "玻璃 Pad：三角族 + 亮截止", gen_glass_pad, "rms"),
    "choir-pad": PatchSpec("choir-pad", "pitched", 36, 84, -24.5, "人声 Pad：共振峰加权 + 揉弦", gen_choir_pad, "rms"),
    "epiano": PatchSpec("epiano", "pitched", 36, 96, -12.0, "电钢琴 FM 1:1 + tine（峰值归一）", gen_epiano, "peak"),
    "bell": PatchSpec("bell", "pitched", 48, 103, -16.0, "钟琴 FM 1:3.5 长衰减（峰值归一）", gen_bell, "peak"),
    "music-box": PatchSpec("music-box", "pitched", 60, 105, -18.0, "八音盒 FM 1:7（峰值归一）", gen_music_box, "peak"),
    "pluck": PatchSpec("pluck", "pitched", 40, 100, -10.0, "拨弦：锯齿 + 快截止包络（峰值归一）", gen_pluck, "peak"),
    "lead-hybrid": PatchSpec("lead-hybrid", "pitched", 40, 98, -23.0, "主奏：延音锯条主体 + 拨弦起音层", gen_lead_hybrid, "rms"),
    "nylon": PatchSpec("nylon", "pitched", 40, 92, -12.0, "尼龙拨奏：柔谱型（峰值归一）", gen_nylon, "peak"),
    "sub-bass": PatchSpec("sub-bass", "pitched", 24, 60, -14.0, "Sub Bass：正弦 + 半频 Sub 层 + 轻谐波", gen_sub_bass, "rms"),
    "strings": PatchSpec("strings", "pitched", 40, 96, -24.0, "弦乐：失谐锯齿 + 揉弦", gen_strings, "rms"),
    "swell-strings": PatchSpec("swell-strings", "pitched", 36, 96, -25.0, "长起音弦乐", gen_swell_strings, "rms"),
    "shimmer": PatchSpec("shimmer", "pitched", 60, 108, -32.0, "高频空气层", gen_shimmer, "rms"),
}


def build_registry() -> dict[str, PatchSpec]:
    from . import drums

    registry = dict(PITCHED_PATCHES)
    registry[drums.DRUMKIT.name] = drums.DRUMKIT
    return registry


PATCHES: dict[str, PatchSpec] = build_registry()


def render_hit(patch: PatchSpec, midi_notes, dur: float, vel: float, sr: int, rng: np.random.Generator) -> np.ndarray:
    """按 patch 渲染一次触发（可能同时多音）。

    - 有音高音色：按 patch.norm 归一化 —— `rms` 用有效 RMS 目标（铺底类响度可比），
      `peak` 用峰值目标（拨弦/电钢/钟琴这类一次性击发音色）。
    - 打击乐：各鼓件自带绝对峰值电平（相对配比写在 drums.py），不再归一化。
    """
    if patch.kind == "drum":
        token = midi_notes if isinstance(midi_notes, str) else midi_notes[0]
        x = patch.gen(token, dur, vel, sr, rng)
        return np.asarray(x, dtype=np.float64)
    freqs = [theory.midi_to_hz(m) for m in midi_notes]
    x = np.asarray(patch.gen(freqs, dur, vel, sr, rng), dtype=np.float64)
    if patch.norm == "peak":
        return _norm_peak(x, patch.nominal_db)
    return _norm_rms(x, patch.nominal_db, sr)


def patch_table() -> list[tuple[str, str, str, str]]:
    rows = []
    for name, spec in sorted(PATCHES.items()):
        if spec.kind == "drum":
            rng = f"{spec.low}..{spec.high}" if spec.high else "无音高"
        else:
            rng = f"{theory.midi_to_name(spec.low)}..{theory.midi_to_name(spec.high)}"
        rows.append((name, spec.kind, rng, spec.desc))
    return rows
