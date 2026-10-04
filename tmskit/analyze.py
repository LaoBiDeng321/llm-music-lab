"""对渲染结果做数值验证：电平、动态弧、频谱、削波。给人看结论，不给机器看。"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from . import effects
from .score import Score


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        ch = f.getnchannels()
        sr = f.getframerate()
        raw = f.readframes(f.getnframes())
    x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if ch == 2:
        x = x.reshape(-1, 2)
    return x, sr


def _db(v: float) -> str:
    return "-inf" if v <= -119 else f"{v:6.2f}"


def band_energy(x: np.ndarray, sr: int, bands: tuple[tuple[float, float, str], ...]) -> list[tuple[str, float]]:
    mono = x if x.ndim == 1 else x.mean(axis=1)
    spec = np.abs(np.fft.rfft(mono * np.hanning(mono.shape[0])))
    f = np.fft.rfftfreq(mono.shape[0], 1.0 / sr)
    total = float(np.sum(spec**2)) + 1e-12
    out = []
    for lo, hi, name in bands:
        mask = (f >= lo) & (f < hi)
        out.append((name, 10.0 * np.log10((float(np.sum(spec[mask] ** 2)) + 1e-12) / total)))
    return out


DEFAULT_BANDS = (
    (20.0, 60.0, "sub 20-60"),
    (60.0, 250.0, "low 60-250"),
    (250.0, 500.0, "lowmid 250-500"),
    (500.0, 2000.0, "mid 0.5-2k"),
    (2000.0, 6000.0, "himid 2-6k"),
    (6000.0, 16000.0, "air 6-16k"),
)


def analyze(x: np.ndarray, sr: int, score: Score | None = None, window_bars: int = 4) -> dict:
    mono = x if x.ndim == 1 else x.mean(axis=1)
    peak = float(np.max(np.abs(x)))
    tp = effects.true_peak_db(x, sr)
    rms = effects.rms_db(x)
    clipped = int(np.sum(np.abs(x) >= 0.999))
    result: dict = {
        "duration": x.shape[0] / sr,
        "peak_db": 20.0 * np.log10(max(peak, 1e-12)),
        "true_peak_db": tp,
        "rms_db": rms,
        "clipped_samples": clipped,
        "crest_db": 20.0 * np.log10(max(peak, 1e-12)) - rms,
        "bands": band_energy(x, sr, DEFAULT_BANDS),
        "sections": [],
        "bars": [],
        "dc_offset": float(np.mean(x)),
        "stereo_correlation": None,
        "side_ratio_db": None,
    }
    if x.ndim == 2:
        left, right = x[:, 0], x[:, 1]
        denom = float(np.sqrt(np.mean(left**2)) * np.sqrt(np.mean(right**2)))
        result["stereo_correlation"] = float(np.mean(left * right) / denom) if denom > 1e-12 else 1.0
        mid = (left + right) / 2.0
        side = (left - right) / 2.0
        result["side_ratio_db"] = effects.rms_db(side) - effects.rms_db(mid)
    if score is not None:
        spb = 60.0 / score.tempo
        for s in score.sections:
            a = (s.start - 1) * score.beats_per_bar * spb
            b = s.end * score.beats_per_bar * spb
            seg = mono[int(a * sr) : int(min(b, mono.shape[0] / sr) * sr)]
            if seg.size:
                result["sections"].append((s.name, s.start, s.end, effects.rms_db(seg)))
        for bar in range(1, score.total_bars + 1, window_bars):
            a = (bar - 1) * score.beats_per_bar * spb
            b = min(bar - 1 + window_bars, score.total_bars) * score.beats_per_bar * spb
            seg = mono[int(a * sr) : int(b * sr)]
            if seg.size:
                result["bars"].append((bar, effects.rms_db(seg)))
    return result


def report(path: str | Path, score: Score | None = None) -> tuple[dict, list[str]]:
    x, sr = read_wav(path)
    info = analyze(x, sr, score)
    lines: list[str] = []
    lines.append(f"文件      {Path(path).name}")
    lines.append(f"时长      {info['duration']:.2f} s   采样率 {sr} Hz   声道 {1 if x.ndim == 1 else x.shape[1]}")
    lines.append(f"峰值      {_db(info['peak_db'])} dBFS    真峰值 {_db(info['true_peak_db'])} dBTP    削波样本 {info['clipped_samples']}")
    lines.append(f"整体 RMS  {_db(info['rms_db'])} dBFS    峰均比 {info['crest_db']:.2f} dB")
    extra = f"直流偏移 {info['dc_offset']:+.5f}"
    if info["stereo_correlation"] is not None:
        extra += f"    声道相关度 {info['stereo_correlation']:+.3f}    Side/Mid {info['side_ratio_db']:+.1f} dB"
    lines.append(extra)
    lines.append("频段能量占比")
    for name, val in info["bands"]:
        bar = "█" * max(int((val + 40) / 2), 0)
        lines.append(f"  {name:<14} {val:7.2f} dB  {bar[:34]}")
    if info["sections"]:
        lines.append("段落 RMS")
        for name, a, b, val in info["sections"]:
            lines.append(f"  {name:<10} 第{a:>3}-{b:<3} 小节  {_db(val)} dBFS")
    if info["bars"]:
        lines.append("动态曲线（每 4 小节，█ 每格 1 dB，以最响处为基准）")
        loudest = max(v for _, v in info["bars"])
        for bar, val in info["bars"]:
            lines.append(f"  第{bar:>3} 小节  {_db(val)}  {'█' * max(int(loudest - val), 0)}")
    verdict = []
    if info["clipped_samples"] > 0:
        verdict.append("✗ 存在削波样本")
    if not (-18.0 <= info["rms_db"] <= -12.0):
        verdict.append(f"✗ 整体 RMS {info['rms_db']:.2f} dBFS 不在 -18..-12")
    if info["true_peak_db"] > -1.0:
        verdict.append(f"✗ 真峰值 {info['true_peak_db']:.2f} dBTP 超过 -1")
    if info["sections"]:
        vals = [v for *_, v in info["sections"]]
        if max(vals) - min(vals) < 3.0:
            verdict.append(f"! 段落动态差仅 {max(vals) - min(vals):.1f} dB（<3 dB 会显得平）")
    if abs(info["dc_offset"]) > 0.001:
        verdict.append(f"! 存在直流偏移 {info['dc_offset']:+.4f}")
    if info["stereo_correlation"] is not None and info["stereo_correlation"] < 0.2:
        verdict.append(f"! 声道相关度仅 {info['stereo_correlation']:+.2f}，单声道下会明显掉内容")
    if not verdict:
        verdict.append("✓ 全部硬性门禁通过")
    lines.append("结论      " + "；".join(verdict))
    return info, lines
