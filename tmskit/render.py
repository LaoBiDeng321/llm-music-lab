"""渲染调度：Score → 分轨缓冲 → 效果 → 总线 → 音频数组 / WAV 文件。

时间轴一律先算「拍」，最后一次性换算成秒，保证变速/摇摆/人性化都不互相污染。
"""

from __future__ import annotations

import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import effects, theory, voices
from .score import Instrument, Pattern, Score

SR = 44100
TAIL_SECONDS = 4.5  # 混响/延迟尾巴预留
MASTER_TARGET_RMS_DB = -14.5
MASTER_CEILING_DB = -1.0


@dataclass(frozen=True)
class Hit:
    """一次发声：单音 / 和弦同时音 / 鼓件。"""

    track: str
    patch: str
    token: str
    start_beat: float
    dur_beats: float
    vel: float
    midis: tuple[int, ...] = ()
    drum: bool = False
    gain: float = 1.0          # 来自 arrange 条目的 gain 修饰（与 instrument 的 gain 相乘）
    pan: float | None = None   # 来自 arrange 条目的 pan 修饰，None 表示用 instrument 的 pan

    @property
    def is_drum(self) -> bool:
        return self.drum


@dataclass
class RenderResult:
    audio: np.ndarray
    sr: int
    hits: list[Hit] = field(default_factory=list)
    track_rms: dict[str, float] = field(default_factory=dict)
    section_rms: dict[str, list[float]] = field(default_factory=dict)
    score: Score | None = None
    duration: float = 0.0
    pre_limit_peak_db: float = -120.0


# ---------------------------------------------------------------- 展开


def _swing_offset(start_beat: float, swing: float, beats_per_bar: float) -> float:
    if swing <= 0:
        return 0.0
    pos = start_beat % beats_per_bar
    sixteenth = int(round(pos * 4.0))
    return swing * 0.25 if sixteenth % 4 == 2 else 0.0


def _chord_sequence(
    token: str, mode: str, subdiv: float, dur_beats: float, octave: int, accent: tuple[float, ...] = ()
) -> list[tuple[float, float, tuple[int, ...], float]]:
    """和弦按 chord 模式展开成 (相对起拍, 时值, MIDI 音组, 力度倍率) 序列。"""
    voicing = tuple(sorted(theory.chord_voicing(token, octave=octave)))

    def acc(i: int) -> float:
        return accent[i % len(accent)] if accent else 1.0

    if not voicing:
        return []
    if mode == "hold":
        return [(0.0, dur_beats, voicing, 1.0)]
    if mode in ("root", "root8"):
        root = theory.parse_chord(token)[0] + 12 * octave
        if mode == "root":
            return [(0.0, dur_beats, (root,), 1.0)]
        steps = max(int(np.floor(dur_beats / subdiv + 1e-6)), 1)
        # 每 4 个八分音符把最后一个换成五度，形成 root-root-root-fifth 的律动
        return [
            (i * subdiv, subdiv * 0.92, (root + (7 if i % 4 == 3 else 0),), acc(i))
            for i in range(steps)
        ]
    steps = max(int(np.floor(dur_beats / subdiv + 1e-6)), 1)
    if mode == "stab":
        return [(i * subdiv, subdiv * 0.45, voicing, acc(i)) for i in range(steps)]
    if mode == "arp-down":
        cycle = list(reversed(voicing))
    elif mode == "arp-updown":
        cycle = list(voicing) + (list(reversed(voicing))[1:-1] if len(voicing) > 2 else [])
    elif mode == "arp-up8":
        cycle = [n + 12 * o for o in range(2) for n in voicing]
    else:  # arp-up
        cycle = list(voicing)
    return [(i * subdiv, subdiv, (cycle[i % len(cycle)],), acc(i)) for i in range(steps)]


def expand(score: Score) -> list[Hit]:
    """把 arrange + pattern 展开成绝对时间轴上的一次次发声。"""
    hits: list[Hit] = []
    bpb = score.beats_per_bar
    rng = np.random.default_rng([score.seed, 20240517])
    for pi, placement in enumerate(score.placements):
        inst = score.instruments.get(placement.track)
        if inst is None:
            continue
        pattern = score.patterns.get((placement.track, placement.pattern))
        if pattern is None:
            continue
        for ri, (bar_start, bar_end) in enumerate(placement.ranges):
            span = pattern.loop_length(bpb)
            range_beats = (bar_end - bar_start + 1) * bpb
            reps = max(int(round(range_beats / span)), 1) if span > 1e-6 else 1
            for rep in range(reps):
                base_beat = (bar_start - 1) * bpb + rep * span
                for ei, ev in enumerate(pattern.events):
                    start = base_beat + ev.start
                    start += _swing_offset(start, score.swing, bpb)
                    hum = inst.humanize if inst.humanize is not None else score.humanize
                    if hum > 0:
                        jitter_beats = rng.uniform(-hum, hum) * score.tempo / 60.0
                        start = max(start + jitter_beats, 0.0)
                    vel = float(np.clip(ev.vel * (1.0 + (rng.uniform(-0.1, 0.1) if hum > 0 else 0.0)), 0.02, 1.0))
                    shift = 12 * (inst.octave + placement.octave) + placement.transpose
                    token = ev.token
                    if token in theory.DRUM_VOICES:
                        hits.append(
                            Hit(
                                track=inst.name,
                                patch=inst.patch,
                                token=token,
                                start_beat=start,
                                dur_beats=ev.dur,
                                vel=vel,
                                drum=True,
                                gain=placement.gain,
                                pan=placement.pan,
                            )
                        )
                        continue
                    if theory.is_pitch(token):
                        midis = (theory.parse_pitch(token) + shift,)
                        hits.append(
                            Hit(
                                track=inst.name,
                                patch=inst.patch,
                                token=token,
                                start_beat=start,
                                dur_beats=ev.dur,
                                vel=vel,
                                midis=midis,
                                gain=placement.gain,
                                pan=placement.pan,
                            )
                        )
                        continue
                    if theory.is_chord(token):
                        for offset, cdur, cmidis, cacc in _chord_sequence(
                            token,
                            inst.chord,
                            inst.subdiv,
                            ev.dur,
                            inst.octave + placement.octave,
                            inst.accent,
                        ):
                            hits.append(
                                Hit(
                                    track=inst.name,
                                    patch=inst.patch,
                                    token=token,
                                    start_beat=start + offset,
                                    dur_beats=cdur,
                                    vel=float(np.clip(vel * cacc, 0.02, 1.0)),
                                    midis=tuple(m + placement.transpose for m in cmidis),
                                    gain=placement.gain,
                                    pan=placement.pan,
                                )
                            )
    hits.sort(key=lambda h: (h.start_beat, h.track, h.token))
    return hits


# ---------------------------------------------------------------- 渲染


def _place(buf: np.ndarray, x: np.ndarray, offset: int) -> None:
    if offset >= buf.shape[0] or x.size == 0:
        return
    end = min(offset + x.shape[0], buf.shape[0])
    buf[offset:end] += x[: end - offset]


def render_score(score: Score, sr: int = SR, verbose: bool = True) -> RenderResult:
    hits = expand(score)
    total_beats = score.total_beats
    total_s = score.beats_to_seconds(total_beats) + TAIL_SECONDS
    n = int(total_s * sr)

    # 每个「声部 × 有效声场位置」一条缓冲：arrange 条目可用 pan= 覆盖 instrument 的 pan
    buffers: dict[tuple[str, float], np.ndarray] = {
        (name, round(inst.pan, 4)): np.zeros(n) for name, inst in score.instruments.items()
    }
    buffer_pan: dict[tuple[str, float], float] = {k: k[1] for k in buffers}
    kick_key = np.zeros(n)
    patch_cache = voices.PATCHES
    spb = 60.0 / score.tempo  # 秒/拍
    # 音色渲染缓存：同一 (音色, 音高组合, 时值, 量化力度) 只合成一次。
    # 乐谱里同一和弦/同一琶音音型会反复出现，缓存能把整曲渲染时间压到几分之一。
    hit_cache: dict[tuple, np.ndarray] = {}

    for hi, hit in enumerate(hits):
        inst = score.instruments.get(hit.track)
        if inst is None:
            continue
        spec = patch_cache.get(hit.patch)
        if spec is None:
            continue
        dur_s = max(hit.dur_beats * spb, 0.02)
        if spec.kind == "drum":
            rng = np.random.default_rng([score.seed, hi, 991])
            x = voices.render_hit(spec, hit.token, dur_s, hit.vel, sr, rng)
        else:
            qvel = round(hit.vel, 1)
            key = (hit.patch, hit.midis, round(dur_s, 4), qvel)
            x = hit_cache.get(key)
            if x is None:
                rng = np.random.default_rng([score.seed, len(hit_cache) + 1, 7])
                x = voices.render_hit(spec, hit.midis, dur_s, qvel, sr, rng)
                if len(hit_cache) < 800:
                    hit_cache[key] = x
        if x.size == 0:
            continue
        x = x * (hit.vel**1.5)
        x = x * inst.gain * hit.gain
        offset = int(hit.start_beat * spb * sr)
        pan = inst.pan if hit.pan is None else float(np.clip(hit.pan, -1.0, 1.0))
        bkey = (hit.track, round(pan, 4))
        if bkey not in buffers:
            buffers[bkey] = np.zeros(n)
            buffer_pan[bkey] = pan
        _place(buffers[bkey], x, offset)
        if hit.drum and hit.token == "kick":
            _place(kick_key, x * 0.6, offset)
        if verbose and hi % 250 == 0:
            print(f"  渲染 {hi}/{len(hits)} 次发声…", flush=True)

    # ---- 分轨处理：合唱
    processed: dict[tuple[str, float], np.ndarray] = {}
    for bkey, buf in buffers.items():
        inst = score.instruments[bkey[0]]
        processed[bkey] = effects.chorus(buf, sr, rate=0.28, depth_ms=8.0, mix=inst.chorus, voices=2) if inst.chorus > 0 else buf

    # ---- 送出：混响 / 延迟
    reverb_send = np.zeros(n)
    delay_send = np.zeros(n)
    for bkey, buf in processed.items():
        inst = score.instruments[bkey[0]]
        mono = buf if buf.ndim == 1 else buf.mean(axis=1)
        if inst.reverb > 0:
            reverb_send += mono * inst.reverb
        if inst.delay > 0:
            delay_send += mono * inst.delay

    ir = effects.make_impulse_response(sr, decay=2.8, predelay=0.032, brightness=0.6, rng=np.random.default_rng([score.seed, 5]))
    wet = effects.fft_convolve(reverb_send, ir) * 0.55
    wet = np.asarray(wet)[:n]
    wet = effects.fft_filter(wet, sr, hp=180.0, order=2)
    echo = effects.feedback_delay(delay_send, sr, time_s=spb * 0.75, feedback=0.32, wet=0.34, pingpong=True)[:n]
    echo = effects.fft_filter(echo, sr, hp=320.0, lp=5200.0)

    # ---- 汇总到立体声总线（每个缓冲用自己那一份 pan：可能来自 arrange 的 pan= 覆盖）
    bus = np.zeros((n, 2))
    for bkey, buf in processed.items():
        track, _ = bkey
        inst = score.instruments[track]
        pan = buffer_pan.get(bkey, inst.pan)
        if inst.duck:
            key = kick_key if inst.duck == "kick" else next(
                (b for k, b in buffers.items() if k[0] == inst.duck), np.zeros(0)
            )
            if key.size:
                buf = effects.sidechain_duck(buf, key, sr, depth_db=-4.5)
        left = float(np.clip(1.0 - max(pan, 0.0), 0.0, 1.0))
        right = float(np.clip(1.0 + min(pan, 0.0), 0.0, 1.0))
        if buf.ndim == 1:
            bus[:, 0] += buf * left
            bus[:, 1] += buf * right
        else:
            bus[:, 0] += buf[:, 0] * left
            bus[:, 1] += buf[:, 1] * right
    bus += wet
    bus += echo

    # ---- 母带：EQ → 胶水压缩 → RMS 归一 → 声场处理 → 限幅 → 真峰值兜底
    bus = effects.fft_filter(bus, sr, hp=28.0, order=4)
    bus = effects.peaking_eq(bus, sr, 350.0, -2.2, q=0.9)
    bus = effects.peaking_eq(bus, sr, 58.0, 2.0, q=0.7)
    bus = effects.peaking_eq(bus, sr, 160.0, 1.8, q=0.8)
    bus = effects.peaking_eq(bus, sr, 2400.0, 1.0, q=1.1)
    bus = effects.peaking_eq(bus, sr, 10500.0, 2.8, q=0.7)
    bus = effects.compress(bus, sr, threshold_db=-11.0, ratio=1.6, attack_ms=25.0, release_ms=180.0, makeup_db=0.0)
    bus = effects.normalize_rms(bus, MASTER_TARGET_RMS_DB)
    bus = effects.mono_bass(bus, sr, fc=120.0)
    bus = effects.stereo_width(bus, 1.08)
    pre_peak = 20.0 * np.log10(max(float(np.max(np.abs(bus))), 1e-12))
    bus = effects.limiter(bus, sr, ceiling_db=MASTER_CEILING_DB, lookahead_ms=6.0, release_ms=90.0)
    tp = effects.true_peak_db(bus, sr)
    if tp > MASTER_CEILING_DB:
        bus = bus * voices.db_to_gain(MASTER_CEILING_DB - tp)

    track_rms: dict[str, float] = {}
    section_rms: dict[str, list[float]] = {}
    for bkey, buf in processed.items():
        track, _ = bkey
        mono = buf if buf.ndim == 1 else buf.mean(axis=1)
        track_rms[track] = max(track_rms.get(track, -120.0), effects.rms_db(mono))
        row = section_rms.setdefault(track, [-120.0] * len(score.sections))
        for i, s in enumerate(score.sections):
            a = int((s.start - 1) * score.beats_per_bar * spb * sr)
            b = int(s.end * score.beats_per_bar * spb * sr)
            seg = mono[a:b]
            if seg.size:
                row[i] = max(row[i], effects.rms_db(seg))

    return RenderResult(
        audio=bus,
        sr=sr,
        hits=hits,
        track_rms=track_rms,
        section_rms=section_rms,
        score=score,
        duration=total_s,
        pre_limit_peak_db=pre_peak,
    )


# ---------------------------------------------------------------- 写文件


def write_wav(path: str | Path, audio: np.ndarray, sr: int = SR, dither: bool = True) -> Path:
    """16-bit PCM WAV；写盘前加 TPDF 抖动，避免量化噪声在安静段落可闻。"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    x = np.clip(audio, -1.0, 1.0)
    if dither:
        rng = np.random.default_rng(12345)
        tpdf = (rng.random(x.shape) - rng.random(x.shape)) / 32768.0
        x = x + tpdf
    ints = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(p), "wb") as f:
        f.setnchannels(2 if ints.ndim == 2 else 1)
        f.setsampwidth(2)
        f.setframerate(sr)
        f.writeframes(ints.tobytes())
    return p
