"""静态校验：不渲染也能发现问题。返回 (错误列表, 警告列表)。"""

from __future__ import annotations

import numpy as np

from . import theory, voices
from .score import Score, ScoreError, parse_file, parse_text
from .render import expand


def validate(score: Score, parse_problems: list[str] | None = None) -> tuple[list[str], list[str]]:
    errors: list[str] = list(parse_problems or [])
    warnings: list[str] = []

    # ---- 声部
    if not score.instruments:
        errors.append("没有任何 instrument 声明")
    for name, inst in score.instruments.items():
        spec = voices.PATCHES.get(inst.patch)
        if spec is None:
            errors.append(f"声部 {name}: 未知 patch={inst.patch}（见 docs/02-音色与混音.md）")
        if not (-1.0 <= inst.pan <= 1.0):
            errors.append(f"声部 {name}: pan 须在 -1..1")
        if not (0.0 <= inst.reverb <= 1.0):
            errors.append(f"声部 {name}: reverb 须在 0..1")
        if not (0.0 <= inst.delay <= 1.0):
            errors.append(f"声部 {name}: delay 须在 0..1")
        if not (0.0 <= inst.chorus <= 1.0):
            errors.append(f"声部 {name}: chorus 须在 0..1")
        if inst.duck and inst.duck != "kick" and inst.duck not in score.instruments:
            errors.append(f"声部 {name}: duck={inst.duck} 指向不存在的声部")

    # ---- arrange
    if not score.placements:
        errors.append("arrange 为空，没有任何内容可渲染")
    for pl in score.placements:
        if pl.track not in score.instruments:
            errors.append(f"第 {pl.line} 行: arrange 引用了未声明的声部 {pl.track}")
            continue
        pat = score.patterns.get((pl.track, pl.pattern))
        if pat is None:
            errors.append(f"第 {pl.line} 行: arrange 引用了未定义的 pattern {pl.track}.{pl.pattern}")
            continue
        span_bars = pat.loop_length(score.beats_per_bar) / score.beats_per_bar
        for a, b in pl.ranges:
            declared = b - a + 1
            if span_bars > declared + 1e-6:
                warnings.append(
                    f"第 {pl.line} 行: pattern {pl.track}.{pl.pattern} 长 {span_bars:.2f} 小节，"
                    f"超过声明范围 {declared} 小节（会越过范围末尾）"
                )
            elif abs(declared / span_bars - round(declared / span_bars)) > 1e-6:
                warnings.append(
                    f"第 {pl.line} 行: 范围 {a}-{b} 长 {declared} 小节，不是 pattern "
                    f"{pl.track}.{pl.pattern}（{span_bars:.2f} 小节）的整数倍，最后一遍会被截断"
                )
        spec = voices.PATCHES.get(score.instruments[pl.track].patch)
        for ev in pat.events:
            if ev.token in theory.DRUM_VOICES:
                if spec is not None and spec.kind != "drum":
                    warnings.append(f"第 {ev.line} 行: 鼓件 {ev.token} 落在非鼓组音色 {spec.name} 上")
                continue
            if spec is not None and spec.kind == "drum":
                warnings.append(f"第 {ev.line} 行: 有音高事件 {ev.token} 落在鼓组音色上（会被忽略或报错）")
                continue
            shift = 12 * (score.instruments[pl.track].octave + pl.octave) + pl.transpose
            if theory.is_pitch(ev.token):
                midis = [theory.parse_pitch(ev.token) + shift]
            else:
                midis = [m + pl.transpose for m in theory.chord_voicing(ev.token, octave=score.instruments[pl.track].octave + pl.octave)]
            for m in midis:
                if spec and (m < spec.low or m > spec.high):
                    warnings.append(
                        f"第 {ev.line} 行: {theory.midi_to_name(m)} 超出 {spec.name} 推荐音域"
                        f" {theory.midi_to_name(spec.low)}..{theory.midi_to_name(spec.high)}"
                    )
            if score.key_token and theory.is_pitch(ev.token):
                midi = midis[0]
                if not theory.in_scale(midi, score.tonic_pc, score.scale):
                    # 小调允许升七级（导音）：自然小调与和声小调混用是常规写法
                    leading = (score.tonic_pc + 11) % 12
                    if not (3 in score.scale and midi % 12 == leading):
                        warnings.append(
                            f"第 {ev.line} 行: {ev.token} 不在 {score.key_token} 调内（可能是刻意的色彩音）"
                        )

    # ---- 强拍和弦音检查
    harmony = _harmony_map(score)
    for pl in score.placements:
        inst = score.instruments.get(pl.track)
        pat = score.patterns.get((pl.track, pl.pattern))
        if inst is None or pat is None:
            continue
        spec = voices.PATCHES.get(inst.patch)
        if spec is None or spec.kind == "drum":
            continue
        for (a, _b) in pl.ranges:
            for ev in pat.events:
                if not theory.is_pitch(ev.token):
                    continue
                bar = a + int(ev.start // score.beats_per_bar)
                pos = ev.start % score.beats_per_bar
                if abs(pos) > 1e-6:
                    continue
                chords = harmony.get(bar)
                if not chords:
                    continue
                pcs = set()
                for c in chords:
                    pcs |= theory.chord_pitch_classes(c)
                if not pcs:
                    continue
                midi = theory.parse_pitch(ev.token)
                if midi % 12 not in pcs:
                    warnings.append(
                        f"第 {ev.line} 行: 第 {bar} 小节强拍上的 {ev.token} 不是当时和弦"
                        f"（{'/'.join(sorted(chords))}）的和弦音"
                    )

    # ---- 同轨重叠：同一小节被两条条目覆盖 = 两层叠加，几乎总是笔误
    occupied: dict[tuple[str, int], tuple[str, int]] = {}
    for pl in score.placements:
        for (a, b) in pl.ranges:
            for bar in range(a, b + 1):
                key = (pl.track, bar)
                prev = occupied.get(key)
                if prev is not None:
                    same = "（同一个 pattern 重复放置）" if prev[0] == pl.pattern else ""
                    warnings.append(
                        f"声部 {pl.track} 第 {bar} 小节被两条条目叠加："
                        f"第 {prev[1]} 行的 {pl.track}.{prev[0]} 与第 {pl.line} 行的 {pl.track}.{pl.pattern}{same}"
                    )
                occupied[key] = (pl.pattern, pl.line)

    # ---- 展开后的密度与重叠
    hits = expand(score)
    if not hits:
        errors.append("展开后没有任何发声事件")
    seen: dict[tuple[str, str, int, tuple[int, ...]], int] = {}
    for h in hits:
        key = (h.track, h.token, int(round(h.start_beat * 1000)), h.midis)
        seen[key] = seen.get(key, 0) + 1
    for (track, token, ms, _), count in seen.items():
        if count > 1:
            warnings.append(f"声部 {track} 在 {ms / 1000:.3f} 拍处有 {count} 个完全相同的 {token} 事件（重复书写？）")

    # ---- 动态弧自检
    if hits:
        bars = np.zeros(score.total_bars + 2)
        for h in hits:
            bar = min(int(h.start_beat // score.beats_per_bar) + 1, score.total_bars)
            bars[bar] += 1
        empty = [i for i in range(1, score.total_bars + 1) if bars[i] == 0]
        if empty:
            warnings.append(f"以下小节没有任何事件（可能是编曲空洞）：{empty[:12]}{' …' if len(empty) > 12 else ''}")
    return errors, warnings


def _harmony_map(score: Score) -> dict[int, set[str]]:
    """每小节出现的和弦记号（来自所有音色非鼓的轨道）。"""
    out: dict[int, set[str]] = {}
    for pl in score.placements:
        inst = score.instruments.get(pl.track)
        pat = score.patterns.get((pl.track, pl.pattern))
        if inst is None or pat is None:
            continue
        spec = voices.PATCHES.get(inst.patch)
        if spec is None or spec.kind == "drum":
            continue
        for (a, _b) in pl.ranges:
            for ev in pat.events:
                if theory.is_chord(ev.token):
                    bar = a + int(ev.start // score.beats_per_bar)
                    out.setdefault(bar, set()).add(ev.token)
    return out


def check_file(path: str) -> tuple[list[str], list[str], Score]:
    score, problems = parse_file(path)
    errors, warnings = validate(score, problems)
    return errors, warnings, score


def check_text(text: str) -> tuple[list[str], list[str], Score]:
    score, problems = parse_text(text)
    errors, warnings = validate(score, problems)
    return errors, warnings, score


__all__ = ["validate", "check_file", "check_text", "ScoreError"]
