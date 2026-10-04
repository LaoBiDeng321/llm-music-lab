"""TMS 文本乐谱解析：把 .tms 文件读成 Score 数据模型。只负责解析，不做任何渲染。"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import theory

# ---------------------------------------------------------------- 数据模型


@dataclass(frozen=True)
class Instrument:
    name: str
    patch: str
    gain: float = 1.0
    pan: float = 0.0
    reverb: float = 0.0
    delay: float = 0.0
    chord: str = "hold"
    subdiv: float = 0.5
    octave: int = 0
    humanize: float | None = None
    duck: str | None = None
    chorus: float = 0.0
    accent: tuple[float, ...] = ()
    line: int = 0


@dataclass(frozen=True)
class Event:
    token: str
    start: float
    dur: float
    vel: float
    line: int


@dataclass
class Pattern:
    track: str
    name: str
    events: list[Event] = field(default_factory=list)
    line: int = 0

    @property
    def length_beats(self) -> float:
        """原始结束位置：最后一个事件的「起拍 + 时值」。"""
        if not self.events:
            return 0.0
        return max(e.start + e.dur for e in self.events)

    def loop_length(self, beats_per_bar: float) -> float:
        """循环长度：把原始结束位置向上取整到整小节。

        这一条很关键：鼓组网格若最后一格是休止（例如 hat 只写到第 14 格），
        原始长度会是 3.75 拍而不是 4 拍，循环时每小节漂 0.25 拍，整段鼓点会逐步错位。
        取整到整小节后，1 小节的 pattern 放进 8 小节范围就是规规矩矩的 8 遍。
        """
        raw = self.length_beats
        if raw <= 0:
            return 0.0
        bars = math.ceil(raw / beats_per_bar - 1e-9)
        return max(bars, 1) * beats_per_bar


@dataclass(frozen=True)
class Placement:
    track: str
    pattern: str
    ranges: tuple[tuple[int, int], ...]
    transpose: int = 0
    gain: float = 1.0
    pan: float | None = None
    octave: int = 0
    line: int = 0

    @property
    def key(self) -> tuple[str, str]:
        return (self.track, self.pattern)


@dataclass(frozen=True)
class Section:
    name: str
    start: int
    end: int


@dataclass
class Score:
    path: Path
    tempo: float = 120.0
    beats_per_bar: float = 4.0
    key_token: str | None = None
    tonic_pc: int = 0
    scale: tuple[int, ...] = theory.SCALES["major"]
    swing: float = 0.0
    humanize: float = 0.0
    seed: int = 0
    instruments: dict[str, Instrument] = field(default_factory=dict)
    patterns: dict[tuple[str, str], Pattern] = field(default_factory=dict)
    placements: list[Placement] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)

    # -------------------------------------------------- 便捷查询

    @property
    def total_bars(self) -> int:
        if not self.placements:
            return 0
        return max(end for p in self.placements for _, end in p.ranges)

    @property
    def total_beats(self) -> float:
        return self.total_bars * self.beats_per_bar

    def beats_to_seconds(self, beats: float) -> float:
        return beats * 60.0 / self.tempo

    def pattern(self, track: str, name: str) -> Pattern | None:
        return self.patterns.get((track, name))

    def section_of_bar(self, bar: int) -> str:
        for s in self.sections:
            if s.start <= bar <= s.end:
                return s.name
        return "?"


# ---------------------------------------------------------------- 解析器

_PITCH_LIMIT_BEATS = 32.0
_CHORD_MODES = ("hold", "arp-up", "arp-down", "arp-updown", "arp-up8", "stab", "root", "root8")


class ScoreError(Exception):
    """乐谱语法/取值错误，带行号，供 check 命令聚合展示。"""

    def __init__(self, message: str, line: int, path: Path | None = None):
        loc = f"{path.name}:{line}" if path else f"line {line}"
        super().__init__(f"{loc}: {message}")
        self.line = line
        self.message = message


def _strip_comment(raw: str) -> str:
    out = []
    prev_ws = True
    for ch in raw:
        if ch == "#" and prev_ws:
            break
        out.append(ch)
        prev_ws = ch.isspace()
    return "".join(out).rstrip()


def _kv_pairs(tokens: list[str], line: int, path: Path, allowed: set[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for tok in tokens:
        if "=" not in tok:
            raise ScoreError(f"参数必须写成 key=value，收到 {tok!r}", line, path)
        k, v = tok.split("=", 1)
        k = k.strip().lower()
        if k not in allowed:
            raise ScoreError(f"未知参数 {k!r}（可用：{', '.join(sorted(allowed))}）", line, path)
        if k in result:
            raise ScoreError(f"参数 {k!r} 重复", line, path)
        result[k] = v.strip()
    return result


def _as_float(value: str, name: str, line: int, path: Path) -> float:
    try:
        return float(value)
    except ValueError:
        raise ScoreError(f"参数 {name} 需要数字，收到 {value!r}", line, path) from None


def _as_int(value: str, name: str, line: int, path: Path) -> int:
    try:
        return int(value)
    except ValueError:
        raise ScoreError(f"参数 {name} 需要整数，收到 {value!r}", line, path) from None


def _parse_grid_symbols(text: str) -> list[float]:
    """`X.x.-o` → 每格力度（0 表示休止）。"""
    table = {"X": 1.0, "x": 0.8, "o": 0.7, ".": 0.0, "-": 0.0}
    return [table[ch] for ch in text if not ch.isspace()]


_GRID_CHARS = set("Xxo.-")


def _parse_accent(text: str, line: int, path: Path) -> tuple[float, ...]:
    """`accent=1,.72,.85,.72` → 循环施加在琶音/柱式步进上的力度倍率。"""
    if not text:
        return ()
    out: list[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        val = _as_float(chunk, "accent", line, path)
        if not (0.0 <= val <= 1.0):
            raise ScoreError(f"accent 取值须在 0..1，收到 {val}", line, path)
        out.append(val)
    return tuple(out)


def parse_text(text: str, path: Path | None = None) -> tuple[Score, list[str]]:
    """解析乐谱文本 → (Score, 问题列表)。问题不抛异常，交给 validate 决定严重度。"""
    path = Path(path) if path else Path("<memory>")
    score = Score(path=path)
    problems: list[str] = []

    lines = text.splitlines()
    idx = 0
    n = len(lines)
    in_arrange = False
    grid_steps: int | None = None
    grid_bar_count: dict[str, int] = {}

    while idx < n:
        raw = lines[idx]
        lineno = idx + 1
        stripped = _strip_comment(raw).strip()
        idx += 1
        if not stripped:
            continue
        lowered = stripped.lower()

        # ---------------- pattern 块
        if lowered.startswith("pattern "):
            head = stripped[len("pattern ") :].strip()
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\s*\{$", head)
            if not m:
                problems.append(f"{path.name}:{lineno}: pattern 头应形如 `pattern track.name {{`，收到 {stripped!r}")
                continue
            track, name = m.groups()
            pat = Pattern(track=track, name=name, line=lineno)
            if (track, name) in score.patterns:
                problems.append(f"{path.name}:{lineno}: pattern {track}.{name} 重复定义（后定义覆盖前者）")
            grid_steps = None
            grid_bar_count = {}
            # 读块体
            while idx < n:
                body_raw = lines[idx]
                body_line = idx + 1
                idx += 1
                body = _strip_comment(body_raw).strip()
                if body == "}":
                    break
                if not body:
                    continue
                if body.lower().startswith("grid "):
                    steps_txt = body[5:].strip()
                    try:
                        grid_steps = int(steps_txt)
                    except ValueError:
                        problems.append(f"{path.name}:{body_line}: grid 需要整数格数，收到 {steps_txt!r}")
                        grid_steps = None
                        continue
                    if grid_steps <= 0 or score.beats_per_bar / grid_steps <= 0:
                        problems.append(f"{path.name}:{body_line}: grid 格数必须为正")
                    grid_bar_count = {}
                    continue
                parts = body.split(None, 1)
                if len(parts) < 2:
                    problems.append(f"{path.name}:{body_line}: 事件行应形如 `<音高> <起拍> <时值> [力度]`，收到 {body!r}")
                    continue
                first, rest = parts[0], parts[1].strip()
                rest_tokens = rest.split()
                looks_like_grid = bool(rest_tokens) and all(set(t) <= _GRID_CHARS for t in rest_tokens)
                if looks_like_grid and grid_steps is None:
                    problems.append(f"{path.name}:{body_line}: 网格行必须出现在 `grid <格数>` 之后")
                    continue
                if looks_like_grid:
                    symbols = _parse_grid_symbols("".join(rest_tokens))
                    if len(symbols) != grid_steps:
                        problems.append(
                            f"{path.name}:{body_line}: 网格行 {first} 有 {len(symbols)} 个符号，应为 {grid_steps} 个"
                        )
                        continue
                    if first not in theory.DRUM_VOICES:
                        problems.append(f"{path.name}:{body_line}: 网格声部 {first!r} 不是鼓件名")
                        continue
                    step = score.beats_per_bar / grid_steps
                    bar_off = grid_bar_count.get(first, 0)
                    for i, vel in enumerate(symbols):
                        if vel <= 0:
                            continue
                        pat.events.append(
                            Event(
                                token=first,
                                start=bar_off * score.beats_per_bar + i * step,
                                dur=step,
                                vel=vel,
                                line=body_line,
                            )
                        )
                    grid_bar_count[first] = bar_off + 1
                    continue
                # 普通事件行
                fields = body.split()
                if len(fields) < 3:
                    problems.append(f"{path.name}:{body_line}: 事件行至少要有 音高 起拍 时值，收到 {body!r}")
                    continue
                token = fields[0]
                try:
                    start = float(fields[1])
                    dur = float(fields[2])
                except ValueError:
                    problems.append(f"{path.name}:{body_line}: 起拍/时值必须是数字，收到 {body!r}")
                    continue
                vel = 0.8
                if len(fields) >= 4:
                    try:
                        vel = float(fields[3])
                    except ValueError:
                        problems.append(f"{path.name}:{body_line}: 力度必须是数字，收到 {fields[3]!r}")
                        continue
                if not (theory.is_pitch(token) or theory.is_chord(token) or token in theory.DRUM_VOICES):
                    problems.append(f"{path.name}:{body_line}: 无法识别的音高/和弦/鼓件记号 {token!r}")
                    continue
                if dur <= 0:
                    problems.append(f"{path.name}:{body_line}: 时值必须 > 0，收到 {dur}")
                    continue
                if start < 0 or start >= _PITCH_LIMIT_BEATS:
                    problems.append(f"{path.name}:{body_line}: 起拍须在 0..{_PITCH_LIMIT_BEATS} 拍内，收到 {start}")
                    continue
                if not (0.0 <= vel <= 1.0):
                    problems.append(f"{path.name}:{body_line}: 力度须在 0..1，收到 {vel}")
                    continue
                pat.events.append(Event(token=token, start=start, dur=dur, vel=vel, line=body_line))
            pat.events.sort(key=lambda e: (e.start, e.token))
            score.patterns[(track, name)] = pat
            continue

        # ---------------- arrange 块
        if lowered == "arrange":
            in_arrange = True
            continue

        # ---------------- 顶层指令
        if in_arrange and lowered.startswith("bars "):
            body = stripped[5:].strip()
            fields = body.split()
            if len(fields) < 2:
                problems.append(f"{path.name}:{lineno}: arrange 行应形如 `bars 1-8 track.pattern`")
                continue
            range_txt = fields[0]
            rest = " ".join(fields[1:])
            ranges: list[tuple[int, int]] = []
            for chunk in range_txt.split("|"):
                m = re.match(r"^(\d+)(?:-(\d+))?$", chunk.strip())
                if not m:
                    problems.append(f"{path.name}:{lineno}: 小节范围应形如 9-16 或 9，收到 {chunk!r}")
                    ranges = []
                    break
                a = int(m.group(1))
                b = int(m.group(2)) if m.group(2) else a
                if a < 1 or b < a:
                    problems.append(f"{path.name}:{lineno}: 小节范围非法 {a}-{b}")
                    ranges = []
                    break
                ranges.append((a, b))
            if not ranges:
                continue
            for item in rest.split(","):
                item = item.strip()
                if not item:
                    continue
                parts = item.split()
                head = parts[0]
                if "." not in head:
                    problems.append(f"{path.name}:{lineno}: arrange 条目应形如 track.pattern，收到 {head!r}")
                    continue
                track, pname = head.split(".", 1)
                mods = _kv_pairs(
                    parts[1:], lineno, path, {"transpose", "gain", "pan", "octave"}
                ) if len(parts) > 1 else {}
                transpose = _as_int(mods["transpose"], "transpose", lineno, path) if "transpose" in mods else 0
                gain = _as_float(mods["gain"], "gain", lineno, path) if "gain" in mods else 1.0
                pan = _as_float(mods["pan"], "pan", lineno, path) if "pan" in mods else None
                octave = _as_int(mods["octave"], "octave", lineno, path) if "octave" in mods else 0
                score.placements.append(
                    Placement(
                        track=track,
                        pattern=pname,
                        ranges=tuple(ranges),
                        transpose=transpose,
                        gain=gain,
                        pan=pan,
                        octave=octave,
                        line=lineno,
                    )
                )
            continue

        if lowered.startswith("section "):
            m = re.match(r"^section\s+(.+?)\s+bars\s+(\d+)-(\d+)$", stripped, re.IGNORECASE)
            if not m:
                problems.append(f"{path.name}:{lineno}: section 行应形如 `section 主歌 bars 9-16`")
                continue
            score.sections.append(Section(m.group(1).strip(), int(m.group(2)), int(m.group(3))))
            continue

        if lowered.startswith("instrument "):
            fields = stripped.split()
            if len(fields) < 2:
                problems.append(f"{path.name}:{lineno}: instrument 行缺少名称")
                continue
            name = fields[1]
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                problems.append(f"{path.name}:{lineno}: 声部名 {name!r} 不合法")
                continue
            allowed = {"patch", "gain", "pan", "reverb", "delay", "chord", "subdiv", "octave", "humanize", "duck", "chorus", "accent"}
            mods = _kv_pairs(fields[2:], lineno, path, allowed)
            if "patch" not in mods:
                problems.append(f"{path.name}:{lineno}: instrument {name} 缺少必填参数 patch")
                continue
            chord = mods.get("chord", "hold").lower()
            if chord not in _CHORD_MODES:
                problems.append(f"{path.name}:{lineno}: chord={chord} 未知（可用 {', '.join(_CHORD_MODES)}）")
                continue
            inst = Instrument(
                name=name,
                patch=mods["patch"],
                gain=_as_float(mods.get("gain", "1"), "gain", lineno, path),
                pan=_as_float(mods.get("pan", "0"), "pan", lineno, path),
                reverb=_as_float(mods.get("reverb", "0"), "reverb", lineno, path),
                delay=_as_float(mods.get("delay", "0"), "delay", lineno, path),
                chord=chord,
                subdiv=_as_float(mods.get("subdiv", "0.5"), "subdiv", lineno, path),
                octave=_as_int(mods.get("octave", "0"), "octave", lineno, path),
                humanize=_as_float(mods["humanize"], "humanize", lineno, path) if "humanize" in mods else None,
                duck=mods.get("duck") or None,
                chorus=_as_float(mods.get("chorus", "0"), "chorus", lineno, path),
                accent=_parse_accent(mods.get("accent", ""), lineno, path),
                line=lineno,
            )
            if inst.subdiv <= 0:
                problems.append(f"{path.name}:{lineno}: subdiv 必须 > 0")
                continue
            if name in score.instruments:
                problems.append(f"{path.name}:{lineno}: 声部 {name} 重复定义（后定义覆盖前者）")
            score.instruments[name] = inst
            continue

        if lowered.startswith("tempo "):
            score.tempo = _as_float(stripped.split()[1], "tempo", lineno, path)
            continue
        if lowered.startswith("meter "):
            val = stripped.split()[1]
            if val != "4/4":
                problems.append(f"{path.name}:{lineno}: v1 只支持 meter 4/4，收到 {val!r}")
            score.beats_per_bar = 4.0
            continue
        if lowered.startswith("key "):
            token = stripped.split()[1]
            try:
                score.tonic_pc, score.scale = theory.parse_key(token)
                score.key_token = token
            except ValueError as exc:
                problems.append(f"{path.name}:{lineno}: {exc}")
            continue
        if lowered.startswith("swing "):
            score.swing = _as_float(stripped.split()[1], "swing", lineno, path)
            if not (0.0 <= score.swing <= 0.4):
                problems.append(f"{path.name}:{lineno}: swing 须在 0..0.4")
            continue
        if lowered.startswith("humanize "):
            score.humanize = _as_float(stripped.split()[1], "humanize", lineno, path)
            if not (0.0 <= score.humanize <= 0.05):
                problems.append(f"{path.name}:{lineno}: 全局 humanize 须在 0..0.05 秒")
            continue
        if lowered.startswith("seed "):
            score.seed = _as_int(stripped.split()[1], "seed", lineno, path)
            continue

        problems.append(f"{path.name}:{lineno}: 无法识别的行 {stripped!r}")

    return score, problems


def parse_file(path: str | Path) -> tuple[Score, list[str]]:
    p = Path(path)
    return parse_text(p.read_text(encoding="utf-8"), p)
