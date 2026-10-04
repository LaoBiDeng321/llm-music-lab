"""音名 / 和弦 / 音阶 / 频率：全部为纯函数，不依赖项目其他模块。"""

from __future__ import annotations

import re
from math import log2

# ---------------------------------------------------------------- 音名

_LETTERS = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACCIDENTAL = {"": 0, "#": 1, "s": 1, "b": -1, "f": -1}

_PITCH_RE = re.compile(r"^([A-Ga-g])([#bsf]?)(-?\d+)$")
_CHORD_RE = re.compile(r"^([A-Ga-g])([#b]?)([^/]*)(?:/([A-Ga-g])([#b]?))?$")

DRUM_VOICES = (
    "kick",
    "snare",
    "hat",
    "ohat",
    "clap",
    "ride",
    "crash",
    "rim",
    "tom",
    "shaker",
    "riser",
    "downlift",
)

# 音名的最高八度：再高就按和弦记号解析（见 is_pitch）
MAX_NOTE_OCTAVE = 6

# 和弦后缀 → 相对根音的半音集合
_CHORD_TONES = {
    "": (0, 4, 7),
    "M": (0, 4, 7),
    "maj": (0, 4, 7),
    "m": (0, 3, 7),
    "min": (0, 3, 7),
    "5": (0, 7),
    "6": (0, 4, 7, 9),
    "m6": (0, 3, 7, 9),
    "7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11),
    "M7": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10),
    "mmaj7": (0, 3, 7, 11),
    "m7b5": (0, 3, 6, 10),
    "dim": (0, 3, 6),
    "dim7": (0, 3, 6, 9),
    "aug": (0, 4, 8),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "add9": (0, 4, 7, 14),
    "madd9": (0, 3, 7, 14),
    "9": (0, 4, 7, 10, 14),
    "maj9": (0, 4, 7, 11, 14),
    "m9": (0, 3, 7, 10, 14),
}

# 音阶：相对主音的半音集合
SCALES = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "harmonicminor": (0, 2, 3, 5, 7, 8, 11),
}

NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def midi_to_hz(midi: float) -> float:
    """MIDI 音高号 → 频率（A4 = 69 = 440 Hz）。"""
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def midi_to_name(midi: int) -> str:
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def is_pitch(token: str) -> bool:
    """音名判定：必须匹配「字母+可选升降号+八度」，且八度 ≤ 6。

    八度 ≥ 7 一律让位给和弦记号（`E7` 是属七和弦，不是第 7 八度的 E 音）。
    这解决了 `E7`/`C7` 这类「音名 vs 和弦」的歧义；和弦表因此不提供 "5"（强力和弦），
    因为 `A5`/`C5` 被约定为音名。
    """
    m = _PITCH_RE.match(token)
    if not m:
        return False
    return int(m.group(3)) <= MAX_NOTE_OCTAVE


def parse_pitch(token: str) -> int:
    """`C#4` / `Bb3` → MIDI 号。非法记号抛 ValueError。"""
    m = _PITCH_RE.match(token)
    if not m:
        raise ValueError(f"无法解析的音名: {token!r}")
    letter, accidental, octave = m.groups()
    return (int(octave) + 1) * 12 + _LETTERS[letter.upper()] + _ACCIDENTAL[accidental.lower()]


def is_chord(token: str) -> bool:
    if is_pitch(token) or token in DRUM_VOICES:
        return False
    m = _CHORD_RE.match(token)
    if not m:
        return False
    return m.group(3) in _CHORD_TONES


def parse_chord(token: str) -> tuple[int, tuple[int, ...], int | None]:
    """和弦记号 → (根音 MIDI 音级, 相对半音集合, 斜杠低音 pitch class 或 None)。

    根音以第 3 八度落位（MIDI 48..59），供渲染层加八度偏移。
    """
    m = _CHORD_RE.match(token)
    if not m:
        raise ValueError(f"无法解析的和弦记号: {token!r}")
    letter, accidental, suffix, bass_letter, bass_acc = m.groups()
    if suffix not in _CHORD_TONES:
        raise ValueError(f"未知和弦后缀: {token!r}")
    root_pc = _LETTERS[letter.upper()] + _ACCIDENTAL[accidental or ""]
    root_midi = 48 + (root_pc % 12)
    bass = None
    if bass_letter:
        bass = (_LETTERS[bass_letter.upper()] + _ACCIDENTAL[bass_acc or ""]) % 12
    return root_midi, _CHORD_TONES[suffix], bass


def parse_key(token: str) -> tuple[int, tuple[int, ...]]:
    """`Am` / `C` / `Dm` → (主音 pitch class, 音阶半音集合)。"""
    m = re.match(r"^([A-Ga-g])([#b]?)(m|min|maj|major|dorian|mixolydian|harmonicminor)?$", token)
    if not m:
        raise ValueError(f"无法解析的调性: {token!r}")
    letter, acc, mode = m.groups()
    pc = (_LETTERS[letter.upper()] + _ACCIDENTAL[acc or ""]) % 12
    if mode in (None, "maj", "major"):
        scale = SCALES["major"]
    elif mode in ("m", "min"):
        scale = SCALES["minor"]
    else:
        scale = SCALES[mode]
    return pc, scale


def in_scale(midi: int, tonic_pc: int, scale: tuple[int, ...]) -> bool:
    return (midi - tonic_pc) % 12 in scale


def chord_pitch_classes(token: str) -> set[int]:
    root, tones, bass = parse_chord(token)
    pcs = {(root + t) % 12 for t in tones}
    if bass is not None:
        pcs.add(bass)
    return pcs


def chord_voicing(token: str, octave: int = 0) -> list[int]:
    """把和弦摊成一组具体 MIDI 音高（默认第 3 八度起，`octave` 叠加八度）。"""
    root, tones, bass = parse_chord(token)
    shift = 12 * octave
    notes = [root + t + shift for t in tones]
    if bass is not None:
        low = 36 + (bass % 12)  # 第 2 八度
        notes = [low + shift] + notes
    return notes


def approx_register(midi: int) -> str:
    """粗略音区名，用于校验器提示。"""
    if midi < 36:
        return "超低"
    if midi < 48:
        return "低"
    if midi < 60:
        return "中低"
    if midi < 72:
        return "中"
    if midi < 84:
        return "中高"
    if midi < 96:
        return "高"
    return "超高"


def cents_between(a_hz: float, b_hz: float) -> float:
    return 1200.0 * log2(a_hz / b_hz)
