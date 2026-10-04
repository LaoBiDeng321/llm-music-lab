"""钢琴卷帘可视化：把展开后的音符画成 PNG，用于人工检查编曲密度与音区分配。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import effects, theory
from .render import Hit, expand
from .score import Score

BG = (16, 18, 24)
PANEL = (24, 27, 35)
GRID = (38, 42, 54)
GRID_STRONG = (60, 66, 84)
TEXT = (198, 206, 220)
TEXT_DIM = (120, 128, 145)
BAR_LINE = (86, 94, 116)

PALETTE = [
    (255, 138, 128),
    (129, 199, 255),
    (149, 225, 190),
    (255, 205, 120),
    (198, 160, 246),
    (255, 168, 220),
    (140, 220, 240),
    (196, 220, 140),
]

DRUM_ORDER = ("kick", "snare", "clap", "hat", "ohat", "ride", "crash", "rim", "tom", "shaker", "riser", "downlift")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_piano_roll(
    score: Score,
    hits: list[Hit] | None = None,
    wav_path: str | Path | None = None,
    out_path: str | Path = "out/piano-roll.png",
    px_per_bar: int = 30,
    row_h: int = 9,
) -> Path:
    hits = hits if hits is not None else expand(score)
    total_bars = max(score.total_bars, 1)
    pitched = [h for h in hits if not h.is_drum and h.midis]
    drums = [h for h in hits if h.is_drum]

    if pitched:
        lo = min(min(h.midis) for h in pitched) - 2
        hi = max(max(h.midis) for h in pitched) + 2
    else:
        lo, hi = 48, 72
    lo = max(lo, 16)
    hi = min(hi, 110)

    margin_left = 74
    margin_right = 34
    header_h = 86
    drum_h = 118
    wave_h = 96
    grid_w = total_bars * px_per_bar
    grid_h = (hi - lo + 1) * row_h
    width = margin_left + grid_w + margin_right
    height = header_h + grid_h + drum_h + wave_h + 26

    img = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(img)
    f_small = _font(13)
    f_mid = _font(15)
    f_big = _font(19)

    grid_top = header_h
    grid_bottom = grid_top + grid_h

    # ---- 行背景（黑键行更暗）
    for m in range(lo, hi + 1):
        y = grid_bottom - (m - lo + 1) * row_h
        if (m % 12) in (1, 3, 6, 8, 10):
            draw.rectangle([margin_left, y, margin_left + grid_w, y + row_h], fill=(20, 22, 29))
        if m % 12 == 0:
            draw.line([margin_left, y + row_h, margin_left + grid_w, y + row_h], fill=GRID_STRONG)
            draw.text((8, y + row_h // 2 - 7), theory.midi_to_name(m), fill=TEXT_DIM, font=f_small)

    # ---- 小节线 + 拍线
    for bar in range(total_bars + 1):
        x = margin_left + bar * px_per_bar
        draw.line([x, grid_top, x, grid_bottom + drum_h], fill=BAR_LINE if bar % 4 == 0 else GRID, width=2 if bar % 4 == 0 else 1)
        for beat in (1, 2, 3):
            bx = x + beat * px_per_bar / 4
            if bx < margin_left + grid_w:
                draw.line([bx, grid_top, bx, grid_bottom], fill=(28, 31, 40))

    # ---- 段落带
    draw.rectangle([margin_left, 8, margin_left + grid_w, 30], fill=PANEL)
    for s in score.sections:
        x0 = margin_left + (s.start - 1) * px_per_bar
        x1 = margin_left + min(s.end, total_bars) * px_per_bar
        draw.rectangle([x0 + 1, 9, x1 - 1, 29], outline=(70, 78, 98))
        draw.text((x0 + 6, 11), f"{s.name} {s.start}-{s.end}", fill=TEXT, font=f_small)

    # ---- 音符
    track_color = {name: PALETTE[i % len(PALETTE)] for i, name in enumerate(sorted(score.instruments))}
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    for h in pitched:
        color = track_color.get(h.track, (200, 200, 200))
        x0 = margin_left + (h.start_beat / score.beats_per_bar) * px_per_bar
        w = max(h.dur_beats / score.beats_per_bar * px_per_bar - 1.0, 1.6)
        for m in h.midis:
            if m < lo or m > hi:
                continue
            y = grid_bottom - (m - lo + 1) * row_h
            alpha = int(120 + 130 * min(max(h.vel, 0.0), 1.0))
            odraw.rectangle([x0, y + 1, x0 + w, y + row_h - 1], fill=color + (alpha,), outline=color + (255,))
    img = Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(img)

    # ---- 鼓轨
    drum_top = grid_bottom + 6
    voices = [v for v in DRUM_ORDER if any(d.token == v for d in drums)]
    if voices:
        lane_h = max(drum_h // max(len(voices), 1), 8)
        for i, voice in enumerate(voices):
            y = drum_top + i * lane_h
            draw.line([margin_left, y, margin_left + grid_w, y], fill=GRID)
            draw.text((8, y + 1), voice, fill=TEXT_DIM, font=f_small)
        for d in drums:
            if d.token not in voices:
                continue
            x = margin_left + (d.start_beat / score.beats_per_bar) * px_per_bar
            y = drum_top + voices.index(d.token) * lane_h
            r = max(int(lane_h * 0.34 * (0.6 + 0.4 * d.vel)), 3)
            color = (255, 170, 90) if d.token in ("kick", "snare", "clap") else (120, 190, 230)
            draw.ellipse([x - r, y + lane_h / 2 - r, x + r, y + lane_h / 2 + r], fill=color)

    # ---- 波形
    wave_top = drum_top + len(voices) * max(drum_h // max(len(voices), 1), 8) + 12
    wave_top = min(wave_top, height - wave_h - 8)
    draw.rectangle([margin_left, wave_top, margin_left + grid_w, wave_top + wave_h], fill=PANEL)
    if wav_path is not None and Path(wav_path).exists():
        from .analyze import read_wav

        audio, sr = read_wav(wav_path)
        mono = audio if audio.ndim == 1 else audio.mean(axis=1)
        step = max(int(mono.shape[0] / grid_w), 1)
        for i in range(grid_w):
            seg = mono[i * step : (i + 1) * step]
            if seg.size == 0:
                continue
            top = wave_top + wave_h / 2 - float(np.max(seg)) * wave_h * 0.46
            bottom = wave_top + wave_h / 2 - float(np.min(seg)) * wave_h * 0.46
            draw.line([margin_left + i, top, margin_left + i, bottom], fill=(96, 176, 220))
        # 段落均方根曲线
        spb = 60.0 / score.tempo
        for s in score.sections:
            a = int((s.start - 1) * score.beats_per_bar * spb * sr)
            b = int(s.end * score.beats_per_bar * spb * sr)
            seg = mono[a:b]
            if seg.size == 0:
                continue
            rms = float(np.sqrt(np.mean(seg**2)))
            x0 = margin_left + (s.start - 1) * px_per_bar
            x1 = margin_left + min(s.end, total_bars) * px_per_bar
            y = wave_top + wave_h - 4 - rms * wave_h * 1.6
            draw.line([x0, y, x1, y], fill=(255, 190, 110), width=2)

    # ---- 标题与图例
    draw.text((margin_left, 34), f"{score.path.stem}   {score.tempo:.0f} BPM   调性 {score.key_token or '未声明'}", fill=TEXT, font=f_big)
    legend_x = margin_left
    for name, color in track_color.items():
        inst = score.instruments[name]
        label = f"{name} · {inst.patch}"
        draw.rectangle([legend_x, 60, legend_x + 12, 72], fill=color)
        draw.text((legend_x + 17, 58), label, fill=TEXT, font=f_mid)
        legend_x += 17 + int(draw.textlength(label, font=f_mid)) + 22

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    return out
