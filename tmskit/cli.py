"""命令行入口：python -m tmskit <命令>。"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

from . import analyze as analyze_mod
from . import flac as flac_mod
from . import render as render_mod
from . import theory
from . import visualize as visualize_mod
from . import voices
from .score import parse_file
from .validate import validate as validate_score

OK = "\033[92m✓\033[0m"
BAD = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"
DIM = "\033[90m"


def _print_issues(errors: list[str], warnings: list[str], show_warnings: bool = True) -> None:
    for e in errors:
        print(f"  {BAD} {e}")
    if show_warnings:
        for w in warnings:
            print(f"  {WARN} {w}")
    print(f"  {DIM}错误 {len(errors)} 条，警告 {len(warnings)} 条\033[0m")


def cmd_check(args) -> int:
    score, problems = parse_file(args.score)
    errors, warnings = validate_score(score, problems)
    print(f"检查 {args.score}")
    _print_issues(errors, warnings, not args.quiet)
    if not errors:
        kinds = {}
        for p in score.placements:
            kinds[p.track] = kinds.get(p.track, 0) + 1
        print(
            f"  {DIM}{score.total_bars} 小节 · {score.tempo:g} BPM · "
            f"{len(score.instruments)} 声部 · {len(score.patterns)} 个 pattern · {len(score.placements)} 条编排\033[0m"
        )
    return 1 if errors else 0


def cmd_render(args) -> int:
    score, problems = parse_file(args.score)
    errors, warnings = validate_score(score, problems)
    if errors:
        print(f"渲染中止：乐谱有 {len(errors)} 条错误")
        _print_issues(errors, warnings, False)
        return 1
    if warnings and not args.quiet:
        _print_issues([], warnings, True)
    t0 = time.time()
    print(f"渲染 {args.score} → {args.out}")
    result = render_mod.render_score(score, verbose=not args.quiet)
    render_mod.write_wav(args.out, result.audio, result.sr)
    dt = time.time() - t0
    print(f"  {OK} 完成：{len(result.hits)} 次发声，{result.duration:.1f} s 音频，耗时 {dt:.1f} s")
    if result.section_rms:
        _print_section_matrix(result)
    print(
        f"  {DIM}限幅前峰值 {result.pre_limit_peak_db:+.1f} dBFS "
        f"→ 限幅增益衰减约 {max(result.pre_limit_peak_db + 1.0, 0.0):.1f} dB（>3 dB 说明配平偏热）\033[0m"
    )
    return 0


def _print_section_matrix(result) -> None:
    """分轨 × 分段 RMS 表：用来校准各声部在各段的相对关系。"""
    score = result.score
    names = list(result.section_rms)
    header = "  " + "声部".ljust(9) + "".join(s.name.rjust(9) for s in score.sections)
    print(f"  {DIM}分轨 × 分段 RMS（dBFS，− 表示该段基本没这个声部）\033[0m")
    print(header)
    for name in names:
        row = "".join(
            ("     sil" if v <= -119 else f"{v:9.1f}") for v in result.section_rms[name]
        )
        print("  " + name.ljust(9) + row)


def cmd_analyze(args) -> int:
    score = None
    if args.score:
        score, _ = parse_file(args.score)
    info, lines = analyze_mod.report(args.wav, score)
    print(f"分析 {args.wav}")
    for line in lines:
        print("  " + line)
    fails = (
        info["clipped_samples"] > 0
        or not (-18.0 <= info["rms_db"] <= -12.0)
        or info["true_peak_db"] > -1.0
    )
    return 1 if fails else 0


def cmd_roll(args) -> int:
    score, problems = parse_file(args.score)
    errors, _ = validate_score(score, problems)
    if errors:
        print("乐谱有错误，无法绘制")
        _print_issues(errors, [], False)
        return 1
    out = visualize_mod.draw_piano_roll(score, wav_path=args.wav, out_path=args.out, px_per_bar=args.bar_width)
    print(f"  {OK} 钢琴卷帘 → {out}")
    return 0


def cmd_flac(args) -> int:
    pcm, sr = flac_mod.read_wav_pcm(args.wav)
    t0 = time.time()
    data = flac_mod.encode(pcm, sr)
    enc = time.time() - t0
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(data)
    raw = pcm.nbytes
    print(f"  {OK} {args.wav} → {args.out}")
    print(
        f"  {DIM}{raw / 1048576:.1f} MB → {len(data) / 1048576:.1f} MB"
        f"（{100.0 * len(data) / raw:.1f}%），无损；编码耗时 {enc:.1f}s\033[0m"
    )
    if args.verify:
        t0 = time.time()
        dec, sr2 = flac_mod.decode(data)
        ok = np.array_equal(dec.reshape(pcm.shape), pcm)
        print(f"  {'✓' if ok else '✗'} 解码自检：{'与原 PCM 逐字节一致' if ok else '不一致'}（耗时 {time.time() - t0:.1f}s）")
        if not ok:
            return 1
    return 0


def cmd_unflac(args) -> int:
    dec, sr = flac_mod.decode(Path(args.flac).read_bytes())
    flac_mod.write_wav_pcm(args.out, dec, sr)
    print(f"  {OK} {args.flac} → {args.out}（{sr} Hz，{dec.shape[0] / sr:.2f}s）")
    return 0


def cmd_patches(args) -> int:
    print(f"{'patch':<14}{'类型':<8}{'推荐音域':<18}说明")
    for name, kind, rng, desc in voices.patch_table():
        print(f"{name:<14}{kind:<8}{rng:<18}{desc}")
    print(f"\n鼓件: {', '.join(theory.DRUM_VOICES)}")
    return 0


def cmd_all(args) -> int:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.score).stem
    wav = outdir / f"{stem}.wav"
    png = outdir / f"{stem}-piano-roll.png"
    rc = cmd_check(argparse.Namespace(score=args.score, quiet=True))
    if rc:
        return rc
    rc = cmd_render(argparse.Namespace(score=args.score, out=str(wav), quiet=False))
    if rc:
        return rc
    cmd_analyze(argparse.Namespace(wav=str(wav), score=args.score))
    cmd_roll(argparse.Namespace(score=args.score, wav=str(wav), out=str(png), bar_width=args.bar_width))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tmskit", description="文本乐谱（TMS）→ 电子音渲染工具链")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="静态校验乐谱")
    c.add_argument("score")
    c.add_argument("-q", "--quiet", action="store_true", help="只看错误")
    c.set_defaults(func=cmd_check)

    r = sub.add_parser("render", help="渲染成 WAV")
    r.add_argument("score")
    r.add_argument("-o", "--out", default="out/piece.wav")
    r.add_argument("-q", "--quiet", action="store_true")
    r.set_defaults(func=cmd_render)

    a = sub.add_parser("analyze", help="分析 WAV（电平/动态/频段）")
    a.add_argument("wav")
    a.add_argument("-s", "--score", help="同时给出乐谱以按段落统计")
    a.set_defaults(func=cmd_analyze)

    v = sub.add_parser("roll", help="画钢琴卷帘 PNG")
    v.add_argument("score")
    v.add_argument("-o", "--out", default="out/piano-roll.png")
    v.add_argument("-w", "--wav", default=None, help="叠加波形")
    v.add_argument("-b", "--bar-width", type=int, default=30)
    v.set_defaults(func=cmd_roll)

    pa = sub.add_parser("patches", help="列出全部音色与推荐音域")
    pa.set_defaults(func=cmd_patches)

    fl = sub.add_parser("flac", help="把 WAV 无损压缩成 FLAC（自带编解码，不依赖外部程序）")
    fl.add_argument("wav")
    fl.add_argument("-o", "--out", default="out/piece.flac")
    fl.add_argument("--verify", action="store_true", help="编码后立刻解码回比对")
    fl.set_defaults(func=cmd_flac)

    uf = sub.add_parser("unflac", help="把 FLAC 解回 WAV")
    uf.add_argument("flac")
    uf.add_argument("-o", "--out", default="out/decoded.wav")
    uf.set_defaults(func=cmd_unflac)

    al = sub.add_parser("all", help="check + render + analyze + roll 一条龙")
    al.add_argument("score")
    al.add_argument("-o", "--outdir", default="out")
    al.add_argument("-b", "--bar-width", type=int, default=30)
    al.set_defaults(func=cmd_all)
    return p


def main(argv: list[str] | None = None) -> int:
    try:  # Windows 控制台默认 GBK，强制 UTF-8 输出中文
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

