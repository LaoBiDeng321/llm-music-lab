"""tmskit 自检。运行： python -m unittest discover -s tests -t . -v

覆盖三件事：
1. 乐理与解析的正确性（含记号歧义的边界情况）；
2. 校验器该报的错要报、不该报的别报；
3. 端到端不变量：pattern 循环、和弦展开、渲染确定性、FLAC 无损往返。
"""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from tmskit import analyze, flac, render, theory
from tmskit.score import parse_text
from tmskit.validate import validate

REPO = Path(__file__).resolve().parent.parent

MINI_SCORE = """
tempo 100
meter 4/4
key Am
seed 7

instrument pad  patch=saw-pad   gain=0.8 reverb=0.5
instrument keys patch=epiano    gain=0.9 chord=arp-up subdiv=0.5
instrument bass patch=sub-bass  gain=1.0 chord=root8 subdiv=0.5 octave=-1 duck=kick
instrument drum patch=drumkit   gain=1.0

pattern pad.prog {
    Am 0 4
    F 4 4
    C 8 4
    G 12 4
}

pattern keys.arp {
    Am 0 4
    F 4 4
    C 8 4
    G 12 4
}

pattern drum.one {
    grid 16
    kick  X . . . . . X . . . X . . . . .
    hat   x . x . x . x . x . x . x . x .
}

pattern bass.prog {
    Am 0 4
    F 4 4
    C 8 4
    G 12 4
}

arrange
    bars 1-4  pad.prog, keys.arp, bass.prog, drum.one
"""


class TestTheory(unittest.TestCase):
    def test_pitch(self):
        self.assertEqual(theory.parse_pitch("C4"), 60)
        self.assertEqual(theory.parse_pitch("A4"), 69)
        self.assertEqual(theory.parse_pitch("C#5"), 73)
        self.assertEqual(theory.parse_pitch("Bb3"), 58)
        self.assertAlmostEqual(theory.midi_to_hz(69), 440.0)

    def test_octave_7_is_a_chord_not_a_note(self):
        """E7 必须解析成属七和弦；A5/C5 这类必须仍是音名。"""
        self.assertFalse(theory.is_pitch("E7"))
        self.assertTrue(theory.is_chord("E7"))
        self.assertTrue(theory.is_pitch("A5"))
        self.assertFalse(theory.is_chord("A5"))
        root, tones, _ = theory.parse_chord("E7")
        self.assertEqual(root, 52)
        self.assertEqual(set(tones), {0, 4, 7, 10})

    def test_chord_voicing_and_slash(self):
        self.assertEqual(theory.chord_voicing("Am"), [57, 60, 64])
        self.assertEqual(theory.chord_voicing("Am", octave=1), [69, 72, 76])
        voicing = theory.chord_voicing("G/B")
        self.assertEqual(voicing[0], 47)  # B1 作为低音
        self.assertIn(55, voicing)  # G3

    def test_key(self):
        tonic, scale = theory.parse_key("Am")
        self.assertEqual(tonic, 9)
        self.assertTrue(theory.in_scale(57, tonic, scale))
        self.assertFalse(theory.in_scale(56, tonic, scale))  # G# 不在自然小调里


class TestParserAndValidator(unittest.TestCase):
    def test_mini_score_parses_clean(self):
        score, problems = parse_text(MINI_SCORE)
        self.assertEqual(problems, [])
        errors, warnings = validate(score, problems)
        self.assertEqual(errors, [], f"不该有错误：{errors}")
        self.assertEqual(len(score.instruments), 4)
        self.assertEqual(score.total_bars, 4)

    def test_unknown_patch_is_an_error(self):
        text = MINI_SCORE.replace("patch=saw-pad", "patch=no-such-patch")
        score, problems = parse_text(text)
        errors, _ = validate(score, problems)
        self.assertTrue(any("未知 patch" in e for e in errors), errors)

    def test_undefined_pattern_is_an_error(self):
        text = MINI_SCORE.replace("pad.prog, ", "pad.nope, ")
        score, problems = parse_text(text)
        errors, _ = validate(score, problems)
        self.assertTrue(any("未定义" in e for e in errors), errors)

    def test_bad_argument_reports_line_number(self):
        text = MINI_SCORE.replace("chord=arp-up", "chord=arp-sideways")
        score, problems = parse_text(text)
        errors, _ = validate(score, problems)
        self.assertTrue(any("arp-sideways" in e for e in errors), errors)

    def test_grid_symbol_count_is_checked(self):
        text = MINI_SCORE.replace("kick  X . . . . . X . . . X . . . . .", "kick  X . . X")
        score, problems = parse_text(text)
        self.assertTrue(any("符号" in p for p in problems), problems)

    def test_out_of_scale_warns_but_leading_tone_does_not(self):
        text = MINI_SCORE.replace("    Am 0 4\n    F 4 4", "    D#5 0 4\n    F 4 4", 1)
        score, problems = parse_text(text)
        _, warnings = validate(score, problems)
        self.assertTrue(any("调内" in w for w in warnings), warnings)

        harmonic = MINI_SCORE.replace("    Am 0 4\n    F 4 4", "    G#4 0 4\n    F 4 4", 1)
        score2, problems2 = parse_text(harmonic)
        _, warnings2 = validate(score2, problems2)
        self.assertFalse(any("G#4" in w and "调内" in w for w in warnings2), warnings2)

    def test_ship_score_is_clean(self):
        """仓库里真正的作品必须始终通过校验。"""
        path = REPO / "score" / "tide-end.tms"
        if not path.exists():
            self.skipTest("作品乐谱不存在")
        from tmskit.score import parse_file

        score, problems = parse_file(path)
        errors, warnings = validate(score, problems)
        self.assertEqual(errors, [], f"作品乐谱有错误：{errors}")
        self.assertEqual(warnings, [], f"作品乐谱有警告：{warnings}")


class TestExpand(unittest.TestCase):
    def test_pattern_loops_to_fill_range(self):
        """1 小节的鼓 pattern 放进 4 小节范围 → 连打 4 遍，且不能有拍位漂移。

        hat 行只写到第 14 格（原始长度 3.75 拍）；循环长度必须向上取整到 4 拍，
        否则每小节会漂 0.25 拍。这个用例就是为这件事写的。
        """
        score, _ = parse_text(MINI_SCORE)
        hits = render.expand(score)
        kicks = sorted(h.start_beat for h in hits if h.token == "kick")
        self.assertEqual(kicks, [0.0, 1.5, 2.5, 4.0, 5.5, 6.5, 8.0, 9.5, 10.5, 12.0, 13.5, 14.5])
        hats = sorted(h.start_beat for h in hits if h.token == "hat" and h.start_beat >= 4)
        self.assertAlmostEqual(hats[0], 4.0, places=6)

    def test_loop_length_snaps_to_whole_bars(self):
        score, _ = parse_text(MINI_SCORE)
        pattern = score.patterns[("drum", "one")]
        self.assertAlmostEqual(pattern.length_beats, 3.75, places=6)
        self.assertAlmostEqual(pattern.loop_length(score.beats_per_bar), 4.0, places=6)

    def test_root8_bass_takes_root_and_fifth(self):
        score, _ = parse_text(MINI_SCORE)
        hits = [h for h in render.expand(score) if h.track == "bass"]
        self.assertEqual(len(hits), 8 * 4)  # 每小节 8 个八分脉冲
        # A 小调根音 A2=45，五度 E3=52（octave=-1 作用于第 3 八度基线）
        first_bar = [h.midis[0] for h in hits if h.start_beat < 4]
        self.assertEqual(first_bar[:4], [45, 45, 45, 52])

    def test_arp_up_cycles_chord_tones(self):
        score, _ = parse_text(MINI_SCORE)
        hits = [h for h in render.expand(score) if h.track == "keys" and h.start_beat < 4]
        self.assertEqual([h.midis[0] for h in hits], [57, 60, 64, 57, 60, 64, 57, 60])

    def test_placement_gain_is_applied(self):
        text = MINI_SCORE.replace("pad.prog,", "pad.prog gain=0.5,")
        score, _ = parse_text(text)
        pads = [h for h in render.expand(score) if h.track == "pad"]
        self.assertTrue(pads)
        self.assertTrue(all(abs(h.gain - 0.5) < 1e-9 for h in pads))

    def test_swing_delays_offbeat_eighths(self):
        text = MINI_SCORE.replace("seed 7", "seed 7\nswing 0.3")
        score, _ = parse_text(text)
        hats = sorted(h.start_beat for h in render.expand(score) if h.token == "hat" and h.start_beat < 4)
        self.assertAlmostEqual(hats[1], 0.5 + 0.3 * 0.25, places=6)


class TestRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        score, _ = parse_text(MINI_SCORE)
        cls.result = render.render_score(score, verbose=False)

    def test_render_is_deterministic(self):
        score, _ = parse_text(MINI_SCORE)
        other = render.render_score(score, verbose=False)
        self.assertTrue(np.array_equal(self.result.audio, other.audio))

    def test_master_gates(self):
        audio = self.result.audio
        self.assertFalse(np.any(np.abs(audio) >= 1.0), "出现削波样本")
        peak = analyze.analyze(audio, self.result.sr)
        self.assertLessEqual(peak["true_peak_db"], -0.999, "真峰值超过 −1 dBTP")
        self.assertTrue(-18.0 <= peak["rms_db"] <= -12.0, f"整体 RMS 越界：{peak['rms_db']:.2f}")
        self.assertLess(abs(peak["dc_offset"]), 1e-3)

    def test_wav_roundtrip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "t.wav"
            render.write_wav(path, self.result.audio, self.result.sr)
            pcm, sr = flac.read_wav_pcm(path)
            self.assertEqual(sr, 44100)
            self.assertEqual(pcm.shape[0], self.result.audio.shape[0])


class TestFlac(unittest.TestCase):
    def test_lossless_roundtrip(self):
        rng = np.random.default_rng(0)
        cases = {
            "silence": np.zeros((5000, 2), dtype=np.int16),
            "dc": np.full((5000, 2), 1234, dtype=np.int16),
            "square": np.tile(np.array([[30000, -30000]], dtype=np.int16), (5000, 1)),
            "noise": rng.integers(-32768, 32767, size=(5000, 2), dtype=np.int16),
            "block_boundary": rng.integers(-32768, 32767, size=(flac.BLOCK_SIZE + 1, 2), dtype=np.int16),
            "mono": rng.integers(-32768, 32767, size=(9000, 1), dtype=np.int16),
            "sine": (np.sin(2 * np.pi * 440 * np.arange(20000) / 44100) * 30000).astype(np.int16).reshape(-1, 1),
        }
        for name, pcm in cases.items():
            with self.subTest(name=name):
                decoded, sr = flac.decode(flac.encode(pcm, 44100))
                self.assertEqual(sr, 44100)
                self.assertTrue(np.array_equal(np.asarray(decoded).reshape(pcm.shape), pcm))

    def test_header_is_flac(self):
        data = flac.encode(np.zeros((100, 2), dtype=np.int16), 44100)
        self.assertEqual(data[:4], b"fLaC")
        self.assertEqual(data[4], 0x80)  # 最后一个元数据块 + STREAMINFO

    def test_md5_is_checked(self):
        pcm = np.zeros((100, 2), dtype=np.int16)
        data = bytearray(flac.encode(pcm, 44100))
        data[-1] ^= 0xFF  # 破坏最后一个字节
        with self.assertRaises(Exception):
            flac.decode(bytes(data))


class TestVisualizeSmoke(unittest.TestCase):
    def test_roll_renders_a_png(self):
        import tempfile

        from tmskit.visualize import draw_piano_roll

        score, _ = parse_text(MINI_SCORE)
        with tempfile.TemporaryDirectory() as tmp:
            out = draw_piano_roll(score, out_path=Path(tmp) / "roll.png", px_per_bar=20)
            self.assertTrue(out.exists())
            self.assertGreater(out.stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
