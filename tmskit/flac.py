"""纯 Python + numpy 的 FLAC 编码器/解码器（子集实现）。

为什么要自己写：项目硬约束禁止第三方依赖与外部程序，而 32 MB 的 WAV 不方便分享。
这里实现 FLAC 的一个规范子集，无损、可被通用播放器解码：

- 元数据：只有 STREAMINFO（含原始 PCM 的 MD5）。
- 帧：固定块长（4096），2 声道独立（channel assignment = 0001），16 bit，44100 Hz。
- 子帧：固定预测器（order 0..4），Rice 编码（4/5 bit 参数），无 wasted bits，无 LPC。
- 每帧写 CRC-8（帧头）与 CRC-16（整帧），最后一块可以短于 4096。

编码器会为每个块/声道在 5 种固定阶数 × 3 种分区阶数里选比特数最少的一种；
解码器只用于自检（往返必须与原始 PCM 逐字节一致），不当播放用。
"""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np

BLOCK_SIZE = 4096
MAX_RICE_K = 14  # 用 4-bit 参数编码（00），避免触发 escape code
FIXED_COEFS = {
    0: (),
    1: (1,),
    2: (2, -1),
    3: (3, -3, 1),
    4: (4, -6, 4, -1),
}
PARTITION_ORDERS = (0, 1, 2, 3, 4)


# ---------------------------------------------------------------- 位操作


def pack_bits(fields: list[tuple[int, int]]) -> bytes:
    """把 (value, nbits) 序列按 MSB-first 打成字节；总位数须为 8 的倍数。"""
    acc = 0
    nbits = 0
    for value, width in fields:
        if value < 0 or value >= (1 << width):
            raise ValueError(f"值 {value} 无法用 {width} 位表示")
        acc = (acc << width) | value
        nbits += width
    if nbits % 8:
        raise ValueError(f"位字段总长 {nbits} 不是 8 的倍数")
    return acc.to_bytes(nbits // 8, "big")


def fields_to_bits(fields: list[tuple[int, int]]) -> np.ndarray:
    """把 (value, nbits) 序列打成 0/1 位数组，允许非字节对齐。"""
    total = sum(width for _, width in fields)
    bits = np.zeros(total, dtype=np.uint8)
    pos = 0
    for value, width in fields:
        if width <= 0:
            continue
        raw = int(value).to_bytes((width + 7) // 8, "big", signed=False)
        bits[pos : pos + width] = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[-width:]
        pos += width
    return bits


def crc8(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def crc16(data: bytes) -> int:
    crc = 0
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x8005) & 0xFFFF if crc & 0x8000 else (crc << 1) & 0xFFFF
    return crc


def utf8_uint(value: int) -> bytes:
    """FLAC 帧号用的 UTF-8 变长整数（与标准 UTF-8 同构，最多 7 字节）。"""
    if value < 0:
        raise ValueError("UTF-8 编码不接受负数")
    if value < 0x80:
        return bytes([value])
    length = 2
    while length < 7 and value >= (1 << (5 * length + 1)):
        length += 1
    out = bytearray([((0xFF << (8 - length)) & 0xFF) | (value >> (6 * (length - 1)))])
    for i in range(length - 2, -1, -1):
        out.append(0x80 | ((value >> (6 * i)) & 0x3F))
    return bytes(out)


class BitReader:
    """MSB-first 位读取器，用于解码自检。"""

    def __init__(self, data: bytes | memoryview):
        self.data = memoryview(data)
        self.pos = 0

    def read(self, n: int) -> int:
        end = self.pos + n
        b0, b1 = self.pos >> 3, (end + 7) >> 3
        chunk = int.from_bytes(self.data[b0:b1], "big")
        value = (chunk >> ((b1 << 3) - end)) & ((1 << n) - 1)
        self.pos = end
        return value

    def read_unary(self) -> int:
        """读一串 0 直到遇到 1，返回 0 的个数。"""
        q = 0
        pos = self.pos
        data = self.data
        while True:
            byte_i = pos >> 3
            if byte_i >= len(data):
                raise EOFError("位流提前结束")
            rest = data[byte_i] & ((1 << (8 - (pos & 7))) - 1)
            if rest:
                lead = (8 - (pos & 7)) - rest.bit_length()
                q += lead
                pos += lead + 1
                break
            q += 8 - (pos & 7)
            pos = (byte_i + 1) << 3
        self.pos = pos
        return q


# ---------------------------------------------------------------- 残留计算


def fixed_residuals(x: np.ndarray, order: int) -> np.ndarray:
    """固定预测器的残留（长度 n-order）。"""
    if order == 0:
        return x.astype(np.int64)
    out = x[order:].astype(np.int64)
    for i, coef in enumerate(FIXED_COEFS[order], start=1):
        out = out - coef * x[order - i : -i].astype(np.int64)
    return out


def fixed_restore(res: np.ndarray, warmup: np.ndarray, order: int) -> np.ndarray:
    """由残留与前 order 个样本重建原始序列。"""
    n = res.size + order
    x = np.empty(n, dtype=np.int64)
    x[:order] = warmup
    if order == 0:
        return res.astype(np.int64)
    coefs = FIXED_COEFS[order]
    for i in range(order, n):
        acc = int(res[i - order])
        for j, coef in enumerate(coefs, start=1):
            acc += coef * int(x[i - j])
        x[i] = acc
    return x


def zigzag(values: np.ndarray) -> np.ndarray:
    return np.where(values >= 0, values << 1, ((-values) << 1) - 1).astype(np.int64)


def unzigzag(values: np.ndarray) -> np.ndarray:
    return ((values >> 1) ^ (-(values & 1))).astype(np.int64)


def best_rice_k(u: np.ndarray) -> int:
    """选使 Rice 码总比特数最小的 k（含 4 bit 参数本身）。"""
    if u.size == 0:
        return 0
    total = int(np.sum(u))
    best_k, best_bits = 0, None
    for k in range(MAX_RICE_K + 1):
        bits = (total >> k) + u.size * (1 + k)
        if best_bits is None or bits < best_bits:
            best_k, best_bits = k, bits
    return best_k


def rice_bit_length(u: np.ndarray, k: int) -> int:
    return int(np.sum(u >> k)) + u.size * (1 + k)


def rice_bits(u: np.ndarray, k: int) -> np.ndarray:
    """把无符号残留编成 0/1 位数组。

    布局：对每个样本 —— q 个 0、一个 1（收尾）、再写 k 位余数。
    `base[i]` 定义为第 i 个样本「一元码之后」的位置，于是
    收尾的那个 1 在 base[i]-1，余数占 [base[i], base[i]+k-1]。
    """
    n = u.size
    if n == 0:
        return np.zeros(0, dtype=np.uint8)
    q = (u >> k).astype(np.int64) if k > 0 else u.astype(np.int64)
    base = np.cumsum(q + 1) + np.arange(n, dtype=np.int64) * k
    total = int(base[-1] + k)
    bits = np.zeros(total, dtype=np.uint8)
    bits[base - 1] = 1
    if k > 0:
        rem = (u & ((1 << k) - 1)).astype(np.int64)
        start = base[:, None] + np.arange(k, dtype=np.int64)[None, :]
        shifts = np.arange(k - 1, -1, -1, dtype=np.int64)
        bits[start] = ((rem[:, None] >> shifts[None, :]) & 1).astype(np.uint8)
    return bits


# ---------------------------------------------------------------- 子帧编码


def _subframe_plan(x: np.ndarray) -> tuple[int, int, int]:
    """为一段样本挑选 (order, partition_order, 预估比特数)。"""
    n = x.size
    best = None
    for order in range(5):
        if n <= order:
            continue
        res = fixed_residuals(x, order)
        u_all = zigzag(res)
        warmup_bits = order * 16
        for porder in PARTITION_ORDERS:
            nparts = 1 << porder
            part = n >> porder if n % nparts == 0 else 0
            if part == 0 or part < order:
                continue
            if u_all.size != n - order:
                continue
            bits = 8 + warmup_bits + 6 + nparts * 4  # 子帧头 + 热身 + 方法/分区阶数 + 每分区 4 bit 参数
            idx = 0
            ok = True
            for p in range(nparts):
                count = part if p > 0 else part - order
                if count < 0:
                    ok = False
                    break
                seg = u_all[idx : idx + count]
                idx += count
                if seg.size:
                    bits += rice_bit_length(seg, best_rice_k(seg))
                else:
                    bits += 4
            if not ok or idx != u_all.size:
                continue
            if best is None or bits < best[2]:
                best = (order, porder, bits)
    if best is None:
        return (0, 0, 8 + 16 * n)
    return best


def encode_subframe(x: np.ndarray) -> np.ndarray:
    """编码一个子帧，返回 0/1 位数组（含子帧头，未补字节对齐）。"""
    order, porder, _ = _subframe_plan(x)
    n = x.size
    res = fixed_residuals(x, order)
    u_all = zigzag(res)
    parts: list[np.ndarray] = []

    header_fields = [(0, 1), (0b001000 | order, 6), (0, 1)]
    parts.append(fields_to_bits(header_fields))

    if order > 0:
        warm = [int(v) & 0xFFFF for v in x[:order]]
        parts.append(fields_to_bits([(v, 16) for v in warm]))

    nparts = 1 << porder
    part = n >> porder
    # 残留编码方法（2 bit，00 = 4-bit Rice 参数）+ 分区阶数（4 bit），只写一次
    parts.append(fields_to_bits([(0, 2), (porder, 4)]))
    idx = 0
    for p in range(nparts):
        count = part if p > 0 else part - order
        seg = u_all[idx : idx + count]
        idx += count
        k = best_rice_k(seg) if seg.size else 0
        parts.append(fields_to_bits([(k, 4)]))
        if seg.size:
            parts.append(rice_bits(seg, k))
    return np.concatenate(parts)


def decode_subframe(reader: BitReader, bps: int, block_size: int) -> np.ndarray:
    """解码一个子帧（支持 constant / verbatim / fixed 0..4）。"""
    reader.read(1)  # 必须是 0
    stype = reader.read(6)
    if reader.read(1):
        raise NotImplementedError("本实现不支持 wasted bits")
    if stype == 0:  # constant
        value = reader.read(bps)
        value = value - (1 << bps) if value >= (1 << (bps - 1)) else value
        return np.full(block_size, value, dtype=np.int64)
    if stype == 1:  # verbatim
        out = np.empty(block_size, dtype=np.int64)
        for i in range(block_size):
            v = reader.read(bps)
            out[i] = v - (1 << bps) if v >= (1 << (bps - 1)) else v
        return out
    if not (0b001000 <= stype <= 0b001100):
        raise NotImplementedError(f"不支持的子帧类型 {stype:06b}")
    order = stype & 0x07
    warm = np.empty(order, dtype=np.int64)
    for i in range(order):
        v = reader.read(bps)
        warm[i] = v - (1 << bps) if v >= (1 << (bps - 1)) else v
    method = reader.read(2)
    if method == 1:
        raise NotImplementedError("本实现不写 5-bit Rice 参数")
    porder = reader.read(4)
    nparts = 1 << porder
    part = block_size >> porder
    out_res = np.empty(block_size - order, dtype=np.int64)
    idx = 0
    for p in range(nparts):
        count = part if p > 0 else part - order
        k = reader.read(4)
        for i in range(count):
            q = reader.read_unary()
            rem = reader.read(k) if k else 0
            u = (q << k) | rem
            out_res[idx] = (u >> 1) ^ -(u & 1)
            idx += 1
    return fixed_restore(out_res, warm, order)


# ---------------------------------------------------------------- 帧 / 流


def _block_size_code(bs: int) -> tuple[int, bytes]:
    table = {192: 1, 576: 2, 1152: 3, 2304: 4, 4608: 5}
    if bs in table:
        return table[bs], b""
    for code, size in ((0b1100, 4096), (0b1101, 8192), (0b1110, 16384), (0b1111, 32768)):
        if bs == size:
            return code, b""
    if bs <= 256:
        return 0b0110, bytes([bs - 1])
    return 0b0111, (bs - 1).to_bytes(2, "big")


def encode_frame(block: np.ndarray, frame_index: int) -> bytes:
    """block: (n, channels) int64。返回完整帧字节（含 CRC）。"""
    n, channels = block.shape
    bs_code, bs_extra = _block_size_code(n)
    header = pack_bits(
        [
            (0b11111111111110, 14),
            (0, 1),
            (0, 1),  # 固定块长
            (bs_code, 4),
            (0b1001, 4),  # 44100 Hz
            (channels - 1, 4),  # 独立声道
            (0b100, 3),  # 16 bit
            (0, 1),
        ]
    ) + utf8_uint(frame_index) + bs_extra
    frame = bytearray(header)
    frame.append(crc8(header))

    sub_bits = [encode_subframe(block[:, ch]) for ch in range(channels)]
    all_bits = np.concatenate(sub_bits)
    pad = (-all_bits.size) % 8
    if pad:
        all_bits = np.concatenate([all_bits, np.zeros(pad, dtype=np.uint8)])
    frame += np.packbits(all_bits).tobytes()
    frame += crc16(bytes(frame)).to_bytes(2, "big")
    return bytes(frame)


def encode(pcm: np.ndarray, sample_rate: int = 44100) -> bytes:
    """pcm: (samples, channels) int16 → FLAC 字节流。"""
    pcm = np.asarray(pcm)
    if pcm.ndim == 1:
        pcm = pcm[:, None]
    if pcm.dtype != np.int16:
        pcm = np.clip(np.round(pcm), -32768, 32767).astype(np.int16)
    n_samples, channels = pcm.shape
    bits_per_sample = 16

    frames = []
    for start in range(0, n_samples, BLOCK_SIZE):
        block = pcm[start : start + BLOCK_SIZE].astype(np.int64)
        frames.append(encode_frame(block, len(frames)))
    frame_sizes = [len(f) for f in frames]

    md5 = hashlib.md5(pcm.astype("<i2").tobytes()).digest()
    total = n_samples
    tail = n_samples % BLOCK_SIZE
    min_bs = min(BLOCK_SIZE, tail) if tail else BLOCK_SIZE
    streaminfo = (
        pack_bits(
            [
                (min_bs, 16),
                (BLOCK_SIZE, 16),
                (min(frame_sizes), 24),
                (max(frame_sizes), 24),
                (sample_rate, 20),
                (channels - 1, 3),
                (bits_per_sample - 1, 5),
                (total, 36),
            ]
        )
        + md5
    )
    out = bytearray(b"fLaC")
    out += bytes([0x80])  # 最后一个元数据块 + 类型 0（STREAMINFO）
    out += len(streaminfo).to_bytes(3, "big")
    out += streaminfo
    for f in frames:
        out += f
    return bytes(out)


def decode(data: bytes) -> tuple[np.ndarray, int]:
    """解码 FLAC 字节流（仅支持本模块写出的子集），返回 (int16 PCM, 采样率)。"""
    if data[:4] != b"fLaC":
        raise ValueError("不是 FLAC 流")
    pos = 4
    streaminfo = None
    while True:
        header = data[pos]
        last, block_type = header >> 7, header & 0x7F
        length = int.from_bytes(data[pos + 1 : pos + 4], "big")
        body = data[pos + 4 : pos + 4 + length]
        pos += 4 + length
        if block_type == 0:
            streaminfo = body
        if last:
            break
    if streaminfo is None:
        raise ValueError("缺少 STREAMINFO")
    bits = int.from_bytes(streaminfo[10:18], "big")
    sample_rate = (bits >> 44) & 0xFFFFF
    channels = ((bits >> 41) & 0x07) + 1
    bps = ((bits >> 36) & 0x1F) + 1
    total = bits & ((1 << 36) - 1)
    md5_expected = streaminfo[18:34]

    out = np.empty((total, channels), dtype=np.int16)
    written = 0
    frame_index = 0
    while written < total:
        frame_start = pos
        reader = BitReader(data)
        reader.pos = pos * 8
        if reader.read(14) != 0b11111111111110:
            raise ValueError(f"第 {frame_index} 帧同步码错误")
        reader.read(1)
        blocking = reader.read(1)
        if blocking != 0:
            raise NotImplementedError("本实现只写固定块长")
        bs_code = reader.read(4)
        sr_code = reader.read(4)
        ch_code = reader.read(4)
        bps_code = reader.read(3)
        reader.read(1)
        frame_no = _read_utf8(reader)
        if frame_no != frame_index:
            raise ValueError(f"帧号不连续：期望 {frame_index}，读到 {frame_no}")
        if bs_code == 0b0110:
            block_size = reader.read(8) + 1
        elif bs_code == 0b0111:
            block_size = reader.read(16) + 1
        else:
            block_size = {1: 192, 2: 576, 3: 1152, 4: 2304, 5: 4608, 0b1100: 4096,
                          0b1101: 8192, 0b1110: 16384, 0b1111: 32768}[bs_code]
        if ch_code > 7:
            raise NotImplementedError("本实现只写独立声道")
        header_bits = reader.pos - frame_start * 8
        header_bytes = data[frame_start : frame_start + (header_bits // 8)]
        if crc8(header_bytes) != reader.read(8):
            raise ValueError(f"第 {frame_index} 帧帧头 CRC-8 校验失败")
        block = np.empty((block_size, channels), dtype=np.int64)
        for ch in range(channels):
            block[:, ch] = decode_subframe(reader, bps, block_size)
        pad = (-(reader.pos - frame_start * 8)) % 8
        if pad:
            if reader.read(pad) != 0:
                raise ValueError("帧尾填充位不是 0")
        crc_expected = crc16(data[frame_start : reader.pos // 8])
        crc_read = reader.read(16)
        if crc_read != crc_expected:
            raise ValueError(f"第 {frame_index} 帧 CRC-16 校验失败")
        pos = reader.pos // 8
        take = min(block_size, total - written)
        out[written : written + take] = block[:take].astype(np.int16)
        written += take
        frame_index += 1

    if hashlib.md5(out.astype("<i2").tobytes()).digest() != md5_expected:
        raise ValueError("MD5 校验失败，解码结果与原 PCM 不一致")
    return out, sample_rate


def _read_utf8(reader: BitReader) -> int:
    first = reader.read(8)
    if first < 0x80:
        return first
    length = 0
    for i in range(8):
        if first & (0x80 >> i):
            length += 1
        else:
            break
    value = first & ((1 << (7 - length)) - 1)
    for _ in range(length - 1):
        value = (value << 6) | (reader.read(8) & 0x3F)
    return value


# ---------------------------------------------------------------- 文件接口


def write_flac(path: str | Path, pcm: np.ndarray, sample_rate: int = 44100) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(encode(pcm, sample_rate))
    return p


def read_wav_pcm(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as f:
        channels = f.getnchannels()
        sr = f.getframerate()
        width = f.getsampwidth()
        raw = f.readframes(f.getnframes())
    if width != 2:
        raise ValueError("只支持 16-bit PCM WAV")
    pcm = np.frombuffer(raw, dtype="<i2")
    return (pcm.reshape(-1, channels) if channels > 1 else pcm), sr


def write_wav_pcm(path: str | Path, pcm: np.ndarray, sample_rate: int = 44100) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.asarray(pcm, dtype="<i2")
    channels = 1 if pcm.ndim == 1 else pcm.shape[1]
    with wave.open(str(p), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())
    return p
