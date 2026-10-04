# 04 无损导出：自带 FLAC 编解码

32 MB 的 WAV 不方便分享，而项目硬约束禁止第三方库与外部程序（没有 ffmpeg 可用）。
因此工具链自带一个 FLAC 编解码器：`tmskit/flac.py`，只依赖标准库（`hashlib`、`wave`）+ numpy。

## 1 命令

```powershell
python -m tmskit flac out/tide-end.wav -o out/tide-end.flac --verify   # 编码 + 立刻解码自检
python -m tmskit unflac out/tide-end.flac -o out/decoded.wav           # 解回 WAV
```

本曲实测：**32.4 MB → 22.1 MB（68.2%），无损，编码 27 秒**。

## 2 实现子集

写出的流完全符合 FLAC 规范，但只用了其中一个子集（够用且好验证）：

| 项 | 取值 | 说明 |
|---|---|---|
| 元数据 | 只有 STREAMINFO | 含原始 PCM 的 MD5，解码端用来自检 |
| 块长 | 固定 4096 采样 | 最后一块可短（帧头用 16-bit 显式块长） |
| 声道 | 2 声道独立 | channel assignment = 0001，未做 mid/side |
| 位深 / 采样率 | 16 bit / 44100 Hz | 帧头用表内编码（1001 / 100） |
| 子帧 | 固定预测器 order 0–4 | 每块每声道在 5 种阶数 × 5 种分区阶数里选比特最少的一种 |
| 残留编码 | Rice，4-bit 参数 | 不触发 escape code（k 上限 14） |
| 校验 | 每帧 CRC-8（帧头）+ CRC-16（整帧） | 解码端逐帧校验 |

未实现（提高压缩率才需要）：LPC 子帧、mid/side 声道去相关、wasted bits、5-bit Rice 参数、可变块长。
即便这样，本曲也已压到 68.2%；真正的编码器（LPC + mid/side）大约能到 60–65%。

## 3 正确性验证（两道独立检查）

1. **自往返**：`tmskit flac --verify` 与 `unflac` 解回的 WAV 与原文件 **SHA-256 逐字节一致**。
2. **外部解码器**：用 Windows 自带媒体栈（WPF `MediaPlayer` → Media Foundation）打开该 FLAC，
   成功解析出时长 `00:03:12`，与乐谱一致 —— 说明这不是「只有自己认」的文件。
   ```powershell
   Add-Type -AssemblyName PresentationCore
   $p = New-Object System.Windows.Media.MediaPlayer
   $p.Open([Uri]"D:\test\out\tide-end.flac"); Start-Sleep 3; $p.NaturalDuration
   ```

单元测试覆盖：静音 / 直流 / 方波 / 白噪声 / 单声道 / 恰好一个整块 / 跨块边界（4097）/ 正弦，
全部往返一致；白噪声会略膨胀到 103%（不可压缩是正常的），正弦压到 22.6%。

## 4 已知取舍

- 解码器是**自检用**的，逐样本 Python 解码（整曲约 1–2 分钟），不作为播放用途；播放请用任意支持 FLAC 的播放器。
- 编码器只处理 16-bit / 44100 Hz / ≤2 声道；其它输入直接报错，不静默降级。
