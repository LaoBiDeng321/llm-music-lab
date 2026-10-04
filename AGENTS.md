# 项目约束（tmskit：文本乐谱 → 电子音）

本文件只写约束与结构。**实际状态以代码与 `out/` 产物为准**；文档落后于代码时，以代码为准并更新文档。

## 1. 项目目标

从纯文本乐谱（TMS 格式）渲染出音频。乐谱由人/agent 直接读写，工具链负责解析、校验、合成、混音、导出。

## 2. 目录职责

| 路径 | 职责 | 备注 |
|---|---|---|
| `tmskit/` | 工具链唯一实现（Python 包） | 见 §4 |
| `score/*.tms` | 乐谱本体（内容，非代码） | 一首曲子一个文件 |
| `docs/` | 规范与设计文档 | 规范先于实现，改格式必须同时改 §docs/01 |
| `out/` | 渲染产物（wav / png / 分析报告） | 可随时删除重生成 |
| `reference/` | 外部调研原始笔记 | 只读资料，不参与构建 |

## 3. 硬约束

- 只用 Python 3.11+ 的**标准库 + numpy**（可视化另需 Pillow）。**禁止**引入其他第三方依赖、禁止联网下载、禁止调用 ffmpeg 等外部程序。
- 不使用 cmd / ps1 / vbs 脚本；一切自动化走 Python 入口 `python -m tmskit ...`。
- 渲染必须**确定性可复现**：同一乐谱 + 同一 `seed` → 相同音频（`out/*.wav` 逐字节一致）。
- 音频内部一律 float64 运算，输出 16-bit PCM WAV（44100 Hz 立体声）。**禁止**未限幅直接导出。
- 分享用的小体积交付走内置 `tmskit flac`（无损），不得为此引入外部编码器。
- 渲染前后端分离：`score.py` 只解析、`render.py` 只调度、`voices.py`/`drums.py` 只产生波形、`effects.py` 只做时频处理。
- 乐谱里不写任何数值混音参数（增益/混响量在 `instrument` 行声明），音色参数属工具链，不属乐谱。

## 4. 工具链分层

```
tmskit/
  theory.py     音名/和弦/音阶/频率换算（纯函数）
  score.py      TMS 解析 → Score 数据模型
  validate.py   静态校验（错误 + 警告）
  voices.py     有音高音色：加法合成 + FM
  drums.py      无音高打击乐
  effects.py    混响/延迟/EQ/声场/限幅（纯数组处理）
  render.py     调度：Score → 分轨缓冲 → 总线 → wav
  analyze.py    对渲染结果做数值验证（电平/动态/频谱/削波）
  visualize.py  钢琴卷帘 PNG
  flac.py       纯 Python FLAC 编解码（无损压缩交付，含解码自检）
  cli.py        命令行入口
```

## 5. 命名例外（技术约束）

规范要求文件名小写短横线，但 Python 包/模块必须可 `import`，故 `tmskit/**/*.py` 使用下划线命名；所有非 Python 文件（docs/score/out）一律小写短横线。

## 6. 修改乐谱时的要求

1. 改 `score/*.tms` 后必须跑 `python -m tmskit check`，错误为 0 才能渲染。
2. 新增 `patch` 必须同时在 `docs/02-音色与混音.md` 登记参数表。
3. 新增 TMS 语法必须同时在 `docs/01-格式规范-tms.md` 登记，并补 `validate.py` 规则。
4. 渲染后必须跑 `python -m tmskit analyze`，确认无削波、整体 RMS 在 -18..-12 dBFS、真峰值 ≤ -1 dBFS。
