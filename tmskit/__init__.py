"""tmskit：文本乐谱（TMS）→ 电子音。

对外只暴露四件事：解析（score）、校验（validate）、渲染（render）、可视化（visualize）。
"""

from .score import Score, parse_file, parse_text  # noqa: F401
from .render import render_score, write_wav  # noqa: F401
from .validate import validate, check_file  # noqa: F401

__version__ = "1.0.0"
__all__ = ["Score", "parse_file", "parse_text", "render_score", "write_wav", "validate", "check_file"]
