"""安全执行器：负责解析 LLM 返回的 JSON，并调用受控工具。

只允许调用白名单中的工具函数，避免任意代码执行。
"""

import json
from typing import Any, Dict

from src.tools.common import generate_structured_diagnosis

# 允许被调用的工具白名单映射
ALLOWED_TOOLS = {
    "generate_structured_diagnosis": generate_structured_diagnosis,
}


def execute_tool_call(raw_text: str) -> Dict[str, Any] | None:
    """解析 LLM 的原始输出文本，并在安全前提下调用工具。

    预期 raw_text 是类似：
    {
      "tool": "generate_structured_diagnosis",
      "args": { ... }
    }
    的 JSON 字符串。
    """

    try:
        data = json.loads(raw_text)
    except Exception:
        # 不是合法 JSON，直接放弃工具调用
        return None

    if not isinstance(data, dict):
        return None

    tool_name = data.get("tool")
    args = data.get("args", {})

    if tool_name not in ALLOWED_TOOLS:
        # 非白名单工具，不执行
        return None

    if not isinstance(args, dict):
        args = {}

    tool_fn = ALLOWED_TOOLS[tool_name]

    try:
        result = tool_fn(args)
    except Exception:
        # 工具内部报错也不向外抛出，仅返回 None
        return None

    return {"tool": tool_name, "result": result}
