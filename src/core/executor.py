"""
模块名称: Tool Executor (工具执行器)

功能描述:

    提供 LLM 调用外部工具的统一执行入口。
    实现了工具的注册、安全校验、参数解析和函数调用。
    目前主要支持 `generate_structured_diagnosis` 等诊断辅助工具。

设计理念:

    1.  **安全沙箱**: 通过 `ALLOWED_TOOLS` 白名单机制，严格限制 AI 可调用的函数，防止恶意代码执行。
    2.  **统一接口**: 所有工具调用通过 `execute_tool_call` 统一分发，便于日志记录和异常处理。
    3.  **JSON 协议**: 遵循 LLM function calling 的 JSON 格式标准，解析 `{"tool": "name", "args": {...}}`。

依赖关系:

    - `src.tools.common`: 具体的工具函数实现。
"""

import json
import traceback
from typing import Any, Dict                                           # 类型提示
# [内部模块 | Internal Modules] =========================================================================================
from src.tools.common import generate_structured_diagnosis             # 结构化诊断工具
# [创建全局变量] =========================================================================================================
# 工具白名单：仅允许执行的安全工具
ALLOWED_TOOLS = {
    "generate_structured_diagnosis": generate_structured_diagnosis,
}
# [定义函数] ############################################################################################################
# [外部-执行工具调用] =====================================================================================================
def execute_tool_call(raw_text: str) -> Dict[str, Any] | None:
    """
    安全执行工具调用。
    仅执行白名单内的工具，防止任意代码执行。
    :param raw_text: JSON 格式的工具调用指令
    :return: 执行结果字典，失败或非法工具返回 None
    """
    # [step1] 解析工具调用指令
    tool_name, args = _parse_tool_call(raw_text)
    # [step2] 卫语句：工具名为空或不在白名单
    if not tool_name or tool_name not in ALLOWED_TOOLS:
        return None
    # [step3] 获取工具函数并执行
    tool_fn = ALLOWED_TOOLS[tool_name]
    try:
        result = tool_fn(args)
    except Exception:
        return None
    # [step4] 返回标准化结果
    return {"tool": tool_name, "result": result}
# [内部-解析工具调用] =====================================================================================================
def _parse_tool_call(raw_text: str) -> tuple[str | None, dict]:
    """
    解析 JSON 格式的工具调用指令。
    :param raw_text: 原始 JSON 字符串
    :return: (工具名称, 参数字典)，解析失败返回 (None, {})
    """
    # [step1] 尝试解析 JSON
    try:
        data = json.loads(raw_text)
    except Exception:
        return None, {}
    # [step2] 校验数据类型
    if not isinstance(data, dict):
        return None, {}
    # [step3] 提取工具名称和参数
    tool_name = data.get("tool")
    args = data.get("args", {})
    if not isinstance(args, dict):
        args = {}
    return tool_name, args
