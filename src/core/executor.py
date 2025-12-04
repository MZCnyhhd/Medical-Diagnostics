"""
安全工具执行器：负责解析 LLM 返回的 JSON，并安全调用受控工具

本模块实现了 ReAct 范式中的工具调用环节，确保：
1. 只允许调用白名单中的工具函数，避免任意代码执行（安全沙箱）
2. 解析 LLM 返回的 JSON 格式工具调用请求
3. 执行工具并返回结果，供 LLM 进行下一步推理

安全机制：
- 白名单机制：只有预先注册的工具才能被调用
- 异常隔离：工具执行失败不会导致整个系统崩溃
- 类型验证：严格验证 JSON 格式和参数类型
"""

import json
from typing import Any, Dict

from src.tools.common import generate_structured_diagnosis

# ========== 工具白名单 ==========
# 只允许调用白名单中的工具函数，这是安全沙箱的核心机制
# 新增工具需要在此注册，否则 LLM 无法调用
ALLOWED_TOOLS = {
    "generate_structured_diagnosis": generate_structured_diagnosis,  # 生成结构化诊断报告
    # 未来可以添加更多工具，例如：
    # "search_medical_literature": search_literature,
    # "calculate_drug_dosage": calculate_dosage,
}


def execute_tool_call(raw_text: str) -> Dict[str, Any] | None:
    """
    安全执行工具调用：解析 LLM 的 JSON 输出并调用受控工具
    
    这是 ReAct 范式中的关键环节：
    - LLM 在 Thought 阶段决定调用哪个工具
    - 本函数解析工具调用请求并安全执行
    - 返回执行结果作为 Observation，供 LLM 继续推理
    
    预期输入格式（raw_text）：
    {
      "tool": "generate_structured_diagnosis",
      "args": {
        "issues": [
          {
            "name": "问题名称",
            "reason": "理由",
            "suggestion": "建议"
          }
        ]
      }
    }
    
    Args:
        raw_text (str): LLM 返回的 JSON 字符串，包含工具名称和参数
    
    Returns:
        Dict[str, Any] | None: 工具执行结果，格式为：
            {
                "tool": "工具名称",
                "result": {...}  # 工具返回的结果
            }
        如果解析失败、工具不在白名单或执行出错，返回 None
    
    Example:
        >>> tool_call_json = '{"tool": "generate_structured_diagnosis", "args": {"issues": [...]}}'
        >>> result = execute_tool_call(tool_call_json)
        >>> print(result)
        {'tool': 'generate_structured_diagnosis', 'result': {...}}
    """
    
    # ========== 第一步：解析 JSON ==========
    try:
        data = json.loads(raw_text)
    except Exception:
        # 不是合法 JSON，直接放弃工具调用（安全：拒绝无效输入）
        return None

    # 验证是否为字典类型
    if not isinstance(data, dict):
        return None

    # 提取工具名称和参数
    tool_name = data.get("tool")
    args = data.get("args", {})

    # ========== 第二步：白名单验证 ==========
    # 安全检查：只允许调用白名单中的工具
    # 这防止了 LLM 尝试执行危险操作（如文件删除、网络请求等）
    if tool_name not in ALLOWED_TOOLS:
        # 非白名单工具，拒绝执行（安全：防止任意代码执行）
        return None

    # ========== 第三步：参数验证 ==========
    # 确保 args 是字典类型，如果不是则使用空字典
    if not isinstance(args, dict):
        args = {}

    # 从白名单获取工具函数
    tool_fn = ALLOWED_TOOLS[tool_name]

    # ========== 第四步：执行工具 ==========
    try:
        # 调用工具函数，传入参数
        result = tool_fn(args)
    except Exception:
        # 工具内部报错也不向外抛出，仅返回 None
        # 这确保工具执行失败不会导致整个系统崩溃
        return None

    # ========== 第五步：返回结果 ==========
    # 返回标准格式的结果，供 LLM 作为 Observation 使用
    return {"tool": tool_name, "result": result}
