"""
模块名称: Common Tools (通用工具集)
功能描述:

    包含被多个 Agent 或组件共享的实用工具函数。
    主要提供 LLM 输出清洗、JSON 修复、结构化数据校验等功能。

设计理念:

    1.  **纯函数**: 尽量设计为无副作用的纯函数，便于测试和复用。
    2.  **鲁棒性**: 针对 LLM 输出的不确定性 (如 Markdown 标记、推理标签)，提供容错处理。

线程安全性:

    - 无状态函数，线程安全。

依赖关系:

    - 标准库 `re`, `json`.
"""

import re
from typing import Any, Dict, List

# [定义函数] ############################################################################################################
# [工具-清洗JSON] =========================================================================================================
def clean_llm_json_response(raw_text: str) -> str:
    """
    清洗 LLM 返回的文本，移除常见的非 JSON 标记。
    处理：
    1. 移除 <think>...</think> 标签（某些模型的推理标记）
    2. 移除 Markdown 代码块标记（```json ... ```）
    :param raw_text: LLM 返回的原始文本
    :return: 清洗后的文本
    """
    # [step1] 移除推理标签
    # 移除某些模型可能添加的推理标签（如 Qwen 的 <think>）
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    
    # [step2] 移除代码块标记
    clean_text = clean_text.replace("```json", "").replace("```", "").strip()
    return clean_text

# [工具-结构化诊断] =======================================================================================================
def generate_structured_diagnosis(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    对模型生成的 issues 列表进行简单清洗与结构化校验。

    期望输入格式：
    
    .. code-block:: json

        {
            "issues": [
                { "name": "str", "reason": "str", "suggestion": "str" },
                ...
            ]
        }

    :param payload: 原始 payload
    :return: 结构化后的字典
    """
    # [step1] 提取 issues 列表
    issues = payload.get("issues") or []
    if not isinstance(issues, list):
        issues = []

    # [step2] 遍历并规范化每个 issue
    normalized: List[Dict[str, Any]] = []

    for item in issues:
        if not isinstance(item, dict):
            continue

        normalized.append(
            {
                "name": str(item.get("name", "")).strip(),
                "reason": str(item.get("reason", "")).strip(),
                "suggestion": str(item.get("suggestion", "")).strip(),
            }
        )

    # [step3] 返回结果
    return {"issues": normalized}

