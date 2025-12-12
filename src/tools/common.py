"""安全工具集合：只包含允许被 LLM 间接调用的函数。

目前主要提供：
- generate_structured_diagnosis: 对多学科团队生成的 issues 结构做基础校验与规范化。
- clean_llm_json_response: 清洗 LLM 返回的 JSON 文本。
"""

import re
from typing import Any, Dict, List


def clean_llm_json_response(raw_text: str) -> str:
    """
    清洗 LLM 返回的文本，移除常见的非 JSON 标记。
    
    处理：
    1. 移除 <think>...</think> 标签（某些模型的推理标记）
    2. 移除 Markdown 代码块标记（```json ... ```）
    
    Args:
        raw_text: LLM 返回的原始文本
    
    Returns:
        清洗后的文本
    """
    # 移除某些模型可能添加的推理标签（如 Qwen 的 <think>）
    clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
    # 移除 Markdown 代码块标记
    clean_text = clean_text.replace("```json", "").replace("```", "").strip()
    return clean_text


def generate_structured_diagnosis(payload: Dict[str, Any]) -> Dict[str, Any]:
    """对模型生成的 issues 列表进行简单清洗与结构化校验。

    期望输入格式：
    {
        "issues": [
            {
                "name": str,       # 问题名称
                "reason": str,     # 诊断理由
                "suggestion": str  # 建议措施
            },
            ...
        ]
    }
    """

    issues = payload.get("issues") or []
    if not isinstance(issues, list):
        issues = []

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

    return {"issues": normalized}
