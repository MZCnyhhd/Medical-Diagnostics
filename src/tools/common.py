"""安全工具集合：只包含允许被 LLM 间接调用的函数。

目前主要提供：
- generate_structured_diagnosis: 对多学科团队生成的 issues 结构做基础校验与规范化。
"""

from typing import Any, Dict, List


def generate_structured_diagnosis(payload: Dict[str, Any]) -> Dict[str, Any]:
    """对模型生成的 issues 列表进行简单清洗与结构化校验。

    期望输入格式：
    {
        "issues": [
            {
                "name": str,
                "reason": str,
                "suggestion": str,
                "department_evidence": {
                    "cardiology": str,
                    "psychology": str,
                    "pulmonology": str,
                    "neurology": str,
                    "endocrinology": str,
                    "immunology": str,
                }
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

        dept = item.get("department_evidence") or {}
        if not isinstance(dept, dict):
            dept = {}

        normalized.append(
            {
                "name": str(item.get("name", "")).strip(),
                "reason": str(item.get("reason", "")).strip(),
                "suggestion": str(item.get("suggestion", "")).strip(),
                "department_evidence": {
                    "cardiology": str(dept.get("cardiology", "")).strip(),
                    "psychology": str(dept.get("psychology", "")).strip(),
                    "psychiatry": str(dept.get("psychiatry", "")).strip(),
                    "pulmonology": str(dept.get("pulmonology", "")).strip(),
                    "neurology": str(dept.get("neurology", "")).strip(),
                    "endocrinology": str(dept.get("endocrinology", "")).strip(),
                    "immunology": str(dept.get("immunology", "")).strip(),
                },
            }
        )

    return {"issues": normalized}
