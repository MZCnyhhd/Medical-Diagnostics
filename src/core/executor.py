"""
安全工具执行器：负责解析 LLM 返回的 JSON，并安全调用受控工具
============================================================

本模块实现了 ReAct 范式中的工具调用环节，是系统安全性的关键保障。

ReAct 范式简介：
ReAct = Reasoning + Acting（推理 + 行动）
是一种让 LLM 通过 "思考 → 行动 → 观察" 循环进行推理的方法。
在这个循环中：
- Thought（思考）：LLM 分析问题，决定下一步
- Action（行动）：调用外部工具执行操作
- Observation（观察）：获取工具的执行结果

本模块的作用：
处理 Action 阶段的工具调用，确保：
1. 只能调用白名单中预先注册的工具（安全沙箱）
2. 正确解析 LLM 返回的 JSON 格式工具调用请求
3. 执行工具并返回结果，供 LLM 进行下一步推理
4. 异常隔离：工具执行失败不会导致整个系统崩溃

安全机制详解：
=============

1. 白名单机制（ALLOWED_TOOLS）
   - 只有预先在白名单中注册的工具才能被调用
   - 即使 LLM 尝试调用其他函数，也会被拒绝
   - 防止 LLM 被 Prompt 注入攻击利用执行危险操作

2. 类型验证
   - 严格验证 JSON 格式和参数类型
   - 拒绝任何不符合预期格式的输入

3. 异常隔离
   - 工具执行过程中的任何异常都会被捕获
   - 返回 None 而不是抛出异常
   - 确保单个工具的问题不会影响整体系统

工具调用流程：
```
LLM 输出 JSON → 解析 JSON → 验证工具白名单 → 验证参数 → 执行工具 → 返回结果
```

核心函数：
- execute_tool_call：安全执行工具调用的主入口

依赖关系：
- src/tools/common.py：工具函数定义
"""

# ==================== 标准库导入 ====================
# json：用于解析 LLM 返回的 JSON 字符串
import json
# typing：类型注解
from typing import Any, Dict

# ==================== 项目内部模块导入 ====================
# 从工具模块导入允许被调用的工具函数
from src.tools.common import generate_structured_diagnosis


# ==================== 工具白名单定义 ====================
# 这是安全沙箱的核心：只有在此字典中注册的工具才能被 LLM 调用
# 
# 字典结构：
# - key: 工具名称（字符串），LLM 在 JSON 中使用这个名称指定要调用的工具
# - value: 工具函数（Python 函数对象），实际执行的函数
#
# 新增工具步骤：
# 1. 在 src/tools/ 目录下实现工具函数
# 2. 在此字典中添加映射
# 3. 更新 ReAct Prompt，告知 LLM 新工具的用法
ALLOWED_TOOLS = {
    # generate_structured_diagnosis：生成结构化诊断报告
    # 输入：包含 issues 数组的字典
    # 输出：标准化的诊断报告结构
    "generate_structured_diagnosis": generate_structured_diagnosis,
    
    # ========== 未来可扩展的工具示例 ==========
    # 以下是一些可能添加的工具，目前被注释掉：
    #
    # "search_medical_literature": search_literature,
    # 功能：搜索医学文献数据库
    # 用途：获取最新的医学研究支持诊断
    #
    # "calculate_drug_dosage": calculate_dosage,
    # 功能：计算药物剂量
    # 用途：根据患者体重、年龄等计算推荐剂量
    #
    # "query_drug_interactions": check_interactions,
    # 功能：查询药物相互作用
    # 用途：检查多种药物同时使用是否有禁忌
}


def execute_tool_call(raw_text: str) -> Dict[str, Any] | None:
    """
    安全执行工具调用：解析 LLM 的 JSON 输出并调用受控工具
    
    这是 ReAct 范式中的关键环节：
    - LLM 在 Thought 阶段决定调用哪个工具
    - 本函数解析工具调用请求并安全执行
    - 返回执行结果作为 Observation，供 LLM 继续推理
    
    安全检查流程：
    ==============
    
    第一步：JSON 解析
    - 尝试将输入解析为 JSON
    - 如果不是有效 JSON，直接拒绝
    
    第二步：格式验证
    - 检查解析结果是否为字典类型
    - 检查是否包含必要的 "tool" 字段
    
    第三步：白名单验证
    - 检查请求的工具是否在 ALLOWED_TOOLS 中
    - 不在白名单中的工具一律拒绝执行
    
    第四步：参数验证
    - 检查 args 是否为字典类型
    - 如果不是，使用空字典作为默认值
    
    第五步：执行工具
    - 从白名单获取对应的函数
    - 调用函数并传入参数
    - 捕获所有异常，防止崩溃
    
    预期输入格式（raw_text）：
    ```json
    {
      "tool": "generate_structured_diagnosis",
      "args": {
        "issues": [
          {
            "name": "功能性消化不良",
            "reason": "患者有上腹部不适、早饱感...",
            "suggestion": "建议调整饮食习惯..."
          },
          {
            "name": "焦虑相关躯体化症状",
            "reason": "患者近期工作压力大...",
            "suggestion": "学习放松技巧..."
          }
        ]
      }
    }
    ```
    
    Args:
        raw_text (str): LLM 返回的 JSON 字符串
            - 必须是有效的 JSON 格式
            - 必须包含 "tool" 字段（工具名称）
            - 可选包含 "args" 字段（工具参数）
    
    Returns:
        Dict[str, Any] | None: 工具执行结果
            
            成功时返回：
            ```python
            {
                "tool": "generate_structured_diagnosis",  # 工具名称
                "result": {                               # 工具返回值
                    "issues": [
                        {"name": "...", "reason": "...", "suggestion": "..."},
                        ...
                    ]
                }
            }
            ```
            
            失败时返回 None，可能的失败原因：
            - raw_text 不是有效的 JSON
            - 解析结果不是字典
            - 工具不在白名单中
            - 工具执行过程中抛出异常
    
    使用示例：
    ```python
    # LLM 返回的工具调用请求
    tool_call_json = '''
    {
        "tool": "generate_structured_diagnosis",
        "args": {
            "issues": [
                {"name": "感冒", "reason": "有发热症状", "suggestion": "多休息"}
            ]
        }
    }
    '''
    
    # 执行工具调用
    result = execute_tool_call(tool_call_json)
    
    if result:
        print(f"工具 {result['tool']} 执行成功")
        print(f"结果: {result['result']}")
    else:
        print("工具调用失败")
    ```
    """
    
    # ==================== 第一步：解析 JSON ====================
    try:
        # 尝试将输入字符串解析为 JSON 对象
        data = json.loads(raw_text)
    except Exception:
        # 不是合法 JSON，直接放弃工具调用
        # 这是安全措施：拒绝任何无效输入
        return None

    # ==================== 第二步：格式验证 ====================
    # 验证解析结果是否为字典类型
    # JSON 对象会被解析为 Python 字典
    if not isinstance(data, dict):
        # 如果是数组或其他类型，拒绝处理
        return None

    # 提取工具名称
    # "tool" 字段指定要调用哪个工具
    tool_name = data.get("tool")
    # 提取工具参数
    # "args" 字段包含传给工具的参数，默认空字典
    args = data.get("args", {})

    # ==================== 第三步：白名单验证 ====================
    # 这是安全检查的核心：只允许调用白名单中的工具
    # 即使 LLM 被恶意 Prompt 诱导尝试调用危险函数
    # （如 os.system、eval 等），也会在这里被拦截
    if tool_name not in ALLOWED_TOOLS:
        # 非白名单工具，拒绝执行
        # 不记录日志，避免泄露敏感信息
        return None

    # ==================== 第四步：参数验证 ====================
    # 确保 args 是字典类型
    # 如果 LLM 返回了其他类型的参数，使用空字典
    if not isinstance(args, dict):
        args = {}

    # 从白名单获取工具函数
    tool_fn = ALLOWED_TOOLS[tool_name]

    # ==================== 第五步：执行工具 ====================
    try:
        # 调用工具函数，传入参数
        # 大多数工具函数接受一个字典参数
        result = tool_fn(args)
    except Exception:
        # 工具内部报错也不向外抛出
        # 这确保工具执行失败不会导致整个系统崩溃
        # 只返回 None，让调用方知道工具调用失败
        return None

    # ==================== 第六步：返回结果 ====================
    # 返回标准格式的结果
    # 包含工具名称和执行结果
    # 这个格式会作为 Observation 传回给 LLM
    return {"tool": tool_name, "result": result}
