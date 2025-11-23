import os
import asyncio
from Utils.Agents import (
    心脏科医生,
    心理医生,
    精神科医生,
    肺科医生,
    神经科医生,
    内分泌科医生,
    免疫科医生,
    消化科医生,
    皮肤科医生,
    肿瘤科医生,
    血液科医生,
    肾脏科医生,
    风湿科医生,
    多学科团队,
)
from Utils.config import RESULTS_DIR
from Utils.logging_utils import log_info, log_warn, log_error
from Utils.triage import triage_specialists

async def generate_diagnosis(medical_report: str):
    """执行多学科诊断，以流式生成器的方式返回结果。

    Yields:
        tuple[str, str]: (角色名称, 回复内容)
        或者
        tuple[str, str]: ("Final Diagnosis", 最终诊断报告)
        或者
        tuple[str, str]: ("Status", 状态信息)
    """
    # 所有可用专科
    all_agents_cls = {
        "心脏科医生": 心脏科医生,
        "心理医生": 心理医生,
        "精神科医生": 精神科医生,
        "肺科医生": 肺科医生,
        "神经科医生": 神经科医生,
        "内分泌科医生": 内分泌科医生,
        "免疫科医生": 免疫科医生,
        "消化科医生": 消化科医生,
        "皮肤科医生": 皮肤科医生,
        "肿瘤科医生": 肿瘤科医生,
        "血液科医生": 血液科医生,
        "肾脏科医生": 肾脏科医生,
        "风湿科医生": 风湿科医生,
    }

    # 1. 智能分诊
    yield "Status", "正在分析病例进行智能分诊..."
    selected_names = await triage_specialists(medical_report, list(all_agents_cls.keys()))
    
    if not selected_names:
        selected_names = list(all_agents_cls.keys()) # 兜底
        
    yield "Status", f"已启动专家会诊：{'、'.join(selected_names)}"

    # 2. 实例化被选中的智能体
    agents = {name: all_agents_cls[name](medical_report) for name in selected_names}

    # 用于收集所有回复，供后续多学科团队使用
    responses: dict[str, str | None] = {}

    # 重新实现并发逻辑：
    # 我们使用一个包装函数来返回 (name, response)，这样 as_completed 的结果里就包含名字了
    async def wrapped_run(name, agent):
        res = await agent.run_async()
        return name, res

    wrapped_tasks = [wrapped_run(name, agent) for name, agent in agents.items()]
    
    for coro in asyncio.as_completed(wrapped_tasks):
        agent_name, response = await coro
        responses[agent_name] = response
        yield agent_name, response

    # 所有专科都结束后，运行多学科团队
    # 注意：未被选中的科室，传入空字符串即可，Agents.py 那边需要适配（只把非空的放进 Prompt）
    team_agent = 多学科团队(
        cardiologist_report=responses.get("心脏科医生") or "",
        psychologist_report=responses.get("心理医生") or "",
        psychiatrist_report=responses.get("精神科医生") or "",
        pulmonologist_report=responses.get("肺科医生") or "",
        neurologist_report=responses.get("神经科医生") or "",
        endocrinologist_report=responses.get("内分泌科医生") or "",
        immunologist_report=responses.get("免疫科医生") or "",
        gastroenterologist_report=responses.get("消化科医生") or "",
        dermatologist_report=responses.get("皮肤科医生") or "",
        oncologist_report=responses.get("肿瘤科医生") or "",
        hematologist_report=responses.get("血液科医生") or "",
        nephrologist_report=responses.get("肾脏科医生") or "",
        rheumatologist_report=responses.get("风湿科医生") or "",
    )

    final_diagnosis = await team_agent.run_react_async()

    # 如果 ReAct 失败回退
    if not final_diagnosis:
        log_warn("ReAct 模式未返回有效结果，回退到普通多学科诊断。")
        final_diagnosis = await team_agent.run_async()

    yield "Final Diagnosis", final_diagnosis
