"""
核心编排模块：医疗诊断多学科会诊流程
==================================

本模块负责协调整个多学科会诊流程，包括：
1. 智能分诊：根据病例自动选择相关专科
2. 并发诊断：多个专科智能体同时分析病例
3. 结果汇总：多学科团队整合所有专科意见

核心函数：
- generate_diagnosis: 执行完整的诊断流程并以流式方式返回结果
"""

import os
import asyncio
import time
from typing import List, Tuple, Optional
from src.agents.base import (
    Agent,
    多学科团队,
    PROMPTS_CONFIG
)
from src.services.logging import log_info, log_warn, log_error
from src.core.triage import triage_specialists
from src.services.cache import get_cache, DiagnosisCache
from src.core.settings import get_settings


async def generate_diagnosis(medical_report: str, use_cache: bool = True):
    """
    执行多学科诊断流程，以流式生成器的方式返回结果
    ================================================
    
    工作流程：
    1. 检查缓存（如果启用）
    2. 从配置文件加载所有可用专科医生
    3. 使用 LLM 进行智能分诊，选择最相关的专科
    4. 并发运行多个专科智能体进行分析
    5. 收集所有专科意见
    6. 由多学科团队汇总并生成最终诊断
    7. 保存到缓存（如果启用）
    
    Args:
        medical_report (str): 患者的医疗报告文本
        use_cache (bool): 是否使用缓存，默认 True
    
    Yields:
        tuple[str, str]: 三种类型的结果：
            - ("Status", 状态信息): 流程进度更新
            - (专科名称, 诊断意见): 单个专科的分析结果
            - ("Final Diagnosis", 最终报告): 多学科团队的综合诊断
    
    Example:
        >>> async for role, content in generate_diagnosis(report):
        >>>     if role == "Status":
        >>>         print(f"进度: {content}")
        >>>     elif role == "Final Diagnosis":
        >>>         print(f"最终诊断: {content}")
        >>>     else:
        >>>         print(f"{role}: {content}")
    """
    
    settings = get_settings()
    start_time = time.time()
    
    # ==================== 第零步：检查缓存 ====================
    if use_cache and settings.enable_cache:
        cache = get_cache()
        report_hash = DiagnosisCache.compute_hash(medical_report)
        
        cached_result = cache.get(report_hash, ttl=settings.cache_ttl)
        if cached_result:
            yield "Status", "📋 从缓存加载诊断结果..."
            log_info(f"[Orchestrator] 使用缓存的诊断结果 (耗时: {time.time() - start_time:.2f}秒)")
            yield "Final Diagnosis", cached_result["diagnosis"]
            return
    
    # ==================== 第一步：加载专科配置 ====================
    # 从 config/prompts.yaml 动态获取所有可用专科医生
    specialist_prompts = PROMPTS_CONFIG.get("specialists", {})
    available_specialists = list(specialist_prompts.keys())
    
    # 如果配置文件为空，使用默认专科列表作为后备方案
    if not available_specialists:
        available_specialists = [
            "心脏科医生", "心理医生", "精神科医生", "肺科医生", "神经科医生", 
            "内分泌科医生", "免疫科医生", "消化科医生", "皮肤科医生", 
            "肿瘤科医生", "血液科医生", "肾脏科医生", "风湿科医生"
        ]

    # ==================== 第二步：智能分诊 ====================
    # 向前端发送进度更新
    yield "Status", "正在分析病例进行智能分诊..."
    
    # 使用 LLM 分析病例，从所有专科中选择最相关的几个
    selected_names = await triage_specialists(medical_report, available_specialists)
    
    # 如果分诊失败（返回空列表），降级为使用所有专科
    if not selected_names:
        selected_names = available_specialists  # 兜底策略
        
    # 通知前端已选择的专科
    yield "Status", f"已启动专家会诊：{'、'.join(selected_names)}"

    # ==================== 第三步：初始化智能体 ====================
    # 为每个被选中的专科创建一个 Agent 实例
    # Agent 会根据 role 从配置文件中加载对应的提示词
    agents = {name: Agent(medical_report, role=name) for name in selected_names}

    # 用于收集所有专科的诊断结果
    responses: dict[str, str | None] = {}

    # ==================== 第四步：并发执行专科诊断（带优化） ====================
    # 定义包装函数，用于在并发执行时保留智能体名称，并添加超时控制
    async def wrapped_run(name, agent):
        """包装 Agent.run_async()，返回 (名称, 结果) 元组，带超时控制"""
        try:
            res = await asyncio.wait_for(
                agent.run_async(), 
                timeout=settings.agent_timeout
            )
            return name, res
        except asyncio.TimeoutError:
            log_warn(f"[Orchestrator] {name} 诊断超时")
            return name, f"诊断超时（超过 {settings.agent_timeout} 秒）"
        except Exception as e:
            log_error(f"[Orchestrator] {name} 诊断出错: {e}")
            return name, f"诊断过程发生错误: {str(e)}"

    # 限制并发数，避免资源过载
    semaphore = asyncio.Semaphore(settings.max_concurrent_agents)
    
    async def limited_run(name, agent):
        """带并发限制的执行"""
        async with semaphore:
            return await wrapped_run(name, agent)
    
    # 创建所有智能体的异步任务
    wrapped_tasks = [limited_run(name, agent) for name, agent in agents.items()]
    
    # 使用 asyncio.as_completed 实现并发执行和流式返回
    # 这样可以在第一个专科完成时立即返回结果，无需等待所有专科
    for coro in asyncio.as_completed(wrapped_tasks):
        agent_name, response = await coro
        responses[agent_name] = response
        # 流式返回每个专科的诊断结果
        yield agent_name, response

    # ==================== 第五步：多学科团队汇总 ====================
    # 过滤掉 None 值（可能由于 API 错误导致某些智能体返回 None）
    valid_responses = {k: v for k, v in responses.items() if v}
    
    # 实例化多学科团队智能体，传入所有专科的诊断结果
    team_agent = 多学科团队(reports=valid_responses)

    # 首先尝试使用 ReAct 模式（可以调用工具进行结构化输出）
    final_diagnosis = await team_agent.run_react_async()

    # 如果 ReAct 失败，回退到普通对话模式
    if not final_diagnosis:
        log_warn("ReAct 模式未返回有效结果，回退到普通多学科诊断。")
        final_diagnosis = await team_agent.run_async()

    # 返回最终的综合诊断报告
    yield "Final Diagnosis", final_diagnosis
    
    # ==================== 第六步：保存到缓存 ====================
    if use_cache and settings.enable_cache and final_diagnosis:
        try:
            cache = get_cache()
            report_hash = DiagnosisCache.compute_hash(medical_report)
            # 计算置信度（基于有效响应的比例）
            confidence = len(valid_responses) / len(selected_names) if selected_names else 0.0
            cache.set(report_hash, final_diagnosis, confidence)
        except Exception as e:
            log_warn(f"[Orchestrator] 保存缓存失败: {e}")
    
    # 记录总耗时
    total_time = time.time() - start_time
    log_info(f"[Orchestrator] 诊断完成，总耗时: {total_time:.2f}秒")
