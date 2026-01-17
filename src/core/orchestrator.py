# [导入模块] ############################################################################################################
# [标准库 | Standard Libraries] =========================================================================================
"""
模块名称: Diagnosis Orchestrator (诊断流程编排器)

功能描述:

    作为系统的"中枢神经"，负责协调和串联整个医疗诊断流程。
    从接收用户输入开始，依次执行分诊 (Triage)、多学科会诊 (MDT)、综合诊断生成等步骤。
    管理全局状态和错误处理，通过生成器 (Generator) 模式向前端流式反馈进度。

设计理念:

    1.  **流程管道化**: 将诊断过程抽象为 Step-by-Step 的管道 (Pipeline)，便于维护和扩展。
    2.  **流式反馈**: 使用 `yield` 关键字实时产出进度信息，提升用户体验 (避免长时间白屏)。
    3.  **容错设计**: 在关键步骤 (如 RAG 检索、模型调用) 包含异常捕获，确保流程不中断。

线程安全性:

    - 编排器本身通常在 Streamlit 的脚本线程中运行。
    - 调用的子模块 (如 RAG, Agent) 可能包含异步或并发操作。

依赖关系:

    - `src.core.triage`: 分诊模块。
    - `src.agents.base`: 智能体模块。
    - `src.services.graph_rag`: 检索服务。
"""

import asyncio
import json
import time                                                            # 时间工具：性能计时
# [内部模块 | Internal Modules] =========================================================================================
from src.agents.base import Agent, 多学科团队, PROMPTS_CONFIG            # 智能体：专科医生与 MDT 团队
from src.services.logging import log_info, log_warn, log_error         # 统一日志服务
from src.core.triage import triage_specialists                         # 智能分诊：动态选择专科
from src.services.cache import get_cache, DiagnosisCache               # 缓存服务：诊断结果复用
from src.services.graph_rag import retrieve_hybrid_knowledge_snippets  # 检索增强
from src.core.settings import get_settings                             # 系统配置：超时、并发等参数
# [定义函数] ############################################################################################################
# [异步-外部-生成诊断] ====================================================================================================
async def generate_diagnosis(medical_report: str, use_cache: bool = True):
    """
    生成医疗诊断的异步生成器。
    流程：缓存检查 -> 智能分诊 -> 并发专科诊断 -> MDT 综合诊断。
    :param medical_report: 医疗报告文本
    :param use_cache: 是否启用缓存
    :yields: (阶段名称, 内容) 元组
    """
    # [step1] 初始化：获取系统配置并开始计时
    settings = get_settings()
    start_time = time.time()
    # [step2] 缓存检查：若命中则直接返回缓存结果
    if use_cache and settings.enable_cache:
        cached_result = await _try_load_cache(medical_report, settings)
        if cached_result:
            yield "Status", "📋 从缓存加载诊断结果..."
            log_info(f"[Orchestrator] 使用缓存的诊断结果 (耗时: {time.time() - start_time:.2f}秒)")
            yield "Final Diagnosis", cached_result["diagnosis"]
            return
    # [step3] 智能分诊：根据报告内容选择相关专科医生
    available_specialists = _get_available_specialists()
    yield "Status", "正在分析病例进行智能分诊..."
    selected_names = await triage_specialists(medical_report, available_specialists)
    if not selected_names:
        selected_names = available_specialists
    yield "Status", f"已启动专家会诊：{'、'.join(selected_names)}"
    yield "Status", "正在检索相关医学知识..."
    # [step4] 预检索 RAG 上下文 (优化：一次检索，多次复用)
    rag_context = None
    try:
        # 在 Executor 中执行 RAG 检索，避免阻塞事件循环
        # 注意：这里简化为直接调用，如果 retrieve_hybrid_knowledge_snippets 内部耗时严重，建议放到 thread pool
        rag_context = retrieve_hybrid_knowledge_snippets(medical_report)
    except Exception as e:
        log_warn(f"[Orchestrator] RAG 预检索失败: {e}")

    # [step5] 并发专科诊断：创建 Agent 实例并并发执行 (注入 RAG 上下文)
    agents = {name: Agent(medical_report, role=name, rag_context=rag_context) for name in selected_names}
    responses = await _run_all_agents(agents, settings)
    # [step6] 逐个输出专科诊断结果
    for agent_name, response in responses.items():
        yield agent_name, response
    # [step6] MDT 综合诊断：汇总专科报告，执行 ReAct 推理
    valid_responses = {k: v for k, v in responses.items() if v}
    team_agent = 多学科团队(reports=valid_responses)
    final_diagnosis = await team_agent.run_react_async()
    # [step7] 降级处理：ReAct 失败时回退到普通模式
    if not final_diagnosis:
        log_warn("ReAct 模式未返回有效结果，回退到普通多学科诊断。")
        final_diagnosis = await team_agent.run_async()
    yield "Final Diagnosis", final_diagnosis
    # [step8] 缓存保存：将诊断结果写入缓存供后续复用
    if use_cache and settings.enable_cache and final_diagnosis:
        _save_to_cache(medical_report, final_diagnosis, len(valid_responses), len(selected_names))
    # [step9] 完成：记录总耗时日志
    total_time = time.time() - start_time
    log_info(f"[Orchestrator] 诊断完成，总耗时: {total_time:.2f}秒")
# [内部-获取可用专科] =====================================================================================================
def _get_available_specialists() -> list[str]:
    """
    获取可用的专科医生列表。
    优先从 YAML 配置加载，配置缺失时使用默认列表。
    :return: 专科名称列表
    """
    # [step1] 从配置获取专科列表
    specialist_prompts = PROMPTS_CONFIG.get("specialists", {})
    available = list(specialist_prompts.keys())
    # [step2] 卫语句：配置为空时使用默认列表
    if available:
        return available
    return [
        "心脏科医生", "心理医生", "精神科医生", "肺科医生", "神经科医生",
        "内分泌科医生", "免疫科医生", "消化科医生", "皮肤科医生",
        "肿瘤科医生", "血液科医生", "肾脏科医生", "风湿科医生"
    ]
# [内部-尝试加载缓存] =====================================================================================================
async def _try_load_cache(medical_report: str, settings) -> dict | None:
    """
    尝试从缓存加载诊断结果。
    :param medical_report: 医疗报告文本
    :param settings: 系统配置对象
    :return: 缓存的诊断结果字典，未命中返回 None
    """
    # [step1] 获取缓存服务实例
    cache = get_cache()
    # [step2] 计算报告的哈希值作为缓存键
    report_hash = DiagnosisCache.compute_hash(medical_report)
    # [step3] 查询缓存并返回结果
    return cache.get(report_hash, ttl=settings.cache_ttl)
# [内部-执行单个专科诊断] ==================================================================================================
async def _run_single_agent(name: str, agent: Agent, timeout: int) -> tuple[str, str]:
    """
    执行单个专科医生的诊断（带超时保护）。
    :param name: 专科名称
    :param agent: Agent 实例
    :param timeout: 超时秒数
    :return: (专科名称, 诊断结果)
    """
    # [step1] 尝试在超时限制内执行异步诊断
    try:
        res = await asyncio.wait_for(agent.run_async(), timeout=timeout)
        return name, res
    # [step2] 捕获超时异常，返回超时提示
    except asyncio.TimeoutError:
        log_warn(f"[Orchestrator] {name} 诊断超时")
        return name, f"诊断超时（超过 {timeout} 秒）"
    # [step3] 捕获其他异常，返回错误信息
    except Exception as e:
        log_error(f"[Orchestrator] {name} 诊断出错: {e}")
        return name, f"诊断过程发生错误: {str(e)}"
# [异步-内部-执行所有代理] =================================================================================================
async def _run_all_agents(agents: dict[str, Agent], settings) -> dict[str, str]:
    """
    并发执行所有专科医生诊断（带并发限制）。
    :param agents: 专科名称到 Agent 实例的映射
    :param settings: 系统配置对象
    :return: 专科名称到诊断结果的映射
    """
    # [step1] 创建信号量限制并发数
    semaphore = asyncio.Semaphore(settings.max_concurrent_agents)
    # [step2] 定义带限流的执行函数
    async def limited_run(name: str, agent: Agent) -> tuple[str, str]:
        async with semaphore:
            return await _run_single_agent(name, agent, settings.agent_timeout)
    # [step3] 并发执行所有任务
    tasks = [limited_run(name, agent) for name, agent in agents.items()]
    results = await asyncio.gather(*tasks)
    # [step4] 转换为字典返回
    return dict(results)
# [内部-保存缓存] ========================================================================================================
def _save_to_cache(medical_report: str, diagnosis: str, valid_count: int, total_count: int):
    """
    将诊断结果保存到缓存。
    :param medical_report: 医疗报告文本
    :param diagnosis: 诊断结果
    :param valid_count: 有效响应数
    :param total_count: 总专科数
    """
    try:
        # [step1] 获取缓存服务实例
        cache = get_cache()
        # [step2] 计算报告哈希值作为缓存键
        report_hash = DiagnosisCache.compute_hash(medical_report)
        # [step3] 计算诊断置信度（有效响应比例）
        confidence = valid_count / total_count if total_count else 0.0
        # [step4] 写入缓存
        cache.set(report_hash, diagnosis, confidence)
    except Exception as e:
        # [step5] 异常处理：记录警告但不中断流程
        log_warn(f"[Orchestrator] 保存缓存失败: {e}")
