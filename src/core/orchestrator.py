"""
核心编排模块：医疗诊断多学科会诊流程
=====================================

本模块是整个诊断系统的"指挥官"，负责协调整个多学科会诊流程。

多学科会诊（MDT）流程说明：
MDT（Multi-Disciplinary Team）是现代医学的重要诊疗模式，
通过多个学科专家共同讨论，为患者制定最佳诊疗方案。

本模块实现的诊断流程：
1. 智能分诊：根据病例自动选择相关专科（避免不必要的专科参与）
2. 并发诊断：多个专科智能体同时分析病例（提高效率）
3. 结果汇总：多学科团队整合所有专科意见（生成综合诊断）

核心函数：
- generate_diagnosis: 执行完整的诊断流程并以流式方式返回结果

技术特点：
- 异步并发：使用 asyncio 实现多智能体并发执行
- 流式输出：使用 async generator 实现实时进度反馈
- 缓存机制：支持诊断结果缓存，提高重复查询效率
- 超时控制：每个智能体有独立的超时限制，避免单点阻塞

依赖关系：
- src/agents/base.py：Agent 和 多学科团队 智能体定义
- src/core/triage.py：智能分诊函数
- src/services/cache.py：诊断结果缓存
- config/prompts.yaml：专科医生配置
"""

# ==================== 标准库导入 ====================
# os：用于环境变量操作
import os
# asyncio：Python 异步编程库，用于并发执行多个智能体
import asyncio
# time：用于计时，统计诊断耗时
import time
# typing：类型注解，提高代码可读性
from typing import List, Tuple, Optional

# ==================== 项目内部模块导入 ====================
# Agent：专科医生智能体基类
# 多学科团队：MDT 智能体，负责综合诊断
# PROMPTS_CONFIG：从 YAML 加载的专科医生配置
from src.agents.base import (
    Agent,
    多学科团队,
    PROMPTS_CONFIG
)
# 日志工具函数
from src.services.logging import log_info, log_warn, log_error
# 智能分诊函数：根据病例选择相关专科
from src.core.triage import triage_specialists
# 缓存服务：用于缓存和检索诊断结果
from src.services.cache import get_cache, DiagnosisCache
# 配置管理：获取系统配置（超时时间、并发数等）
from src.core.settings import get_settings


async def generate_diagnosis(medical_report: str, use_cache: bool = True):
    """
    执行多学科诊断流程，以流式生成器的方式返回结果
    ==================================================
    
    这是整个诊断系统的核心入口函数，实现了完整的多学科会诊流程。
    使用 Python 的 async generator 实现流式输出，让前端可以
    实时显示诊断进度和各专科的意见。
    
    工作流程详解：
    ==============
    
    第零步：检查缓存
    ---------------
    - 计算医疗报告的哈希值
    - 查询缓存是否有相同报告的历史诊断结果
    - 如果有且未过期，直接返回缓存结果（节省 API 调用）
    
    第一步：加载专科配置
    -------------------
    - 从 config/prompts.yaml 读取所有可用专科医生列表
    - 如果配置文件为空，使用内置的默认专科列表
    
    第二步：智能分诊
    ---------------
    - 将医疗报告发送给 LLM 进行分析
    - LLM 从所有专科中选择最相关的几个（通常 2-5 个）
    - 避免不必要的专科参与，提高效率和相关性
    
    第三步：初始化智能体
    -------------------
    - 为每个被选中的专科创建一个 Agent 实例
    - 每个 Agent 会加载对应的 Prompt 模板
    
    第四步：并发执行专科诊断
    ----------------------
    - 使用 asyncio.as_completed 并发执行所有智能体
    - 每个智能体完成时立即返回结果（无需等待所有完成）
    - 使用信号量控制并发数，避免资源过载
    - 每个智能体有独立的超时控制
    
    第五步：多学科团队汇总
    --------------------
    - 收集所有专科的诊断意见
    - 创建 多学科团队 智能体
    - 使用 ReAct 模式进行结构化推理
    - 生成最终的综合诊断报告
    
    第六步：保存到缓存
    ----------------
    - 将最终诊断结果保存到缓存
    - 下次相同报告查询时可以直接返回
    
    Args:
        medical_report (str): 患者的医疗报告文本
            - 包含患者主诉、症状、检查结果等信息
            - 这是整个诊断流程的输入
        
        use_cache (bool): 是否使用缓存，默认 True
            - True：检查缓存，如果有历史结果则直接返回
            - False：跳过缓存，强制重新诊断
    
    Yields:
        tuple[str, str]: 三种类型的结果元组
            
            1. 状态更新：("Status", 状态信息)
               - 用于前端显示进度，如 "正在分析病例进行智能分诊..."
            
            2. 专科意见：(专科名称, 诊断意见)
               - 如 ("心脏科医生", "根据心电图结果...")
               - 每个专科完成时立即返回，无需等待其他专科
            
            3. 最终诊断：("Final Diagnosis", 最终报告)
               - 多学科团队的综合诊断结果
               - 这是整个流程的最终输出
    
    使用示例：
    ```python
    async def main():
        report = "患者，男，45岁，主诉胸闷、气短..."
        
        async for role, content in generate_diagnosis(report):
            if role == "Status":
                print(f"进度: {content}")
            elif role == "Final Diagnosis":
                print(f"最终诊断:\\n{content}")
            else:
                print(f"{role}: {content}")
    ```
    
    性能说明：
    - 并发执行：多个专科同时诊断，总时间约等于最慢的专科
    - 流式返回：第一个专科完成即可显示，用户体验更好
    - 缓存加速：相同报告的重复查询可以秒级返回
    """
    
    # ==================== 获取配置 ====================
    # 获取全局配置实例（包含超时时间、并发数等设置）
    settings = get_settings()
    # 记录开始时间，用于统计总耗时
    start_time = time.time()
    
    # ==================== 第零步：检查缓存 ====================
    # 如果启用了缓存，先检查是否有历史诊断结果
    if use_cache and settings.enable_cache:
        # 获取缓存服务实例
        cache = get_cache()
        # 计算医疗报告的哈希值作为缓存 key
        # 相同内容的报告会得到相同的哈希值
        report_hash = DiagnosisCache.compute_hash(medical_report)
        
        # 尝试从缓存获取历史结果
        # ttl 参数指定缓存有效期（秒）
        cached_result = cache.get(report_hash, ttl=settings.cache_ttl)
        
        # 如果找到有效的缓存结果
        if cached_result:
            # 向前端发送状态更新
            yield "Status", "📋 从缓存加载诊断结果..."
            # 记录日志
            log_info(f"[Orchestrator] 使用缓存的诊断结果 (耗时: {time.time() - start_time:.2f}秒)")
            # 直接返回缓存的诊断结果
            yield "Final Diagnosis", cached_result["diagnosis"]
            # 提前结束函数
            return
    
    # ==================== 第一步：加载专科配置 ====================
    # 从 config/prompts.yaml 动态获取所有可用专科医生
    # PROMPTS_CONFIG 是在 base.py 中加载的全局配置
    specialist_prompts = PROMPTS_CONFIG.get("specialists", {})
    # 提取所有专科名称（字典的 key）
    available_specialists = list(specialist_prompts.keys())
    
    # 如果配置文件为空或未正确加载，使用默认专科列表作为后备方案
    # 这确保即使配置文件丢失，系统也能正常运行
    if not available_specialists:
        available_specialists = [
            "心脏科医生", "心理医生", "精神科医生", "肺科医生", "神经科医生", 
            "内分泌科医生", "免疫科医生", "消化科医生", "皮肤科医生", 
            "肿瘤科医生", "血液科医生", "肾脏科医生", "风湿科医生"
        ]

    # ==================== 第二步：智能分诊 ====================
    # 向前端发送状态更新
    yield "Status", "正在分析病例进行智能分诊..."
    
    # 使用 LLM 分析病例，从所有专科中选择最相关的几个
    # triage_specialists 是异步函数，会调用 LLM 进行分析
    selected_names = await triage_specialists(medical_report, available_specialists)
    
    # 如果分诊失败（返回空列表），降级为使用所有专科
    # 这是一个兜底策略，确保诊断流程不会因分诊失败而中断
    if not selected_names:
        selected_names = available_specialists
        
    # 通知前端已选择的专科
    # 使用中文顿号连接专科名称，更符合阅读习惯
    yield "Status", f"已启动专家会诊：{'、'.join(selected_names)}"

    # ==================== 第三步：初始化智能体 ====================
    # 为每个被选中的专科创建一个 Agent 实例
    # 使用字典推导式，key 是专科名称，value 是 Agent 实例
    # Agent 会根据 role 从配置文件中加载对应的提示词
    agents = {name: Agent(medical_report, role=name) for name in selected_names}

    # 用于收集所有专科的诊断结果
    # key 是专科名称，value 是诊断意见文本
    responses: dict[str, str | None] = {}

    # ==================== 第四步：并发执行专科诊断（带优化） ====================
    
    async def wrapped_run(name, agent):
        """
        包装 Agent.run_async()，返回 (名称, 结果) 元组，带超时控制
        
        这个包装函数的作用：
        1. 保留智能体名称，方便后续识别结果来源
        2. 添加超时控制，避免单个智能体阻塞整个流程
        3. 捕获异常，将错误转换为友好的错误信息
        
        Args:
            name (str): 智能体名称，如 "心脏科医生"
            agent (Agent): 智能体实例
        
        Returns:
            tuple[str, str]: (智能体名称, 诊断结果或错误信息)
        """
        try:
            # 使用 asyncio.wait_for 添加超时控制
            # timeout 从配置中读取（默认 30 秒）
            res = await asyncio.wait_for(
                agent.run_async(), 
                timeout=settings.agent_timeout
            )
            return name, res
        except asyncio.TimeoutError:
            # 超时处理：记录警告日志并返回超时提示
            log_warn(f"[Orchestrator] {name} 诊断超时")
            return name, f"诊断超时（超过 {settings.agent_timeout} 秒）"
        except Exception as e:
            # 其他异常处理：记录错误日志并返回错误信息
            log_error(f"[Orchestrator] {name} 诊断出错: {e}")
            return name, f"诊断过程发生错误: {str(e)}"

    # 创建信号量，限制并发数
    # 信号量是一种同步原语，用于控制同时执行的任务数量
    # 这避免了同时向 LLM API 发送过多请求，导致限流或资源耗尽
    semaphore = asyncio.Semaphore(settings.max_concurrent_agents)
    
    async def limited_run(name, agent):
        """
        带并发限制的执行函数
        
        使用 async with 获取信号量，确保同时执行的任务数不超过限制。
        当任务数达到上限时，新任务会等待直到有任务完成释放信号量。
        
        Args:
            name (str): 智能体名称
            agent (Agent): 智能体实例
        
        Returns:
            tuple[str, str]: (智能体名称, 诊断结果)
        """
        # async with 会自动获取和释放信号量
        async with semaphore:
            return await wrapped_run(name, agent)
    
    # 创建所有智能体的异步任务
    # 每个任务都是一个协程对象，还没有开始执行
    wrapped_tasks = [limited_run(name, agent) for name, agent in agents.items()]
    
    # 使用 asyncio.as_completed 实现并发执行和流式返回
    # as_completed 返回一个迭代器，按照任务完成的顺序产生结果
    # 这样可以在第一个专科完成时立即返回结果，无需等待所有专科
    for coro in asyncio.as_completed(wrapped_tasks):
        # await 等待当前最先完成的任务
        agent_name, response = await coro
        # 保存结果到字典
        responses[agent_name] = response
        # 流式返回每个专科的诊断结果
        # 前端可以实时显示每个专科的意见
        yield agent_name, response

    # ==================== 第五步：多学科团队汇总 ====================
    # 过滤掉 None 值和空字符串
    # 某些智能体可能因为 API 错误或超时返回无效结果
    valid_responses = {k: v for k, v in responses.items() if v}
    
    # 实例化多学科团队智能体
    # 传入所有专科的诊断结果，MDT 会综合分析
    team_agent = 多学科团队(reports=valid_responses)

    # 首先尝试使用 ReAct 模式
    # ReAct 可以调用工具进行结构化输出，生成更规范的诊断报告
    final_diagnosis = await team_agent.run_react_async()

    # 如果 ReAct 失败（返回 None），回退到普通对话模式
    # 这是一个兜底策略，确保总能输出诊断结果
    if not final_diagnosis:
        log_warn("ReAct 模式未返回有效结果，回退到普通多学科诊断。")
        # 使用普通的 run_async 方法，不使用工具调用
        final_diagnosis = await team_agent.run_async()

    # 返回最终的综合诊断报告
    yield "Final Diagnosis", final_diagnosis
    
    # ==================== 第六步：保存到缓存 ====================
    # 只有在启用缓存且有有效诊断结果时才保存
    if use_cache and settings.enable_cache and final_diagnosis:
        try:
            # 获取缓存服务实例
            cache = get_cache()
            # 计算报告哈希值（与检查缓存时使用相同的算法）
            report_hash = DiagnosisCache.compute_hash(medical_report)
            # 计算置信度（基于有效响应的比例）
            # 如果所有专科都成功返回，置信度为 1.0
            # 如果有专科失败，置信度会降低
            confidence = len(valid_responses) / len(selected_names) if selected_names else 0.0
            # 保存到缓存
            cache.set(report_hash, final_diagnosis, confidence)
        except Exception as e:
            # 缓存保存失败不影响主流程，只记录警告
            log_warn(f"[Orchestrator] 保存缓存失败: {e}")
    
    # ==================== 记录统计信息 ====================
    # 计算总耗时
    total_time = time.time() - start_time
    # 记录日志，便于性能分析
    log_info(f"[Orchestrator] 诊断完成，总耗时: {total_time:.2f}秒")
