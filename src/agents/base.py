"""
智能体（Agent）定义模块：实现专科医生智能体和多学科团队智能体
=================================================================

本模块是整个医疗诊断系统的核心，定义了两种类型的智能体：

1. Agent（专科医生智能体）：
   - 每个专科医生是一个独立的 Agent 实例
   - 负责分析特定领域的医疗问题
   - 例如：心脏科医生、消化科医生、心理医生等

2. 多学科团队（MDT 智能体）：
   - 综合所有专科医生的诊断意见
   - 使用 ReAct 推理范式生成结构化诊断报告
   - 最终输出面向患者的综合诊断结果

核心特性：
- 支持 RAG 知识检索增强：从向量数据库检索相关医学知识，注入到 Prompt 中
- 支持 ReAct 推理范式：多学科团队通过"思考-行动-观察"循环进行推理
- 自动重试机制：网络错误时使用指数退避策略自动重试
- 动态 Prompt 加载：从 YAML 配置文件加载各专科医生的提示词模板

依赖关系：
- src/services/llm.py：获取 LLM 模型实例
- src/services/rag.py：RAG 知识检索
- src/core/executor.py：安全工具执行器
- config/prompts.yaml：专科医生提示词配置
"""

# ==================== 第三方库导入 ====================
# tenacity：提供重试机制，支持指数退避策略
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
# json：用于 JSON 数据的序列化和反序列化
import json
# os：用于环境变量和文件路径操作
import os
# re：正则表达式，用于文本清洗
import re

# ==================== LangChain 导入 ====================
# PromptTemplate：LangChain 的提示词模板类，支持变量插值
from langchain_core.prompts import PromptTemplate

# ==================== 项目内部模块导入 ====================
# LLM 工厂函数：根据配置返回合适的 Chat 模型实例（Qwen/OpenAI/Gemini）
from src.services.llm import get_chat_model
# 安全工具执行器：用于 ReAct 模式中的工具调用
from src.core.executor import execute_tool_call
# Graph RAG 混合检索函数：结合向量检索和知识图谱的增强检索
# 提供两个接口：
# - retrieve_hybrid_knowledge_snippets: 简化接口，返回字符串，兼容原有 RAG
# - retrieve_hybrid_knowledge: 完整接口，返回结构化结果（包含实体、向量结果、图谱结果）
from src.services.graph_rag import retrieve_hybrid_knowledge_snippets, retrieve_hybrid_knowledge
# 日志工具函数：统一的日志记录接口
from src.services.logging import log_info, log_error, log_warn

# yaml：用于解析 YAML 格式的配置文件
import yaml


def load_prompts():
    """
    加载提示词配置文件
    
    从 config/prompts.yaml 加载所有专科医生和多学科团队的 Prompt 模板。
    YAML 文件结构示例：
    ```yaml
    specialists:
      心脏科医生: "你是一位心脏科专家..."
      消化科医生: "你是一位消化科专家..."
    multidisciplinary_team: "你是多学科医疗团队..."
    ```
    
    Returns:
        dict: Prompt 配置字典，包含以下键：
            - "specialists": 各专科医生的提示词模板
            - "multidisciplinary_team": 多学科团队的提示词模板
        如果加载失败，返回空字典 {}
    
    异常处理：
        捕获所有异常并记录错误日志，返回空字典作为兜底方案
    """
    try:
        # 打开配置文件，使用 UTF-8 编码以支持中文
        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            # 使用 safe_load 安全解析 YAML，避免代码注入风险
            return yaml.safe_load(f)
    except Exception as e:
        # 记录错误日志，但不抛出异常，保证系统可以继续运行
        log_error(f"加载 config/prompts.yaml 失败: {e}")
        # 返回空字典，后续代码会使用默认模板作为兜底
        return {}


# ==================== 全局配置初始化 ====================
# 在模块加载时初始化 Prompt 配置
# 这是一个全局变量，所有 Agent 实例共享同一份配置
# 避免每次创建 Agent 时重复读取文件
PROMPTS_CONFIG = load_prompts()


class Agent:
    """
    专科医生智能体基类
    ==================
    
    每个专科医生（如心脏科医生、消化科医生）都是一个 Agent 实例。
    Agent 负责：
    1. 根据角色从 YAML 配置文件加载对应的 Prompt 模板
    2. 使用 RAG 检索相关医学知识增强 Prompt
    3. 调用 LLM 生成专科诊断意见
    
    工作流程：
    ```
    医疗报告 → RAG 检索知识 → 构建增强 Prompt → LLM 推理 → 专科诊断意见
    ```
    
    属性说明：
    - medical_report (str): 患者的医疗报告文本
    - role (str): 智能体角色，例如 "心脏科医生"
    - extra_info (dict): 额外上下文信息（主要供子类使用）
    - prompt_template (PromptTemplate): LangChain 提示词模板对象
    - model: LLM 模型实例（支持 Qwen/OpenAI/Gemini 等）
    
    使用示例：
    ```python
    # 创建一个心脏科医生智能体
    agent = Agent(medical_report="患者主诉胸闷...", role="心脏科医生")
    
    # 异步执行诊断
    diagnosis = await agent.run_async()
    print(diagnosis)  # 输出心脏科医生的诊断意见
    ```
    """
    
    def __init__(self, medical_report=None, role=None, extra_info=None):
        """
        初始化智能体
        
        创建一个新的专科医生智能体实例，加载对应的 Prompt 模板和 LLM 模型。
        
        Args:
            medical_report (str, optional): 患者的医疗报告文本
                - 包含患者主诉、症状、检查结果等信息
                - 会被插入到 Prompt 模板中供 LLM 分析
            
            role (str, optional): 智能体角色，例如：
                - "心脏科医生"：分析心血管相关问题
                - "消化科医生"：分析消化系统问题
                - "心理医生"：分析心理/情绪问题
                - 角色名必须与 prompts.yaml 中的 key 一致
            
            extra_info (dict, optional): 额外上下文信息
                - 主要供 多学科团队 子类使用
                - 可以传入其他专科的诊断结果等附加信息
        
        初始化流程：
        1. 保存传入的参数到实例属性
        2. 调用 create_prompt_template() 生成提示词模板
        3. 调用 get_chat_model() 获取 LLM 模型实例
        """
        # 保存医疗报告内容，后续会被插入到 Prompt 中
        self.medical_report = medical_report
        # 保存智能体角色类型，用于从配置文件加载对应的 Prompt
        self.role = role
        # 保存额外上下文信息（多学科团队会用到）
        self.extra_info = extra_info
        # 根据角色生成对应的提示词模板
        self.prompt_template = self.create_prompt_template()
        # 通过工厂函数获取底层 LLM 实例
        # get_chat_model() 会根据环境变量自动选择 Qwen/OpenAI/Gemini
        # 并配置好自动容灾切换机制
        self.model = get_chat_model()

    def create_prompt_template(self):
        """
        构建提示词模板
        
        从 config/prompts.yaml 中加载对应角色的 Prompt 模板。
        如果找不到对应角色的模板，使用默认简单模板作为兜底。
        
        Prompt 模板支持 LangChain 的变量插值语法，例如：
        ```
        你是一位{role}，请分析以下报告：
        {medical_report}
        ```
        
        Returns:
            PromptTemplate: LangChain 的提示词模板对象
                - 支持 .format() 方法进行变量替换
                - 支持 .partial() 方法进行部分变量填充
        
        注意：
            多学科团队子类会重写此方法以实现不同的模板构建逻辑
        """
        # 从全局配置中获取专科医生的 Prompt 字典
        specialist_prompts = PROMPTS_CONFIG.get("specialists", {})
        # 根据当前智能体的角色获取对应的 Prompt 模板
        template = specialist_prompts.get(self.role, "")
        
        # 如果配置文件中没有找到对应角色的模板，使用默认模板
        if not template:
            # 记录警告日志，提醒管理员添加配置
            log_warn(f"未在 prompts.yaml 中找到角色 '{self.role}' 的提示词，使用默认模板。")
            # 默认模板：简单的角色扮演 + 报告分析指令
            # {medical_report} 是 LangChain 的变量占位符
            template = f"请以{self.role}的身份分析以下报告：{{medical_report}}"
        
        # 使用 LangChain 的 PromptTemplate 类创建模板对象
        # from_template() 会自动解析模板中的 {} 变量占位符
        return PromptTemplate.from_template(template)

    def _prepare_prompt(self) -> str:
        """
        准备增强后的 Prompt（核心方法）
        
        这是 Agent 执行前的准备阶段，负责：
        1. 将医疗报告填充到 Prompt 模板中
        2. 使用 RAG 检索相关医学知识并注入 Prompt
        
        RAG 增强的作用：
        - 从向量数据库检索与当前病例相关的医学知识
        - 将检索到的知识片段添加到 Prompt 开头
        - 让 LLM 在回答时可以参考专业医学知识
        - 提高诊断的准确性和专业性
        
        Returns:
            str: 增强后的完整 Prompt 文本，格式如下：
            ```
            以下是与患者情况相关的医学知识片段（供你参考，不必逐条复述）：
            [参考1] 糖尿病的典型症状包括...
            [参考2] 血糖检测方法...
            
            在参考以上知识的基础上，回答下面的任务：
            [原始 Prompt 内容]
            ```
        
        注意：
            如果 RAG 未配置或检索失败，会返回不带知识增强的原始 Prompt
        """
        # 记录日志：标识当前正在运行的智能体
        log_info(f"{self.role} 智能体正在运行……")
        
        # 第一步：格式化基础 Prompt
        # 将 medical_report 变量替换到模板中的 {medical_report} 占位符
        prompt = self.prompt_template.format(medical_report=self.medical_report)
        
        # 第二步：Graph RAG 混合知识检索增强
        # 结合向量检索（语义相似度）和知识图谱（结构化关系）
        # 只有在有医疗报告的情况下才进行检索
        if self.medical_report:
            # 调用 Graph RAG 模块进行混合检索
            # retrieve_hybrid_knowledge_snippets 会：
            # 1. 从查询中提取医学实体（症状、疾病、检查项目等）
            # 2. 在向量数据库中进行语义相似度搜索
            # 3. 在知识图谱中进行结构化查询（症状-疾病关联、疾病详情等）
            # 4. 融合两种检索结果，返回格式化的知识文本
            # 如果 Graph RAG 未配置或失败，会自动降级为纯向量检索
            rag_context = retrieve_hybrid_knowledge_snippets(self.medical_report)
            
            # 如果成功检索到相关知识，将其添加到 Prompt 前面
            if rag_context:
                prompt = (
                    # 知识片段说明：告诉 LLM 这些是参考资料，不需要逐条复述
                    "以下是与患者情况相关的医学知识（来自知识图谱和向量数据库，供你参考）：\n"
                    f"{rag_context}\n\n"
                    # 任务指引：在参考知识的基础上完成诊断任务
                    "在参考以上知识的基础上，回答下面的任务：\n"
                    f"{prompt}"
                )
        
        # 返回最终的增强 Prompt
        return prompt

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def run(self):
        """
        同步执行智能体（阻塞式）
        
        注意：此方法会阻塞当前线程，在异步环境中建议使用 run_async()
        
        重试机制配置（使用 tenacity 库）：
        - stop_after_attempt(3)：最多重试 3 次
        - wait_exponential：指数退避等待
            - multiplier=1：基础等待时间倍数
            - min=4：最小等待 4 秒
            - max=10：最大等待 10 秒
        
        工作流程：
        1. 调用 _prepare_prompt() 准备增强后的 Prompt
        2. 调用 LLM 的 invoke() 方法获取回复
        3. 提取回复内容并返回
        
        Returns:
            str: 专科医生的诊断意见文本
        
        Raises:
            Exception: LLM 调用失败时抛出异常
                - 会触发 tenacity 的自动重试机制
                - 重试 3 次后仍失败则抛出最终异常
        """
        # 准备增强后的 Prompt（包含 RAG 知识）
        prompt = self._prepare_prompt()
        try:
            # 同步调用 LLM 模型
            # invoke() 是 LangChain 的标准同步调用方法
            response = self.model.invoke(prompt)
            # 提取响应内容
            # LangChain 返回的是消息对象，需要通过 content 属性获取文本
            # 如果没有 content 属性，则将整个响应转为字符串
            return getattr(response, "content", str(response))
        except Exception as e:
            # 记录错误日志
            log_error("调用模型时发生错误：", e)
            # 重新抛出异常，触发 tenacity 的重试机制
            raise e

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def run_async(self):
        """
        异步执行智能体（非阻塞式）- 推荐使用
        
        这是推荐使用的方法，支持并发执行多个智能体。
        在 orchestrator.py 中会同时启动多个专科医生进行诊断，
        使用异步方法可以显著提高整体响应速度。
        
        重试机制配置（与同步方法相同）：
        - 最多重试 3 次
        - 使用指数退避等待策略（4-10 秒）
        
        工作流程：
        1. 调用 _prepare_prompt() 准备增强后的 Prompt
        2. 调用 LLM 的 ainvoke() 异步方法获取回复
        3. 提取回复内容并返回
        
        Returns:
            str: 专科医生的诊断意见文本
        
        Raises:
            Exception: LLM 调用失败时抛出异常（会触发自动重试）
        
        使用示例：
        ```python
        # 并发执行多个专科医生
        async def diagnose_all():
            agents = [
                Agent(report, role="心脏科医生"),
                Agent(report, role="消化科医生"),
            ]
            results = await asyncio.gather(*[a.run_async() for a in agents])
            return results
        ```
        """
        # 准备增强后的 Prompt（包含 RAG 知识）
        prompt = self._prepare_prompt()
        try:
            # 异步调用 LLM 模型
            # ainvoke() 是 LangChain 的标准异步调用方法
            # 使用 await 等待结果，但不会阻塞其他协程
            response = await self.model.ainvoke(prompt)
            # 提取响应内容并返回
            return getattr(response, "content", str(response))
        except Exception as e:
            # 记录错误日志
            log_error("调用模型时发生错误：", e)
            # 重新抛出异常，触发重试机制
            raise e


class 多学科团队(Agent):
    """
    多学科团队（MDT）智能体
    =======================
    
    这是 Agent 的子类，负责综合所有专科医生的诊断意见，
    生成最终的综合诊断报告。
    
    MDT（Multi-Disciplinary Team）是现代医学的重要诊疗模式，
    通过多个学科专家共同讨论，为患者制定最佳诊疗方案。
    
    核心功能：
    1. 接收所有专科医生的诊断报告
    2. 使用 ReAct 范式进行结构化推理
    3. 生成面向患者的综合诊断报告（3 个主要问题 + 理由 + 建议）
    
    ReAct 范式说明：
    ReAct = Reasoning + Acting，是一种让 LLM 通过
    "思考 → 行动 → 观察 → 再思考" 循环进行推理的方法。
    
    工作流程：
    ```
    各专科报告 
        ↓
    构建 MDT Prompt 
        ↓
    ReAct 推理循环：
        → Thought（思考）：分析所有专科意见
        → Action（行动）：调用 generate_structured_diagnosis 工具
        → Observation（观察）：获取结构化诊断结果
        ↓
    格式化输出最终诊断报告
    ```
    
    使用示例：
    ```python
    # 假设已经收集了各专科的诊断报告
    reports = {
        "心脏科医生": "患者心电图正常...",
        "消化科医生": "建议进行胃镜检查...",
        "心理医生": "患者存在焦虑情绪..."
    }
    
    # 创建多学科团队智能体
    mdt = 多学科团队(reports=reports)
    
    # 执行 ReAct 推理
    final_diagnosis = await mdt.run_react_async()
    ```
    """
    
    def __init__(self, reports: dict[str, str]):
        """
        初始化多学科团队智能体
        
        Args:
            reports (dict[str, str]): 各专科医生的诊断报告字典
                - key: 专科医生角色名称，如 "心脏科医生"、"消化科医生"
                - value: 该专科医生的诊断报告文本
                
                格式示例：
                ```python
                {
                    "心脏科医生": "根据心电图结果，患者心律正常...",
                    "消化科医生": "患者有胃部不适症状，建议...",
                    "心理医生": "患者近期工作压力大，存在焦虑..."
                }
                ```
        
        初始化流程：
        1. 将专科报告保存到 extra_info 属性
        2. 调用父类构造函数，设置角色为 "多学科团队"
        3. 父类会自动调用 create_prompt_template() 构建模板
        
        注意：
            多学科团队不需要原始的 medical_report，
            因为它处理的是各专科医生已经分析过的报告
        """
        # 将专科报告保存为额外信息
        # 后续 create_prompt_template() 会使用这些报告构建 Prompt
        extra_info = reports
        # 调用父类构造函数
        # - role 设为 "多学科团队"，用于日志记录
        # - medical_report 留空，因为我们处理的是专科报告而非原始病例
        # - extra_info 传入专科报告字典
        super().__init__(role="多学科团队", extra_info=extra_info)

    def create_prompt_template(self):
        """
        构建多学科团队的提示词模板
        
        与父类不同，多学科团队需要动态构建 Prompt：
        1. 收集所有有效的专科报告
        2. 将报告拼接成文本
        3. 从配置文件加载 MDT 模板
        4. 使用 partial() 预填充报告内容
        
        Returns:
            PromptTemplate: 部分填充的提示词模板
                - specialists_text: 参与会诊的专科名称列表
                - reports_text: 所有专科报告的拼接文本
        
        模板变量说明：
        - {specialists_text}: "心脏科医生、消化科医生、心理医生"
        - {reports_text}: 
            ```
            心脏科医生报告：患者心电图正常...
            消化科医生报告：建议进行胃镜检查...
            心理医生报告：患者存在焦虑情绪...
            ```
        """
        # ========== 第一步：收集有效的专科报告 ==========
        # 用于存储格式化后的报告文本
        active_reports = []
        # 用于存储参与会诊的专科名称
        active_specialists = []
        
        # 遍历所有专科报告，过滤掉空报告
        for agent_name, report_content in self.extra_info.items():
            # 检查报告内容是否非空
            if report_content and report_content.strip():
                # 格式化报告：添加专科名称前缀
                active_reports.append(f"{agent_name}报告：{report_content}")
                # 记录参与会诊的专科名称
                active_specialists.append(agent_name)

        # ========== 第二步：拼接报告文本 ==========
        # 将所有报告用换行符连接成一个大文本块
        reports_text = "\n".join(active_reports)
        # 将所有专科名称用顿号连接，用于显示
        specialists_text = "、".join(active_specialists)

        # ========== 第三步：加载模板 ==========
        # 从 YAML 配置文件获取多学科团队的 Prompt 模板
        template_str = PROMPTS_CONFIG.get("multidisciplinary_team", "")
        
        # 如果配置文件中没有模板，使用默认模板
        if not template_str:
            # 默认模板：要求 LLM 综合所有报告，列出 3 个主要问题
            template_str = """
            请以多学科医疗团队的身份进行推理。
            你将获得以下专科医生提供的患者报告：{specialists_text}。
            任务：综合全部报告，列出 3 个可能的健康问题，并逐条说明对应理由与后续建议。
            输出格式：仅返回 3 个要点的列表，每个要点包含"问题 + 理由/建议"。

            {reports_text}
            """
        
        # ========== 第四步：部分填充模板 ==========
        # 使用 partial() 方法预先填充 specialists_text 和 reports_text
        # 这样在后续调用时，这些变量就已经有值了
        return PromptTemplate.from_template(template_str).partial(
            specialists_text=specialists_text,
            reports_text=reports_text
        )

    async def run_react_async(self, max_steps: int = 2):
        """
        ReAct 模式执行：使用推理-行动范式生成结构化诊断
        
        ReAct（Reasoning + Acting）是一种让 LLM 通过
        "思考-行动-观察" 循环进行推理的方法，特别适合需要
        工具调用和多步推理的复杂任务。
        
        ReAct 循环步骤：
        1. Thought（思考）：LLM 分析当前状态，决定下一步行动
        2. Action（行动）：调用工具（如 generate_structured_diagnosis）
        3. Observation（观察）：获取工具执行结果
        4. 重复上述步骤，直到得出最终答案
        
        本方法的具体工作流程：
        - Step 1: LLM 思考 → 决定调用 generate_structured_diagnosis 工具
        - Step 2: 执行工具 → 获取结构化诊断（issues 数组）
        - Step 3: LLM 基于工具结果 → 生成面向患者的中文总结
        
        Args:
            max_steps (int): 最大推理步数，默认 2 步
                - 通常 2 步足够：第一步调用工具，第二步生成总结
                - 增加步数可以处理更复杂的推理，但也会增加延迟
        
        Returns:
            str | None: 最终诊断报告文本
                - 成功时返回格式化的诊断报告（Markdown 格式）
                - 失败时返回 None
        
        输出格式示例：
        ```markdown
        #### 1. 功能性消化不良
        - 理由：患者有上腹部不适、早饱感等症状...
        - 建议：调整饮食习惯，必要时进行胃镜检查...
        
        #### 2. 焦虑相关躯体化症状
        - 理由：患者近期工作压力大...
        - 建议：学习放松技巧，考虑心理咨询...
        ```
        """
        # ==================== 初始化状态 ====================
        # 记录推理历史（每个步骤的 thought 和 tool）
        history = []
        # 当前观察结果（工具执行后的返回值）
        observation = None

        # 直接使用中文角色名作为 key
        # 将专科报告复制一份，避免修改原始数据
        reports_state = dict(self.extra_info)

        # ==================== ReAct 推理循环 ====================
        for step in range(max_steps):
            # ========== 构建 ReAct Prompt ==========
            # 这个 Prompt 指导 LLM 按照 ReAct 范式输出 JSON 格式的推理步骤
            prompt = (
                # 角色设定：多学科医疗团队
                "你是一支多学科医疗团队，正在使用 ReAct 策略进行推理。"
                # 输出格式要求：严格的 JSON 格式
                "请只输出一个 JSON，对象格式如下："
                "{"
                "  \"thought\": \"当前一步的思考\","  # 思考过程
                "  \"tool\": \"generate_structured_diagnosis\" 或 null,"  # 要调用的工具
                "  \"args\": { ... } 或 null,"  # 工具参数
                "  \"final_answer\": 如果已经完成推理则给出最终面向患者的中文总结，否则为 null"
                "}"
                "。不要输出除该 JSON 外的任何文字。"
                # 关键规则：指导 LLM 何时调用工具、何时输出最终答案
                "非常重要的规则："
                # 规则1：第一次调用时必须使用工具
                "1）当 last_observation 为 null 时，你必须设置 tool = \"generate_structured_diagnosis\"，"
                "   且 args 必须是形如 {\"issues\": [...]} 的对象，issues 数组不能为空，"
                "   每个元素需包含 name、reason、suggestion 和 department_evidence（支持各专科字段）。"
                # 规则2：工具返回结果后，生成最终答案
                "2）当 last_observation 中已经包含非空的 issues 时，你必须设置 tool = null, args = null，"
                "   并在 final_answer 中给出面向患者的中文总结，不要再调用任何工具。"
            )

            # ========== 构建当前状态 ==========
            # 状态包含了 LLM 做决策所需的所有信息
            state = {
                "history": history,  # 之前的推理步骤（用于追踪思维链）
                "last_observation": observation,  # 上次工具调用的结果
                "reports": reports_state,  # 所有专科医生的诊断报告
            }

            # 将状态信息序列化为 JSON 并附加到 Prompt
            # ensure_ascii=False 保证中文正常显示
            full_prompt = prompt + "\n当前状态：" + json.dumps(state, ensure_ascii=False)

            # ========== Step 1: 调用 LLM 进行思考 ==========
            try:
                # 异步调用 LLM
                response = await self.model.ainvoke(full_prompt)
            except Exception as e:
                # LLM 调用失败，记录错误并返回 None
                log_error("多学科团队 ReAct 调用模型时发生错误：", e)
                return None

            # 提取 LLM 返回的文本内容
            raw_text = getattr(response, "content", str(response))

            # ========== Step 2: 解析 JSON 响应 ==========
            try:
                # 尝试直接解析 JSON
                data = json.loads(raw_text)
            except Exception:
                # JSON 解析失败，尝试清洗文本
                # 某些模型（如 DeepSeek）会添加 <think> 标签
                clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
                # 移除 Markdown 代码块标记
                clean_text = clean_text.replace("```json", "").replace("```", "").strip()
                try:
                    # 再次尝试解析
                    data = json.loads(clean_text)
                except:
                    # 如果还是解析失败，直接返回原始文本（降级处理）
                    # 这确保即使 JSON 解析失败，用户也能看到 LLM 的输出
                    return raw_text

            # ========== 提取 ReAct 步骤信息 ==========
            # 从 JSON 中提取关键字段
            thought = data.get("thought")  # LLM 的思考过程
            tool_name = data.get("tool")  # 要调用的工具名称（或 null）

            # 记录推理过程到日志（便于调试）
            log_info(f"[ReAct Step {step+1}] Thought:", thought)
            log_info(f"[ReAct Step {step+1}] Tool:", tool_name)

            # 将当前步骤添加到历史记录
            # 这样下一轮循环时 LLM 可以看到之前的推理过程
            history.append({"thought": thought, "tool": tool_name})

            # ========== Step 3: 检查是否已有最终答案 ==========
            # 如果 LLM 已经给出了 final_answer，直接返回
            final_answer = data.get("final_answer")
            if final_answer:
                return final_answer

            # ========== Step 4: 执行工具调用 ==========
            # 获取工具参数（如果有）
            tool_args = data.get("args") or {}

            # 检查是否需要调用 generate_structured_diagnosis 工具
            if tool_name == "generate_structured_diagnosis":
                # 构建工具调用的 JSON
                tool_call = json.dumps({"tool": tool_name, "args": tool_args}, ensure_ascii=False)
                # 调用安全执行器执行工具
                # execute_tool_call 会验证工具是否在白名单中，然后执行
                observation = execute_tool_call(tool_call)
                
                # ========== 打印工具调用结果（美化输出） ==========
                if isinstance(observation, dict):
                    # 提取结果中的 issues 数组
                    result_for_log = observation.get("result") or {}
                    issues_for_log = result_for_log.get("issues") or []

                    log_info("[ReAct] Observation: 工具 generate_structured_diagnosis 返回结构化诊断：")

                    # 遍历并打印每个问题
                    if isinstance(issues_for_log, list) and issues_for_log:
                        for idx, issue_item in enumerate(issues_for_log, start=1):
                            if not isinstance(issue_item, dict):
                                continue

                            # 提取问题的各个字段
                            name = str(issue_item.get("name", "")).strip() or "未命名问题"
                            reason = str(issue_item.get("reason", "")).strip()
                            suggestion = str(issue_item.get("suggestion", "")).strip()

                            # 格式化输出
                            log_info(f"  问题 {idx}：{name}")
                            if reason:
                                log_info(f"    理由：{reason}")
                            if suggestion:
                                log_info(f"    建议：{suggestion}")
                    else:
                        log_info("[ReAct] Observation: 未从 result 中解析到有效的 issues。")
                else:
                    # 如果返回的不是字典，直接打印
                    log_info("[ReAct] Observation:", observation)

                # ========== 格式化最终诊断报告 ==========
                # 如果工具成功返回非空 issues，则将其格式化为 Markdown 文本
                if isinstance(observation, dict):
                    result = observation.get("result") or {}
                    issues = result.get("issues") or []
                    
                    # 检查是否有有效的问题列表
                    if isinstance(issues, list) and issues:
                        # 用于存储格式化后的各个问题
                        parts: list[str] = []
                        
                        # 遍历每个问题，生成 Markdown 格式的文本
                        for idx, issue in enumerate(issues, start=1):
                            # 提取字段并清理空白字符
                            name = str(issue.get("name", "")).strip() or "未命名问题"
                            reason = str(issue.get("reason", "")).strip()
                            suggestion = str(issue.get("suggestion", "")).strip()

                            # 构建问题的 Markdown 块
                            # 使用四级标题显示问题名称
                            lines: list[str] = [f"#### {idx}. {name}"]
                            # 添加理由（如果有）
                            if reason:
                                lines.append(f"- 理由：{reason}")
                            # 添加建议（如果有）
                            if suggestion:
                                lines.append(f"- 建议：{suggestion}")

                            # 将各行合并为一个文本块
                            block = "\n".join(lines)
                            parts.append(block)

                        # 如果成功生成了问题列表，返回格式化的报告
                        if parts:
                            # 各个问题之间用两个换行分隔
                            return "\n\n".join(parts)
            else:
                # 如果不是调用工具，清空观察结果
                observation = None

        # 如果循环结束还没有返回结果，返回 None
        # 这种情况通常表示 LLM 没有按预期工作
        return None
