"""
智能体（Agent）定义模块：实现专科医生智能体和多学科团队智能体

本模块定义了两种类型的智能体：
1. Agent（专科医生智能体）：每个专科医生是一个独立的 Agent，负责分析特定领域的医疗问题
2. 多学科团队（MDT 智能体）：综合所有专科意见，生成最终诊断报告

核心特性：
- 支持 RAG 知识检索增强
- 支持 ReAct 推理范式（多学科团队）
- 自动重试机制（网络错误时自动重试）
- 动态 Prompt 加载（从 YAML 配置文件）
"""

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json
import os

from langchain_core.prompts import PromptTemplate  # LangChain 的提示词模板类
from src.services.llm import get_chat_model  # LLM 工厂函数，获取 Chat 模型实例
from src.core.executor import execute_tool_call  # 安全工具执行器
from src.services.rag import retrieve_knowledge_snippets  # RAG 知识检索函数
from src.services.logging import log_info, log_error, log_warn

import yaml


def load_prompts():
    """
    加载提示词配置文件
    
    从 config/prompts.yaml 加载所有专科医生和多学科团队的 Prompt 模板。
    如果加载失败，返回空字典，系统会使用默认模板作为兜底。
    
    Returns:
        dict: Prompt 配置字典，包含 "specialists" 和 "multidisciplinary_team" 两个键
    """
    try:
        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        log_error(f"加载 config/prompts.yaml 失败: {e}")
        return {}


# 全局 Prompt 配置，在模块加载时初始化
PROMPTS_CONFIG = load_prompts()

class Agent:
    """
    专科医生智能体基类
    
    每个专科医生（如心脏科医生、消化科医生）都是一个 Agent 实例。
    Agent 负责：
    1. 根据角色加载对应的 Prompt 模板
    2. 使用 RAG 检索相关医学知识
    3. 调用 LLM 生成专科诊断意见
    
    工作流程：
    医疗报告 → RAG 检索知识 → 构建 Prompt → LLM 推理 → 专科诊断意见
    """
    
    def __init__(self, medical_report=None, role=None, extra_info=None):
        """
        初始化智能体
        
        Args:
            medical_report (str, optional): 患者的医疗报告文本
            role (str, optional): 智能体角色，例如 "心脏科医生"、"消化科医生"
            extra_info (dict, optional): 额外上下文信息（多学科团队使用）
        """
        self.medical_report = medical_report  # 保存医疗报告内容
        self.role = role  # 保存智能体角色类型
        self.extra_info = extra_info  # 保存额外上下文信息
        self.prompt_template = self.create_prompt_template()  # 根据角色生成对应提示模板
        self.model = get_chat_model()  # 通过工厂函数获取底层 LLM（支持 Qwen/OpenAI/Gemini）

    def create_prompt_template(self):
        """
        构建提示词模板
        
        从 config/prompts.yaml 中加载对应角色的 Prompt 模板。
        如果找不到对应角色的模板，使用默认简单模板作为兜底。
        
        Returns:
            PromptTemplate: LangChain 的提示词模板对象
        """
        # 专科医生逻辑（多学科团队逻辑在子类中重写）
        specialist_prompts = PROMPTS_CONFIG.get("specialists", {})
        template = specialist_prompts.get(self.role, "")
        
        if not template:
            # 兜底策略：如果 YAML 加载失败或找不到角色，使用默认简单 Prompt
            log_warn(f"未在 prompts.yaml 中找到角色 '{self.role}' 的提示词，使用默认模板。")
            template = f"请以{self.role}的身份分析以下报告：{{medical_report}}"
            
        return PromptTemplate.from_template(template)

    def _enhance_prompt_with_rag(self, prompt: str) -> str:
        """
        使用 RAG 检索相关医学知识并增强 Prompt
        
        Args:
            prompt (str): 原始 Prompt
            
        Returns:
            str: 增强后的 Prompt
        """
        if self.medical_report:
            # 从向量数据库检索与医疗报告相关的医学知识片段
            rag_context = retrieve_knowledge_snippets(self.medical_report)
            if rag_context:
                # 将检索到的知识注入到 Prompt 中，作为 LLM 的参考上下文
                prompt = (
                    "以下是与患者情况相关的医学知识片段（供你参考，不必逐条复述）：\n"
                    f"{rag_context}\n\n"
                    "在参考以上知识的基础上，回答下面的任务：\n"
                    f"{prompt}"
                )
        return prompt

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def run(self):
        """
        同步执行智能体（阻塞式）
        
        工作流程：
        1. 格式化 Prompt 模板，填充医疗报告
        2. 使用 RAG 检索相关医学知识并注入 Prompt
        3. 调用 LLM 生成诊断意见
        4. 返回诊断结果
        
        注意：此方法会阻塞线程，建议在异步环境中使用 run_async()
        
        Returns:
            str: 专科医生的诊断意见文本
        
        Raises:
            Exception: LLM 调用失败时抛出异常（会触发自动重试）
        """
        log_info(f"{self.role} 智能体正在运行……")
        
        # 第一步：格式化基础 Prompt
        prompt = self.prompt_template.format(medical_report=self.medical_report)
        
        # 第二步：RAG 知识检索增强
        prompt = self._enhance_prompt_with_rag(prompt)
        
        # 第三步：调用 LLM 生成诊断意见
        try:
            response = self.model.invoke(prompt)  # 同步调用 LLM
            return getattr(response, "content", str(response))  # 提取文本内容
        except Exception as e:
            log_error("调用模型时发生错误：", e)
            raise e  # 抛出异常以触发自动重试（最多重试 3 次）

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def run_async(self):
        """
        异步执行智能体（非阻塞式）
        
        这是推荐使用的方法，支持并发执行多个智能体。
        工作流程与 run() 相同，但使用异步调用，不会阻塞事件循环。
        
        Returns:
            str: 专科医生的诊断意见文本
        
        Raises:
            Exception: LLM 调用失败时抛出异常（会触发自动重试）
        """
        log_info(f"{self.role} 智能体正在运行……")
        
        # 第一步：格式化基础 Prompt
        prompt = self.prompt_template.format(medical_report=self.medical_report)
        
        # 第二步：RAG 知识检索增强
        prompt = self._enhance_prompt_with_rag(prompt)
        
        # 第三步：异步调用 LLM 生成诊断意见
        try:
            response = await self.model.ainvoke(prompt)  # 异步调用 LLM
            return getattr(response, "content", str(response))  # 提取文本内容
        except Exception as e:
            log_error("调用模型时发生错误：", e)
            raise e  # 抛出异常以触发自动重试

class 多学科团队(Agent):
    """
    多学科团队（MDT）智能体
    
    这是 Agent 的子类，负责综合所有专科医生的诊断意见，生成最终的综合诊断报告。
    
    核心功能：
    1. 接收所有专科医生的诊断报告
    2. 使用 ReAct 范式进行结构化推理
    3. 生成面向患者的综合诊断报告（3 个主要问题 + 理由 + 建议）
    
    工作流程：
    各专科报告 → 构建 MDT Prompt → ReAct 推理 → 工具调用 → 结构化诊断 → 格式化输出
    """
    
    def __init__(self, reports: dict[str, str]):
        """
        初始化多学科团队智能体
        
        Args:
            reports (dict[str, str]): 各专科医生的诊断报告字典
                格式：{"心脏科医生": "报告内容...", "心理医生": "报告内容...", ...}
        """
        # 将专科报告保存为额外信息，供后续使用
        extra_info = reports
        # 多学科团队不需要自己的医疗报告，因为它处理的是专科医生的报告
        super().__init__(role="多学科团队", extra_info=extra_info)

    def create_prompt_template(self):
        """
        构建多学科团队的提示词模板
        
        动态构建 Prompt，只包含有有效报告的科室。
        从 config/prompts.yaml 加载模板，如果加载失败使用默认模板。
        
        Returns:
            PromptTemplate: 部分填充的提示词模板（specialists_text 和 reports_text 已填充）
        """
        # 动态构建提示模板，只包含有报告的科室
        active_reports = []  # 有效的专科报告列表
        active_specialists = []  # 参与会诊的专科名称列表
        
        # 遍历所有专科报告，过滤掉空报告
        for agent_name, report_content in self.extra_info.items():
            if report_content and report_content.strip():
                active_reports.append(f"{agent_name}报告：{report_content}")
                active_specialists.append(agent_name)

        # 将所有报告拼接成文本
        reports_text = "\n".join(active_reports)
        # 将所有专科名称拼接（用于显示）
        specialists_text = "、".join(active_specialists)

        # 从 YAML 配置文件获取模板，如果获取失败使用默认值
        template_str = PROMPTS_CONFIG.get("multidisciplinary_team", "")
        if not template_str:
            # 默认模板：要求 LLM 综合所有报告，列出 3 个主要问题
            template_str = """
            请以多学科医疗团队的身份进行推理。
            你将获得以下专科医生提供的患者报告：{specialists_text}。
            任务：综合全部报告，列出 3 个可能的健康问题，并逐条说明对应理由与后续建议。
            输出格式：仅返回 3 个要点的列表，每个要点包含"问题 + 理由/建议"。

            {reports_text}
            """
            
        # 使用 partial() 方法部分填充模板变量
        # 这样在调用时只需要填充剩余变量（如果有）
        return PromptTemplate.from_template(template_str).partial(
            specialists_text=specialists_text,
            reports_text=reports_text
        )

    async def run_react_async(self, max_steps: int = 2):
        """
        ReAct 模式执行：使用推理-行动范式生成结构化诊断
        
        ReAct（Reasoning + Acting）是一种让 LLM 通过"思考-行动-观察"循环进行推理的方法：
        1. Thought（思考）：分析当前状态，决定下一步行动
        2. Action（行动）：调用工具（如 generate_structured_diagnosis）
        3. Observation（观察）：获取工具执行结果
        4. 重复上述步骤，直到得出最终答案
        
        工作流程：
        - Step 1: LLM 思考 → 决定调用 generate_structured_diagnosis 工具
        - Step 2: 执行工具 → 获取结构化诊断（issues 数组）
        - Step 3: LLM 基于工具结果 → 生成面向患者的中文总结
        
        Args:
            max_steps (int): 最大推理步数，默认 2 步（通常足够）
        
        Returns:
            str | None: 最终诊断报告文本，如果失败返回 None
        """
        history = []  # 记录推理历史（thought 和 tool 的序列）
        observation = None  # 当前观察结果（工具执行后的返回值）

        # 将中文角色名映射回英文 key，用于 ReAct 的 state 展示（便于调试）
        role_to_key_map = {
            "心脏科医生": "cardiology",
            "心理医生": "psychology",
            "精神科医生": "psychiatry",
            "肺科医生": "pulmonology",
            "神经科医生": "neurology",
            "内分泌科医生": "endocrinology",
            "免疫科医生": "immunology",
            "消化科医生": "gastroenterology",
            "皮肤科医生": "dermatology",
            "肿瘤科医生": "oncology",
            "血液科医生": "hematology",
            "肾脏科医生": "nephrology",
            "风湿科医生": "rheumatology"
        }
        
        reports_state = {}
        for name, content in self.extra_info.items():
            key = role_to_key_map.get(name, name)
            reports_state[key] = content

        # ========== ReAct 推理循环 ==========
        for step in range(max_steps):
            # 构建 ReAct Prompt，要求 LLM 输出 JSON 格式的推理步骤
            prompt = (
                "你是一支多学科医疗团队，正在使用 ReAct 策略进行推理。"
                "请只输出一个 JSON，对象格式如下："
                "{"
                "  \"thought\": \"当前一步的思考\","
                "  \"tool\": \"generate_structured_diagnosis\" 或 null,"
                "  \"args\": { ... } 或 null,"
                "  \"final_answer\": 如果已经完成推理则给出最终面向患者的中文总结，否则为 null"
                "}"
                "。不要输出除该 JSON 外的任何文字。"
                "非常重要的规则："
                "1）当 last_observation 为 null 时，你必须设置 tool = \"generate_structured_diagnosis\"，"
                "   且 args 必须是形如 {\"issues\": [...]} 的对象，issues 数组不能为空，"
                "   每个元素需包含 name、reason、suggestion 和 department_evidence（支持各专科字段）。"
                "2）当 last_observation 中已经包含非空的 issues 时，你必须设置 tool = null, args = null，"
                "   并在 final_answer 中给出面向患者的中文总结，不要再调用任何工具。"
            )

            # 构建当前状态（包含推理历史、上次观察结果、所有专科报告）
            state = {
                "history": history,  # 之前的推理步骤
                "last_observation": observation,  # 上次工具调用的结果
                "reports": reports_state,  # 所有专科医生的诊断报告
            }

            # 将状态信息注入 Prompt
            full_prompt = prompt + "\n当前状态：" + json.dumps(state, ensure_ascii=False)

            # ========== Step 1: 调用 LLM 进行思考 ==========
            try:
                response = await self.model.ainvoke(full_prompt)
            except Exception as e:
                log_error("多学科团队 ReAct 调用模型时发生错误：", e)
                return None

            # 提取 LLM 返回的文本
            raw_text = getattr(response, "content", str(response))

            # ========== Step 2: 解析 JSON 响应 ==========
            try:
                data = json.loads(raw_text)
            except Exception:
                # JSON 解析失败，尝试清洗文本（移除可能的 markdown 标记等）
                import re
                # 移除某些模型可能添加的推理标签
                clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
                # 移除 markdown 代码块标记
                clean_text = clean_text.replace("```json", "").replace("```", "").strip()
                try:
                    data = json.loads(clean_text)
                except:
                    # 如果还是解析失败，直接返回原始文本（降级处理）
                    return raw_text

            # 提取 ReAct 步骤的关键信息
            thought = data.get("thought")  # LLM 的思考过程
            tool_name = data.get("tool")  # 要调用的工具名称（或 null）

            # 记录推理过程（便于调试和日志）
            log_info(f"[ReAct Step {step+1}] Thought:", thought)
            log_info(f"[ReAct Step {step+1}] Tool:", tool_name)

            # 将当前步骤添加到历史记录
            history.append({"thought": thought, "tool": tool_name})

            # ========== Step 3: 检查是否已有最终答案 ==========
            # 如果 LLM 已经直接给出了最终答案（跳过工具调用），直接返回
            final_answer = data.get("final_answer")
            if final_answer:
                return final_answer

            # ========== Step 4: 执行工具调用 ==========
            tool_args = data.get("args") or {}

            if tool_name == "generate_structured_diagnosis":
                tool_call = json.dumps({"tool": tool_name, "args": tool_args}, ensure_ascii=False)
                observation = execute_tool_call(tool_call)
                # 打印工具调用后的观察结果（美化输出）
                if isinstance(observation, dict):
                    result_for_log = observation.get("result") or {}
                    issues_for_log = result_for_log.get("issues") or []

                    log_info("[ReAct] Observation: 工具 generate_structured_diagnosis 返回结构化诊断：")

                    if isinstance(issues_for_log, list) and issues_for_log:
                        for idx, issue_item in enumerate(issues_for_log, start=1):
                            if not isinstance(issue_item, dict):
                                continue

                            name = str(issue_item.get("name", "")).strip() or "未命名问题"
                            reason = str(issue_item.get("reason", "")).strip()
                            suggestion = str(issue_item.get("suggestion", "")).strip()

                            log_info(f"  问题 {idx}：{name}")
                            if reason:
                                log_info(f"    理由：{reason}")
                            if suggestion:
                                log_info(f"    建议：{suggestion}")
                    else:
                        log_info("[ReAct] Observation: 未从 result 中解析到有效的 issues。")
                else:
                    log_info("[ReAct] Observation:", observation)

                # 如果工具成功返回非空 issues，则将其格式化为最终诊断文本
                if isinstance(observation, dict):
                    result = observation.get("result") or {}
                    issues = result.get("issues") or []
                    if isinstance(issues, list) and issues:
                        parts: list[str] = []
                        for idx, issue in enumerate(issues, start=1):
                            name = str(issue.get("name", "")).strip() or "未命名问题"
                            reason = str(issue.get("reason", "")).strip()
                            suggestion = str(issue.get("suggestion", "")).strip()

                            lines: list[str] = [f"#### {idx}. {name}"]
                            if reason:
                                lines.append(f"- 理由：{reason}")
                            if suggestion:
                                lines.append(f"- 建议：{suggestion}")

                            block = "\n".join(lines)
                            parts.append(block)

                        if parts:
                            return "\n\n".join(parts)
            else:
                observation = None

        return None
