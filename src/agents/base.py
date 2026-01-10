# [导入模块] ############################################################################################################
# [标准库 | Standard Libraries] =========================================================================================
import re                                                              # 文本模式匹配与复杂字符串解析
import json                                                            # 结构化数据序列化与反序列化
# [第三方库 | Third-party Libraries] ====================================================================================
import yaml                                                            # 系统配置文件加载与解析
from langchain_core.prompts import PromptTemplate                      # 提示词工程：动态模板构建
from tenacity import (
    retry,                                                             # 容错机制：失败自动重试策略
    stop_after_attempt,                                                # 重试限制：最大尝试次数控制
    wait_exponential                                                   # 指数退避：重试间隔时间优化
)
# [内部模块 | Internal Modules] =========================================================================================
from src.services.llm import get_chat_model                            # 模型工厂：初始化大语言模型实例
from src.core.executor import execute_tool_call                        # 动作执行器：处理 Agent 工具调用指令
from src.services.logging import log_info, log_error, log_warn         # 统一日志服务：结构化运行状态追踪
from src.services.graph_rag import retrieve_hybrid_knowledge_snippets  # 检索增强：知识图谱与向量混合检索
# [定义函数] ############################################################################################################
# [外部-加载提示词] =======================================================================================================
def load_prompts() -> dict:
    """
    从 YAML 配置文件加载所有 Agent 的提示词模板。
    该函数在模块初始化时调用，将提示词缓存到全局变量 PROMPTS_CONFIG 中。
    :return: 包含所有角色提示词的字典，加载失败时返回空字典
    """
    # [step1] 尝试读取并解析 YAML 配置文件
    try:
        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    # [step2] 捕获异常并记录错误日志，返回空字典作为降级处理
    except Exception as e:
        log_error(f"加载 config/prompts.yaml 失败: {e}")
        return {}
# [创建全局变量] =========================================================================================================
PROMPTS_CONFIG = load_prompts()
# [内部-提取问题] ========================================================================================================
def _extract_issues(observation: dict) -> list:
    """
    从工具调用返回的 observation 字典中安全提取 issues 列表。
    支持嵌套结构：observation -> result -> issues
    :param observation: 工具返回的原始观察结果
    :return: issues 列表，提取失败时返回空列表
    """
    # [step1] 类型校验：确保输入为字典
    if not isinstance(observation, dict):
        return []
    # [step2] 逐层提取嵌套数据
    result = observation.get("result") or {}
    issues = result.get("issues") or []
    # [step3] 最终类型校验并返回
    return issues if isinstance(issues, list) else []
# [内部-记录单个问题] =====================================================================================================
def _log_single_issue(idx: int, issue: dict) -> None:
    """
    将单个诊断问题格式化输出到日志系统。
    输出格式包含问题名称、理由和建议三个字段。
    :param idx: 问题序号（从 1 开始）
    :param issue: 包含 name/reason/suggestion 的问题字典
    """
    # [step1] 类型校验：非字典直接跳过
    if not isinstance(issue, dict):
        return
    # [step2] 安全提取字段并设置默认值
    name = str(issue.get("name", "")).strip() or "未命名问题"
    reason = str(issue.get("reason", "")).strip()
    suggestion = str(issue.get("suggestion", "")).strip()
    # [step3] 输出问题名称（必输出）
    log_info(f"  问题 {idx}：{name}")
    # [step4] 条件输出理由和建议（非空才输出）
    if reason:
        log_info(f"    理由：{reason}")
    if suggestion:
        log_info(f"    建议：{suggestion}")
# [内部-记录问题] ========================================================================================================
def _log_issues(issues: list) -> None:
    """
    批量记录诊断问题列表到日志系统。
    用于 ReAct 循环中观察阶段的结果输出。
    :param issues: 诊断问题列表
    """
    # [step1] 输出日志标题
    log_info("[ReAct] Observation: 工具 generate_structured_diagnosis 返回结构化诊断：")
    # [step2] 卫语句：空列表直接返回并记录警告
    if not issues:
        log_info("[ReAct] Observation: 未从 result 中解析到有效的 issues。")
        return
    # [step3] 遍历并委托单个问题的日志输出
    for idx, issue in enumerate(issues, start=1):
        _log_single_issue(idx, issue)
# [内部-格式化单个问题] ===================================================================================================
def _format_single_issue(idx: int, issue: dict) -> str | None:
    """
    将单个诊断问题格式化为 Markdown 字符串。
    输出格式：四级标题 + 无序列表（理由/建议）。
    :param idx: 问题序号（从 1 开始）
    :param issue: 包含 name/reason/suggestion 的问题字典
    :return: Markdown 格式字符串，无效输入返回 None
    """
    # [step1] 类型校验：非字典返回 None
    if not isinstance(issue, dict):
        return None
    # [step2] 安全提取字段并设置默认值
    name = str(issue.get("name", "")).strip() or "未命名问题"
    reason = str(issue.get("reason", "")).strip()
    suggestion = str(issue.get("suggestion", "")).strip()
    # [step3] 构建 Markdown 行列表（标题必加，理由/建议按需添加）
    lines = [f"#### {idx}. {name}"]
    if reason:
        lines.append(f"- 理由：{reason}")
    if suggestion:
        lines.append(f"- 建议：{suggestion}")
    # [step4] 拼接并返回
    return "\n".join(lines)
# [内部-Markdown格式问题] ================================================================================================
def _format_issues_markdown(issues: list) -> str | None:
    """
    将诊断问题列表批量格式化为 Markdown 文档。
    每个问题之间用双换行分隔，便于前端渲染。
    :param issues: 诊断问题列表
    :return: 完整的 Markdown 字符串，空列表返回 None
    """
    # [step1] 卫语句：空列表直接返回 None
    if not issues:
        return None
    # [step2] 列表推导：批量格式化每个问题
    parts = [_format_single_issue(idx, issue) for idx, issue in enumerate(issues, start=1)]
    # [step3] 过滤无效结果（None 值）
    parts = [p for p in parts if p]
    # [step4] 拼接并返回（双换行分隔）
    return "\n\n".join(parts) if parts else None
# [创建类] ##############################################################################################################
# [外部-代理] ===========================================================================================================
class Agent:  # Agent 代理
    """
    医疗 AI 智能体基类。
    封装了专科医生的核心行为：身份初始化、提示词构建、RAG 知识增强、模型调用。
    子类通过重写 create_prompt_template() 实现角色定制。
    """
    # [定义方法] ---------------------------------------------------------------------------------------------------------
    # [实例初始化] .......................................................................................................
    def __init__(self, medical_report: str = None, role: str = None, extra_info: dict = None):
        """
        初始化智能体实例。
        :param medical_report: 待分析的医疗报告文本
        :param role: 智能体角色名称（如"心血管专家"）
        :param extra_info: 扩展信息字典，供子类使用
        """
        # [step1] 绑定核心属性
        self.medical_report = medical_report
        self.role = role
        self.extra_info = extra_info
        # [step2] 构建角色专属提示词模板
        self.prompt_template = self.create_prompt_template()
        # [step3] 初始化大语言模型实例
        self.model = get_chat_model()
    # [外部-实例-创建提示词模板] ...........................................................................................
    def create_prompt_template(self):
        """
        从 YAML 配置加载角色专属的提示词模板。
        若配置缺失则使用通用兜底模板。
        :return: LangChain PromptTemplate 对象
        """
        # [step1] 从全局配置获取专科医生提示词字典
        specialist_prompts = PROMPTS_CONFIG.get("specialists", {})
        # [step2] 按角色名查找对应模板
        template = specialist_prompts.get(self.role, "")
        # [step3] 卫语句：未找到则使用默认模板并记录警告
        if not template:
            log_warn(f"未在 prompts.yaml 中找到角色 '{self.role}' 的提示词，使用默认模板。")
            template = f"请以{self.role}的身份分析以下报告：{{medical_report}}"
        # [step4] 构建并返回 LangChain 模板对象
        return PromptTemplate.from_template(template)
    # [装饰器-外部-实例] ..................................................................................................
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def run(self):
        """
        同步执行智能体诊断逻辑。
        自动重试机制：最多 3 次，指数退避（4-10 秒）。
        :return: 模型生成的诊断建议文本
        """
        # [step1] 构建 RAG 增强后的提示词
        prompt = self._prepare_prompt()
        try:
            # [step2] 同步调用大语言模型
            response = self.model.invoke(prompt)
            # [step3] 兼容提取响应内容（适配不同模型返回格式）
            return getattr(response, "content", str(response))
        except Exception as e:
            # [step4] 记录错误并抛出，触发重试机制
            log_error("调用模型时发生错误：", e)
            raise e
    # [装饰器-异步-外部-实例] ..............................................................................................
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def run_async(self):
        """
        异步执行智能体诊断逻辑。
        用于多专科医生并发诊断场景，显著提升系统吞吐量。
        :return: 模型生成的诊断建议文本
        """
        # [step1] 构建 RAG 增强后的提示词
        prompt = self._prepare_prompt()
        try:
            # [step2] 异步调用大语言模型
            response = await self.model.ainvoke(prompt)
            # [step3] 兼容提取响应内容
            return getattr(response, "content", str(response))
        except Exception as e:
            # [step4] 记录错误并抛出，触发重试机制
            log_error("异步调用模型时发生错误：", e)
            raise e
    # [内部-实例-准备提示词] ...............................................................................................
    def _prepare_prompt(self) -> str:
        """准备包含 RAG 知识增强的最终提示词"""
        log_info(f"{self.role} 智能体正在运行……")
        # [step1] 基础提示词构建
        prompt = self.prompt_template.format(medical_report=self.medical_report)
        # [step2] 卫语句：无报告或无 RAG 知识则直接返回
        if not self.medical_report:
            return prompt
        rag_context = retrieve_hybrid_knowledge_snippets(self.medical_report)
        if not rag_context:
            return prompt
        # [step3] 组合结构化增强提示词
        return (
            "### 参考医学知识 (RAG)\n"
            "以下是从权威医学库检索到的相关信息，请结合参考：\n"
            f"{rag_context}\n\n"
            "--- \n"
            "### 任务指令\n"
            "请基于以上参考知识及患者报告，以专科医生身份给出专业分析：\n"
            f"{prompt}"
        )
# [外部-MDT] ============================================================================================================
class 多学科团队(Agent):
    """
    多学科团队（MDT）智能体。
    核心职责：汇总多位专科医生的诊断报告，通过 ReAct 推理策略进行深度分析，
    输出结构化的综合诊断结论。继承自 Agent 基类。
    """
    # [定义方法] --------------------------------------------------------------------------------------------------------
    # [实例初始化] .......................................................................................................
    def __init__(self, reports: dict[str, str]):
        """
        初始化 MDT 团队智能体。
        :param reports: 专科报告字典，Key 为专科名称，Value 为诊断内容
        """
        # [step1] 将专科报告存入 extra_info 供后续使用
        extra_info = reports
        # [step2] 调用父类初始化，固定角色为"多学科团队"
        super().__init__(role="多学科团队", extra_info=extra_info)
    # [外部-实例] ........................................................................................................
    def create_prompt_template(self):
        """
        构建 MDT 专属的提示词模板。
        动态汇总各专科报告，预填充模板变量。
        :return: 预填充的 LangChain PromptTemplate 对象
        """
        # [step1] 初始化容器，过滤并收集有效的专科报告
        activate_reports = []
        active_specialists = []
        for agent_name, report_content in self.extra_info.items():
            if report_content and report_content.strip():
                activate_reports.append(f"{agent_name}报告：{report_content}")
                active_specialists.append(agent_name)
        # [step2] 格式化报告文本和专家名单
        reposts_text = "\n".join(activate_reports)
        specialists_text = "、".join(active_specialists)
        # [step3] 从 YAML 获取模板，若缺失则使用兜底模板
        template_str = PROMPTS_CONFIG.get("multidisciplinary_team", "")
        if not template_str:
            template_str = """
            请以多学科医疗团队的身份进行推理。
            你将获得以下专科医生提供的患者报告：{specialists_text}。
            任务：综合全部报告，列出 3 个可能的健康问题，并逐条说明对应理由与后续建议。
            输出格式：请使用 Markdown 格式输出。
            1. 使用三级标题（###）列出每个健康问题。
            2. 每个问题下使用无序列表（-）详细说明“理由”和“建议”。
            3. 确保排版整洁，易于阅读。

            {reports_text}
            """
        # [step4] 预填充固定变量并返回模板
        return PromptTemplate.from_template(template_str).partial(
            specialists_text=specialists_text,
            reposts_text=reposts_text
        )
    # [异步-外部-实例] ...................................................................................................
    async def run_react_async(self, max_steps: int = 2):
        """
        执行 ReAct（Reasoning and Acting）推理循环。
        流程：思考 → 工具调用 → 观察 → 循环/输出最终答案。
        :param max_steps: 最大推理步数，防止死循环
        :return: Markdown 格式的结构化诊断结论
        """
        # [step1] 初始化推理上下文
        history, observation = [], None
        reports_state = dict(self.extra_info)
        for step in range(max_steps):
            # [step2] 获取 LLM 决策（包含思考、工具选择或最终答案）
            state = {"history": history, "last_observation": observation, "reports": reports_state}
            data = await self._get_decision(state)
            if not isinstance(data, dict):
                return data  # 解析失败直接返回原文本
            # [step3] 记录当前步骤的思考过程
            log_info(f"[ReAct Step {step+1}] Thought: {data.get('thought')}")
            history.append({"thought": data.get("thought"), "tool": data.get("tool")})
            # [step4] 检查是否达成最终答案
            if final_answer := data.get("final_answer"):
                return final_answer
            # [step5] 执行工具调用
            observation = self._execute_tool(data.get("tool"), data.get("args") or {})
            if not observation:
                continue
            # [step6] 格式化观察结果并返回
            issues = _extract_issues(observation)
            if formatted := _format_issues_markdown(issues):
                return formatted
        return None
    # [内部-实例] .......................................................................................................
    def _get_react_prompt(self) -> str:
        """
        获取 ReAct 策略的系统指令。
        约束模型以 JSON 格式输出，并遵守两阶段推理规则。
        :return: 系统提示词字符串
        """
        # [step1] 返回硬编码的 ReAct 系统指令（含 JSON 格式约束和推理规则）
        return (
            "你是一支多学科医疗团队，正在使用 ReAct 策略进行推理。"
            "请只输出一个 JSON，格式如下：{\"thought\": \"...\", \"tool\": \"...\", \"args\": {...}, \"final_answer\": \"...\"}。"
            "非常重要的规则：\n"
            "1）首步必须设置 tool = \"generate_structured_diagnosis\" 并提供 args。\n"
            "2）观测到结果后，必须设置 tool = null 并给出 final_answer。\n"
            "3）final_answer 必须使用 Markdown 格式（使用 ### 标题和 - 列表）。"
        )
    # [内部-实例] .......................................................................................................
    def _parse_react_json(self, raw_text: str) -> dict | None:
        """
        稳健的 JSON 解析器。
        能处理包含 Markdown 代码块或 <think> 标签的模型输出。
        :param raw_text: 模型返回的原始文本
        :return: 解析后的字典，失败返回 None
        """
        # [step1] 尝试直接解析
        try:
            return json.loads(raw_text)
        except Exception:
            # [step2] 清洗：移除 DeepSeek 等模型的 <think> 标签
            clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            # [step3] 清洗：移除 Markdown JSON 代码块标记
            clean_text = clean_text.replace("```json", "").replace("```", "").strip()
            # [step4] 再次尝试解析
            try:
                return json.loads(clean_text)
            except:
                return None
    # [异步-内部-实例] ...................................................................................................
    async def _get_decision(self, state: dict) -> dict | str:
        """
        调用 LLM 获取单步决策。
        组合系统指令与当前状态，解析 JSON 响应。
        :param state: 当前推理状态（历史、观察、报告）
        :return: 解析后的决策字典，或原始文本（解析失败时）
        """
        # [step1] 拼接系统指令与当前状态
        full_prompt = self._get_react_prompt() + "\n当前状态：" + json.dumps(state, ensure_ascii=False)
        try:
            # [step2] 异步调用 LLM
            response = await self.model.ainvoke(full_prompt)
            raw_text = getattr(response, "content", str(response))
            # [step3] 解析 JSON，失败则返回原文本
            return self._parse_react_json(raw_text) or raw_text
        except Exception as e:
            # [step4] 错误降级：返回友好提示
            log_error("多学科团队 ReAct 调用模型时发生错误：", e)
            return "诊断暂时不可用，请稍后重试。"
    # [内部-实例] .......................................................................................................
    def _execute_tool(self, tool_name: str, tool_args: dict) -> dict | None:
        """
        安全执行工具调用。
        仅支持白名单工具 generate_structured_diagnosis，防止任意代码执行。
        :param tool_name: 工具名称
        :param tool_args: 工具参数字典
        :return: 工具返回的观察结果，非法工具返回 None
        """
        # [step1] 白名单校验：仅允许结构化诊断工具
        if tool_name != "generate_structured_diagnosis":
            return None
        # [step2] 序列化工具调用指令
        tool_call = json.dumps({"tool": tool_name, "args": tool_args}, ensure_ascii=False)
        # [step3] 委托执行器执行工具
        observation = execute_tool_call(tool_call)
        # [step4] 记录观察结果日志
        if isinstance(observation, dict):
            _log_issues(_extract_issues(observation))
        else:
            log_info("[ReAct] Observation:", observation)
        return observation
