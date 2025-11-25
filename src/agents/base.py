"""智能体定义：全部使用中文命名。

本模块现在通过 src.services.llm.get_chat_model 来获取底层 LLM，
由 OpenAI 提供对话能力。"""

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json
import os

from langchain_core.prompts import PromptTemplate  # 从 langchain_core 中导入 PromptTemplate，用于构造提示模板
from src.services.llm import get_chat_model  # 从自定义工厂函数中获取 Chat 模型实例
from src.core.executor import execute_tool_call  # 从安全执行器导入工具调用函数
from src.services.rag import retrieve_knowledge_snippets
from src.services.logging import log_info, log_error

import yaml

# 加载提示词配置
def load_prompts():
    try:
        with open("config/prompts.yaml", "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        log_error(f"加载 config/prompts.yaml 失败: {e}")
        return {}

PROMPTS_CONFIG = load_prompts()

class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):  # 初始化通用智能体，接收病例、角色与额外信息
        self.medical_report = medical_report  # 保存医疗报告内容
        self.role = role  # 保存智能体角色类型
        self.extra_info = extra_info  # 保存额外上下文信息
        self.prompt_template = self.create_prompt_template()  # 根据角色生成对应提示模板
        self.model = get_chat_model()  # 通过工厂函数获取底层 LLM（基于 OpenAI）

    def create_prompt_template(self):  # 构建提示模板的方法
        # 专科医生逻辑 (多学科团队逻辑在子类中重写)
        specialist_prompts = PROMPTS_CONFIG.get("specialists", {})
        template = specialist_prompts.get(self.role, "")
        
        if not template:
            # 兜底：如果 YAML 加载失败或找不到角色，使用默认简单 Prompt
            log_warn(f"未在 prompts.yaml 中找到角色 '{self.role}' 的提示词，使用默认模板。")
            template = f"请以{self.role}的身份分析以下报告：{{medical_report}}"
            
        return PromptTemplate.from_template(template)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def run(self):  # 执行智能体的方法
        log_info(f"{self.role} 智能体正在运行……")  # 提示当前运行的智能体角色（中文）
        prompt = self.prompt_template.format(medical_report=self.medical_report)  # 用医疗报告填充提示模板
        if self.medical_report:
            rag_context = retrieve_knowledge_snippets(self.medical_report)
            if rag_context:
                prompt = (
                    "以下是与患者情况相关的医学知识片段（供你参考，不必逐条复述）：\n"
                    f"{rag_context}\n\n"
                    "在参考以上知识的基础上，回答下面的任务：\n"
                    f"{prompt}"
                )
        try:
            response = self.model.invoke(prompt)  # 调用 LLM 获取回复
            return getattr(response, "content", str(response))  # 返回模型输出的文本内容
        except Exception as e:  # 捕获运行异常
            log_error("调用模型时发生错误：", e)  # 在终端输出中文错误提示
            raise e # 抛出异常以触发重试

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def run_async(self):
        log_info(f"{self.role} 智能体正在运行……")
        prompt = self.prompt_template.format(medical_report=self.medical_report)
        if self.medical_report:
            rag_context = retrieve_knowledge_snippets(self.medical_report)
            if rag_context:
                prompt = (
                    "以下是与患者情况相关的医学知识片段（供你参考，不必逐条复述）：\n"
                    f"{rag_context}\n\n"
                    "在参考以上知识的基础上，回答下面的任务：\n"
                    f"{prompt}"
                )
        try:
            response = await self.model.ainvoke(prompt)
            return getattr(response, "content", str(response))
        except Exception as e:
            log_error("调用模型时发生错误：", e)
            raise e

class 多学科团队(Agent):  # 定义多学科团队智能体
    def __init__(self, reports: dict[str, str]):  # 接收动态的专科报告字典
        # reports 格式：{"心脏科医生": "报告内容...", "心理医生": "报告内容..."}
        extra_info = reports
        super().__init__(role="多学科团队", extra_info=extra_info)  # 初始化时仅传角色与额外信息

    def create_prompt_template(self):  # 重写 create_prompt_template 以包含新科室
        # 动态构建提示模板，只包含有报告的科室
        active_reports = []
        active_specialists = []
        
        for agent_name, report_content in self.extra_info.items():
            if report_content and report_content.strip():
                active_reports.append(f"{agent_name}报告：{report_content}")
                active_specialists.append(agent_name)

        reports_text = "\n".join(active_reports)
        specialists_text = "、".join(active_specialists)

        # 从 YAML 获取模板，如果获取失败使用默认值
        template_str = PROMPTS_CONFIG.get("multidisciplinary_team", "")
        if not template_str:
            template_str = """
            请以多学科医疗团队的身份进行推理。
            你将获得以下专科医生提供的患者报告：{specialists_text}。
            任务：综合全部报告，列出 3 个可能的健康问题，并逐条说明对应理由与后续建议。
            输出格式：仅返回 3 个要点的列表，每个要点包含“问题 + 理由/建议”。

            {reports_text}
            """
            
        return PromptTemplate.from_template(template_str).partial(
            specialists_text=specialists_text,
            reports_text=reports_text
        )

    async def run_react_async(self, max_steps: int = 2):
        history = []
        observation = None

        # 简便起见，我们将中文角色名映射回英文 key，用于 ReAct 的 state 展示
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

        for _ in range(max_steps):
            prompt = (
                "你是一支多学科医疗团队，正在使用 ReAct 策略进行推理。"\
                "请只输出一个 JSON，对象格式如下："\
                "{"\
                "  \"thought\": \"当前一步的思考\","\
                "  \"tool\": \"generate_structured_diagnosis\" 或 null,"\
                "  \"args\": { ... } 或 null,"\
                "  \"final_answer\": 如果已经完成推理则给出最终面向患者的中文总结，否则为 null"\
                "}"\
                "。不要输出除该 JSON 外的任何文字。"\
                "非常重要的规则："\
                "1）当 last_observation 为 null 时，你必须设置 tool = \"generate_structured_diagnosis\"，"\
                "   且 args 必须是形如 {\"issues\": [...]} 的对象，issues 数组不能为空，"\
                "   每个元素需包含 name、reason、suggestion 和 department_evidence（支持各专科字段）。"\
                "2）当 last_observation 中已经包含非空的 issues 时，你必须设置 tool = null, args = null，"\
                "   并在 final_answer 中给出面向患者的中文总结，不要再调用任何工具。"\
            )

            state = {
                "history": history,
                "last_observation": observation,
                "reports": reports_state,
            }

            full_prompt = prompt + "\n当前状态：" + json.dumps(state, ensure_ascii=False)

            try:
                response = await self.model.ainvoke(full_prompt)
            except Exception as e:
                log_error("多学科团队 ReAct 调用模型时发生错误：", e)
                return None

            raw_text = getattr(response, "content", str(response))

            try:
                data = json.loads(raw_text)
            except Exception:
                # 尝试简单的清洗
                import re
                clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
                clean_text = clean_text.replace("```json", "").replace("```", "").strip()
                try:
                    data = json.loads(clean_text)
                except:
                    return raw_text

            thought = data.get("thought")
            tool_name = data.get("tool")

            # 打印当前 ReAct 步骤的思考与计划调用的工具
            log_info("[ReAct] Thought:", thought)
            log_info("[ReAct] Tool:", tool_name)

            history.append({"thought": thought, "tool": tool_name})

            # 如果模型已经直接给出了最终答案，则优先返回
            final_answer = data.get("final_answer")
            if final_answer:
                return final_answer

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
