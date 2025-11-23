"""智能体定义：全部使用中文命名。

本模块现在通过 Utils.llm_factory.get_chat_model 来获取底层 LLM，
由 OpenAI 提供对话能力。"""

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import json
import os

from langchain_core.prompts import PromptTemplate  # 从 langchain_core 中导入 PromptTemplate，用于构造提示模板
from Utils.llm_factory import get_chat_model  # 从自定义工厂函数中获取 Chat 模型实例
from Utils.safe_executor import execute_tool_call  # 从安全执行器导入工具调用函数
from Utils.rag import retrieve_knowledge_snippets
from Utils.logging_utils import log_info, log_error

class Agent:
    def __init__(self, medical_report=None, role=None, extra_info=None):  # 初始化通用智能体，接收病例、角色与额外信息
        self.medical_report = medical_report  # 保存医疗报告内容
        self.role = role  # 保存智能体角色类型
        self.extra_info = extra_info  # 保存额外上下文信息
        self.prompt_template = self.create_prompt_template()  # 根据角色生成对应提示模板
        self.model = get_chat_model()  # 通过工厂函数获取底层 LLM（基于 OpenAI）

    def create_prompt_template(self):  # 构建提示模板的方法
        if self.role == "多学科团队":  # 若角色为多学科团队
            templates = f"""
                请以多学科医疗团队的身份进行推理。
                你将获得心脏科、心理科、精神科、肺科、神经科、内分泌科、免疫科医生提供的患者报告。
                任务：综合全部报告，列出 3 个可能的健康问题，并逐条说明对应理由与后续建议。
                输出格式：仅返回 3 个要点的列表，每个要点包含“问题 + 理由/建议”。

                心脏科报告：{self.extra_info.get('cardiologist_report', '')}
                心理科报告：{self.extra_info.get('psychologist_report', '')}
                精神科报告：{self.extra_info.get('psychiatrist_report', '')}
                肺科报告：{self.extra_info.get('pulmonologist_report', '')}
                神经科报告：{self.extra_info.get('neurologist_report', '')}
                内分泌科报告：{self.extra_info.get('endocrinologist_report', '')}
                免疫科报告：{self.extra_info.get('immunologist_report', '')}
            """  # 构造多学科会诊提示，嵌入各科室报告
        else:  # 若为单一专科智能体
            templates = {  # 定义各角色的提示模板
                "心脏科医生": """
                    请以心脏科医生的身份分析以下患者报告。
                    任务：审阅心脏相关检查（如心电图、血液检测、Holter 监测、超声心动图），判断可能存在的心脏问题。
                    重点：找出可能导致症状的隐匿性心脏异常，排查心律失常与结构问题。
                    建议：说明下一步需要的检测/监测方案或管理策略。
                    输出：仅给出潜在心脏病因及推荐措施。
                    医疗报告：{medical_report}
                """,
                "心理医生": """
                    请以心理医生的身份分析以下患者报告。
                    任务：评估心理状态，识别焦虑、抑郁、创伤等潜在心理因素。
                    重点：解释这些心理因素如何影响患者整体健康。
                    建议：提供治疗、咨询、干预或管理建议。
                    输出：仅给出可能的心理问题及推荐的下一步措施。
                    患者报告：{medical_report}
                """,
                "精神科医生": """
                    请以精神科医生的身份分析以下患者报告。
                    任务：重点评估精神科相关疾病，如重性抑郁障碍、双相情感障碍、精神分裂谱系障碍、焦虑相关障碍等。
                    重点：综合躯体症状与病史，判断是否存在需要精神科干预的严重情绪或思维异常。
                    建议：给出药物治疗、心理治疗、住院/随访等方面的建议。
                    输出：仅给出可能的精神科诊断方向及推荐的下一步措施。
                    患者报告：{medical_report}
                """,
                "肺科医生": """
                    请以肺科医生的身份分析以下患者报告。
                    任务：识别哮喘、慢阻肺、肺部感染等可能的呼吸系统问题。
                    重点：说明这些呼吸问题如何导致患者症状。
                    建议：提出肺功能检查、影像学检查或治疗方案。
                    输出：仅给出可能的呼吸系统问题及推荐的下一步措施。
                    患者报告：{medical_report}
                """
                ,
                "神经科医生": """
                    请以神经科医生身份分析以下患者报告。
                    任务：关注神经系统异常，如癫痫、脑血流不足、周围神经病变等。
                    重点：解释这些神经因素如何诱发或加剧患者症状。
                    建议：提供进一步的影像学检查、神经传导检查或药物调整方案。
                    输出：仅列出可能的神经系统问题及下一步跟进措施。
                    患者报告：{medical_report}
                """
                ,
                "内分泌科医生": """
                    请以内分泌科医生身份分析以下患者报告。
                    任务：评估甲状腺、肾上腺、血糖、代谢等激素相关异常。
                    重点：说明激素失衡如何影响心血管、情绪或呼吸表现。
                    建议：给出进一步的激素检测、代谢评估或治疗建议。
                    输出：仅列出潜在的内分泌问题及推荐措施。
                    患者报告：{medical_report}
                """
                ,
                "免疫科医生": """
                    请以免疫科医生身份分析以下患者报告。
                    任务：排查自身免疫疾病、慢性炎症或过敏因素。
                    重点：阐述免疫异常对患者多系统症状的影响。
                    建议：提出免疫学化验、炎症指标监测或免疫调节治疗方案。
                    输出：仅列出潜在的免疫相关问题及调控建议。
                    患者报告：{medical_report}
                """,
                "消化科医生": """
                    请以消化科医生身份分析以下患者报告。
                    任务：评估胃肠道功能、肝胆胰疾病及消化系统相关症状。
                    重点：关注腹痛、腹泻、便秘、消化不良等症状，排查器质性或功能性病变。
                    建议：提供胃肠镜、幽门螺杆菌检测或饮食调整建议。
                    输出：仅列出可能的消化系统问题及推荐措施。
                    患者报告：{medical_report}
                """,
                "皮肤科医生": """
                    请以皮肤科医生身份分析以下患者报告。
                    任务：评估皮肤表现，如皮疹、瘙痒、色素沉着等。
                    重点：分析皮肤症状是否为系统性疾病（如红斑狼疮、过敏）的外在表现。
                    建议：提供皮肤活检、过敏原测试或外用药物建议。
                    输出：仅列出可能的皮肤问题及推荐措施。
                    患者报告：{medical_report}
                """,
                "肿瘤科医生": """
                    请以肿瘤科医生身份分析以下患者报告。
                    任务：排查潜在的恶性肿瘤风险或副肿瘤综合征。
                    重点：关注不明原因的体重下降、肿块、持续疼痛或异常的肿瘤标志物。
                    建议：提供肿瘤筛查、影像学复查或病理诊断建议。
                    输出：仅列出潜在的肿瘤风险及排查建议。
                    患者报告：{medical_report}
                """,
                "血液科医生": """
                    请以血液科医生身份分析以下患者报告。
                    任务：评估血常规异常、凝血功能障碍或淋巴系统疾病。
                    重点：分析贫血、出血倾向、白细胞异常等血液学指标。
                    建议：提供骨髓穿刺、凝血因子检测或血液专科检查建议。
                    输出：仅列出可能的血液系统问题及推荐措施。
                    患者报告：{medical_report}
                """,
                "肾脏科医生": """
                    请以肾脏科医生身份分析以下患者报告。
                    任务：评估肾功能、尿液异常及电解质平衡。
                    重点：关注蛋白尿、血尿、水肿或肾功能衰竭迹象。
                    建议：提供肾脏超声、24小时尿蛋白定量或肾活检建议。
                    输出：仅列出可能的肾脏问题及推荐措施。
                    患者报告：{medical_report}
                """,
                "风湿科医生": """
                    请以风湿科医生身份分析以下患者报告。
                    任务：评估关节疼痛、肌肉酸痛及自身免疫性结缔组织病。
                    重点：分析关节炎、血管炎或干燥综合征等风湿免疫特征。
                    建议：提供自身抗体谱、关节影像学或抗风湿治疗建议。
                    输出：仅列出可能的风湿免疫问题及推荐措施。
                    患者报告：{medical_report}
                """
            }
            templates = templates[self.role]  # 按角色选择对应的提示模板
        return PromptTemplate.from_template(templates)  # 使用 PromptTemplate 生成可格式化模板

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

class 心脏科医生(Agent):  # 定义心脏科智能体，继承基类 Agent
    def __init__(self, medical_report):  # 初始化时仅需传入医疗报告
        super().__init__(medical_report, "心脏科医生")  # 指定角色为心脏科

class 心理医生(Agent):  # 定义心理科智能体
    def __init__(self, medical_report):
        super().__init__(medical_report, "心理医生")  # 指定角色为心理科

class 精神科医生(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "精神科医生")

class 肺科医生(Agent):  # 定义肺科智能体
    def __init__(self, medical_report):
        super().__init__(medical_report, "肺科医生")  # 指定角色为肺科

class 多学科团队(Agent):  # 定义多学科团队智能体
    def __init__(self, cardiologist_report, psychologist_report, psychiatrist_report, pulmonologist_report,
                 neurologist_report, endocrinologist_report, immunologist_report,
                 gastroenterologist_report, dermatologist_report, oncologist_report,
                 hematologist_report, nephrologist_report, rheumatologist_report):  # 需要多份专科报告
        extra_info = {  # 将所有报告装入字典，供模板调用
            "cardiologist_report": cardiologist_report,
            "psychologist_report": psychologist_report,
            "psychiatrist_report": psychiatrist_report,
            "pulmonologist_report": pulmonologist_report,
            "neurologist_report": neurologist_report,
            "endocrinologist_report": endocrinologist_report,
            "immunologist_report": immunologist_report,
            "gastroenterologist_report": gastroenterologist_report,
            "dermatologist_report": dermatologist_report,
            "oncologist_report": oncologist_report,
            "hematologist_report": hematologist_report,
            "nephrologist_report": nephrologist_report,
            "rheumatologist_report": rheumatologist_report
        }
        super().__init__(role="多学科团队", extra_info=extra_info)  # 初始化时仅传角色与额外信息

    def create_prompt_template(self):  # 重写 create_prompt_template 以包含新科室
        # 动态构建提示模板，只包含有报告的科室
        active_reports = []
        active_specialists = []
        
        # 映射英文键名到中文科室名，用于 Prompt 展示
        key_map = {
            "cardiologist_report": "心脏科",
            "psychologist_report": "心理科",
            "psychiatrist_report": "精神科",
            "pulmonologist_report": "肺科",
            "neurologist_report": "神经科",
            "endocrinologist_report": "内分泌科",
            "immunologist_report": "免疫科",
            "gastroenterologist_report": "消化科",
            "dermatologist_report": "皮肤科",
            "oncologist_report": "肿瘤科",
            "hematologist_report": "血液科",
            "nephrologist_report": "肾脏科",
            "rheumatologist_report": "风湿科"
        }

        for key, value in self.extra_info.items():
            if value and value.strip():
                dept_name = key_map.get(key, "未知科室")
                active_reports.append(f"{dept_name}报告：{value}")
                active_specialists.append(dept_name)

        reports_text = "\n".join(active_reports)
        specialists_text = "、".join(active_specialists)

        templates = f"""
            请以多学科医疗团队的身份进行推理。
            你将获得以下专科医生提供的患者报告：{specialists_text}。
            任务：综合全部报告，列出 3 个可能的健康问题，并逐条说明对应理由与后续建议。
            输出格式：仅返回 3 个要点的列表，每个要点包含“问题 + 理由/建议”。

            {reports_text}
        """
        return PromptTemplate.from_template(templates)

    async def run_react_async(self, max_steps: int = 2):
        history = []
        observation = None

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
                "   每个元素需包含 name、reason、suggestion 和 department_evidence（多个科室字段，至少包括 cardiology、psychology、psychiatry、pulmonology、neurology、endocrinology、immunology、gastroenterology、dermatology、oncology、hematology、nephrology、rheumatology）。"\
                "2）当 last_observation 中已经包含非空的 issues 时，你必须设置 tool = null, args = null，"\
                "   并在 final_answer 中给出面向患者的中文总结，不要再调用任何工具。"\
            )

            state = {
                "history": history,
                "last_observation": observation,
                "reports": {
                    "cardiology": self.extra_info.get("cardiologist_report", ""),
                    "psychology": self.extra_info.get("psychologist_report", ""),
                    "psychiatry": self.extra_info.get("psychiatrist_report", ""),
                    "pulmonology": self.extra_info.get("pulmonologist_report", ""),
                    "neurology": self.extra_info.get("neurologist_report", ""),
                    "endocrinology": self.extra_info.get("endocrinologist_report", ""),
                    "immunology": self.extra_info.get("immunologist_report", ""),
                    "gastroenterology": self.extra_info.get("gastroenterologist_report", ""),
                    "dermatology": self.extra_info.get("dermatologist_report", ""),
                    "oncology": self.extra_info.get("oncologist_report", ""),
                    "hematology": self.extra_info.get("hematologist_report", ""),
                    "nephrology": self.extra_info.get("nephrologist_report", ""),
                    "rheumatology": self.extra_info.get("rheumatologist_report", ""),
                },
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

                            dept = issue.get("department_evidence") or {}
                            if not isinstance(dept, dict):
                                dept = {}

                            cardiology = str(dept.get("cardiology", "")).strip()
                            psychology = str(dept.get("psychology", "")).strip()
                            psychiatry = str(dept.get("psychiatry", "")).strip()
                            pulmonology = str(dept.get("pulmonology", "")).strip()
                            neurology = str(dept.get("neurology", "")).strip()
                            endocrinology = str(dept.get("endocrinology", "")).strip()
                            immunology = str(dept.get("immunology", "")).strip()
                            gastroenterology = str(dept.get("gastroenterology", "")).strip()
                            dermatology = str(dept.get("dermatology", "")).strip()
                            oncology = str(dept.get("oncology", "")).strip()
                            hematology = str(dept.get("hematology", "")).strip()
                            nephrology = str(dept.get("nephrology", "")).strip()
                            rheumatology = str(dept.get("rheumatology", "")).strip()

                            lines: list[str] = [f"#### {idx}. {name}"]
                            if reason:
                                lines.append(f"- 理由：{reason}")
                            if suggestion:
                                lines.append(f"- 建议：{suggestion}")

                            # 如果存在任意科室观点，则按 Markdown 子项目形式展示
                            # if any([cardiology, psychology, psychiatry, pulmonology, neurology, endocrinology, immunology, gastroenterology, dermatology, oncology, hematology, nephrology, rheumatology]):
                            #     lines.append("- 各科室观点：")
                            #     if cardiology:
                            #         lines.append(f"  - 心脏科：{cardiology}")
                            #     if psychology:
                            #         lines.append(f"  - 心理科：{psychology}")
                            #     if psychiatry:
                            #         lines.append(f"  - 精神科：{psychiatry}")
                            #     if pulmonology:
                            #         lines.append(f"  - 肺科：{pulmonology}")
                            #     if neurology:
                            #         lines.append(f"  - 神经科：{neurology}")
                            #     if endocrinology:
                            #         lines.append(f"  - 内分泌科：{endocrinology}")
                            #     if immunology:
                            #         lines.append(f"  - 免疫科：{immunology}")
                            #     if gastroenterology:
                            #         lines.append(f"  - 消化科：{gastroenterology}")
                            #     if dermatology:
                            #         lines.append(f"  - 皮肤科：{dermatology}")
                            #     if oncology:
                            #         lines.append(f"  - 肿瘤科：{oncology}")
                            #     if hematology:
                            #         lines.append(f"  - 血液科：{hematology}")
                            #     if nephrology:
                            #         lines.append(f"  - 肾脏科：{nephrology}")
                            #     if rheumatology:
                            #         lines.append(f"  - 风湿科：{rheumatology}")

                            block = "\n".join(lines)
                            parts.append(block)

                        if parts:
                            return "\n\n".join(parts)
            else:
                observation = None

        return None

class 神经科医生(Agent):  # 定义神经科智能体
    def __init__(self, medical_report):
        super().__init__(medical_report, "神经科医生")

class 内分泌科医生(Agent):  # 定义内分泌科智能体
    def __init__(self, medical_report):
        super().__init__(medical_report, "内分泌科医生")

class 免疫科医生(Agent):  # 定义免疫科智能体
    def __init__(self, medical_report):
        super().__init__(medical_report, "免疫科医生")

class 消化科医生(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "消化科医生")

class 皮肤科医生(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "皮肤科医生")

class 肿瘤科医生(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "肿瘤科医生")

class 血液科医生(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "血液科医生")

class 肾脏科医生(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "肾脏科医生")

class 风湿科医生(Agent):
    def __init__(self, medical_report):
        super().__init__(medical_report, "风湿科医生")
