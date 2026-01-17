"""
模块名称: Smart Triage (智能分诊)
功能描述:

    基于 LLM 分析用户的医疗报告，自动推荐最合适的专科医生列表。
    充当系统的"导诊台"角色，确保诊断任务分配给具备相关专业知识的 Agent。

设计理念:

    1.  **少样本学习 (Few-Shot)**: Prompt 中包含示例，引导 LLM 输出标准 JSON 格式。
    2.  **鲁棒性解析**: 针对 LLM 可能输出的不规范 JSON (如包含 Markdown 代码块)，内置清理和修复逻辑。
    3.  **动态扩展**: 专科列表通过参数传入，不硬编码在 Prompt 中，便于后续扩展科室。

线程安全性:

    - 无状态函数，线程安全。

依赖关系:

    - `src.services.llm`: 调用 LLM 进行推理。
    - `src.core.settings`: 获取默认模型配置。
"""

import json
import ast
import re
# [第三方库 | Third-party Libraries] ====================================================================================
from langchain_core.prompts import PromptTemplate                      # 提示词模板
# [内部模块 | Internal Modules] =========================================================================================
from src.services.llm import get_chat_model                            # 模型工厂
from src.services.logging import log_info, log_error                   # 统一日志服务
from src.tools.common import clean_llm_json_response                   # LLM 响应清洗
# [定义函数] ############################################################################################################
# [异步-外部-智能分诊] ====================================================================================================
async def triage_specialists(medical_report: str, available_specialists: list[str]) -> list[str]:
    """
    智能分诊：根据医疗报告自动选择相关专科医生。
    :param medical_report: 医疗报告文本
    :param available_specialists: 可用专科列表
    :return: 选中的专科列表，失败时返回全部可用专科
    """
    # [step1] 构建 LLM 调用链
    llm = get_chat_model()
    prompt = _build_triage_prompt()
    chain = prompt | llm
    # [step2] 调用 LLM 获取分诊结果
    try:
        response = await chain.ainvoke({
            "specialists": ", ".join(available_specialists),
            "report": medical_report
        })
        content = getattr(response, "content", str(response))
        clean_content = clean_llm_json_response(content)
    except Exception as e:
        log_error(f"分诊过程发生错误: {e}")
        return available_specialists
    # [step3] 解析 JSON 数组
    selected_specialists = _parse_json_array(clean_content)
    if selected_specialists is None:
        log_error(f"无法解析分诊结果 JSON: {clean_content}")
        return available_specialists
    # [step4] 校验结果类型
    if not isinstance(selected_specialists, list):
        log_error("分诊模型返回格式错误，非列表。")
        return available_specialists
    # [step5] 过滤无效专科名称
    valid_specialists = [s for s in selected_specialists if s in available_specialists]
    log_info(f"分诊结果：{valid_specialists}")
    return valid_specialists
# [内部-提取JSON数组] ====================================================================================================
def _extract_json_array(text: str) -> str | None:
    """
    从文本中提取 JSON 数组部分。
    :param text: 原始文本
    :return: JSON 数组字符串，未找到返回 None
    """
    # [step1] 查找左方括号位置（第一个出现）
    start_index = text.find('[')
    # [step2] 查找右方括号位置（最后一个出现）
    end_index = text.rfind(']')
    # [step3] 校验边界有效性并提取子串
    if start_index != -1 and end_index != -1 and end_index > start_index:
        return text[start_index:end_index + 1]
    return None
# [内部-解析JSON数组] ====================================================================================================
def _parse_json_array(text: str) -> list | None:
    """
    稳健的 JSON 数组解析器（支持多种格式）。
    :param text: 待解析文本
    :return: 解析后的列表，失败返回 None
    """
    # [step1] 尝试提取并解析 JSON 数组片段
    json_str = _extract_json_array(text)
    if json_str:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
        # [step2] 尝试 ast.literal_eval 作为备用
        try:
            return ast.literal_eval(json_str)
        except:
            pass
    # [step3] 尝试直接解析整个文本
    try:
        return json.loads(text)
    except:
        return None
# [内部-构建分诊提示词] ===================================================================================================
def _build_triage_prompt() -> PromptTemplate:
    """
    构建分诊提示词模板。
    :return: LangChain PromptTemplate 对象
    """
    # [step1] 定义分诊提示词模板（含角色设定、任务说明、输出格式约束）
    template = """你是一位经验丰富的全科分诊医生。
请阅读以下患者的医疗报告，并从给定的专科医生列表中，挑选出最需要参与会诊的科室。
可用专科列表：{specialists}
患者报告：
{report}
请遵循以下原则：
1. 选择与症状最直接相关的科室（例如腹痛选消化科，皮疹选皮肤科）。
2. 如果病情复杂，可选择多个相关科室（通常 2-5 个）。
3. 必须只返回一个 JSON 数组，包含选中的科室名称字符串。不要返回任何其他文字。
示例输出：
["消化科医生", "心理医生"]"""
    # [step2] 构建并返回 LangChain 模板对象
    return PromptTemplate.from_template(template)
