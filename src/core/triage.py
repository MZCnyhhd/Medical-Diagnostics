"""
智能分诊模块：根据医疗报告自动选择相关专科医生

本模块使用 LLM 分析患者医疗报告，从所有可用专科中智能选择最相关的科室，
避免不必要的专科参与会诊，提升诊断效率。
"""

from langchain_core.prompts import PromptTemplate
from src.services.llm import get_chat_model
import json
from src.services.logging import log_info, log_error


async def triage_specialists(medical_report: str, available_specialists: list[str]) -> list[str]:
    """
    智能分诊：根据医疗报告内容，从可用专科医生列表中选择最相关的科室
    
    工作流程：
    1. 构建分诊 Prompt，要求 LLM 分析报告并选择相关专科
    2. 调用 LLM 获取 JSON 格式的分诊结果
    3. 解析并验证返回的专科列表
    4. 如果解析失败，降级为返回所有专科（兜底策略）
    
    Args:
        medical_report (str): 患者的医疗报告文本
        available_specialists (list[str]): 所有可用的专科医生列表，例如：
            ["心脏科医生", "消化科医生", "心理医生", ...]
    
    Returns:
        list[str]: 选中的专科医生列表，例如：["消化科医生", "心理医生"]
        如果分诊失败，返回所有可用专科（兜底策略）
    
    Example:
        >>> specialists = await triage_specialists("患者主诉腹痛、焦虑", 
        ...     ["消化科医生", "心理医生", "心脏科医生"])
        >>> print(specialists)
        ['消化科医生', '心理医生']
    """
    # 获取 LLM 实例（支持 Qwen/OpenAI/Gemini 等）
    llm = get_chat_model()
    
    # 构建分诊 Prompt 模板
    # 要求 LLM 扮演全科分诊医生，根据症状选择相关专科
    prompt_template = """
    你是一位经验丰富的全科分诊医生。
    请阅读以下患者的医疗报告，并从给定的专科医生列表中，挑选出最需要参与会诊的科室。
    
    可用专科列表：{specialists}
    
    患者报告：
    {report}
    
    请遵循以下原则：
    1. 选择与症状最直接相关的科室（例如腹痛选消化科，皮疹选皮肤科）。
    2. 如果病情复杂，可选择多个相关科室（通常 2-5 个）。
    3. 必须只返回一个 JSON 数组，包含选中的科室名称字符串。不要返回任何其他文字。
    
    示例输出：
    ["消化科医生", "心理医生"]
    """
    
    # 使用 LangChain 的 LCEL 语法构建调用链
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    try:
        # 异步调用 LLM，传入专科列表和医疗报告
        response = await chain.ainvoke({
            "specialists": ", ".join(available_specialists),
            "report": medical_report
        })
        
        # 提取 LLM 返回的文本内容
        content = getattr(response, "content", str(response))
        
        import re
        
        # ========== JSON 解析与清洗 ==========
        # LLM 可能返回包含额外文本的 JSON，需要清洗
        
        # 1. 移除某些模型可能添加的推理标签（如 Qwen 的 <think>）
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # 2. 移除 markdown 代码块标记（如 ```json ... ```）
        clean_content = clean_content.replace("```json", "").replace("```", "").strip()

        # 3. 寻找 JSON 数组的起始和结束位置
        # 使用 find() 和 rfind() 定位第一个 '[' 和最后一个 ']'
        start_index = clean_content.find('[')
        end_index = clean_content.rfind(']')
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            # 找到了数组边界，尝试解析
            try:
                json_str = clean_content[start_index:end_index+1]
                selected_specialists = json.loads(json_str)
            except json.JSONDecodeError:
                # 如果截取后解析失败，尝试整个字符串解析作为兜底
                try:
                    selected_specialists = json.loads(clean_content)
                except json.JSONDecodeError:
                    # 再次尝试修复常见的单引号问题（Python 风格而非 JSON 风格）
                    try:
                        import ast
                        selected_specialists = ast.literal_eval(json_str)
                    except:
                        selected_specialists = None
        else:
            # 找不到数组边界，尝试直接解析整个字符串
            try:
                selected_specialists = json.loads(clean_content)
            except:
                selected_specialists = None
        
        # ========== 验证与过滤 ==========
        if selected_specialists is None:
            # 解析完全失败，记录错误并返回所有专科（兜底策略）
            log_error(f"无法解析分诊结果 JSON: {clean_content}")
            return available_specialists
        
        if isinstance(selected_specialists, list):
            # 过滤掉不在可用列表中的专科（防止 LLM 返回不存在的专科名称）
            valid_specialists = [s for s in selected_specialists if s in available_specialists]
            log_info(f"分诊结果：{valid_specialists}")
            return valid_specialists
        else:
            # LLM 返回的不是列表格式，记录错误
            log_error("分诊模型返回格式错误，非列表。")
            return available_specialists  # 降级：返回所有专科
            
    except Exception as e:
        # 任何异常都记录并返回所有专科（确保系统不会因分诊失败而崩溃）
        log_error(f"分诊过程发生错误: {e}")
        return available_specialists  # 降级：返回所有专科
