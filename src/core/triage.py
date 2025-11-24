from langchain_core.prompts import PromptTemplate
from src.services.llm import get_chat_model
import json
from src.services.logging import log_info, log_error

async def triage_specialists(medical_report: str, available_specialists: list[str]) -> list[str]:
    """
    根据医疗报告内容，从可用专科医生列表中选择最相关的科室。
    """
    llm = get_chat_model()
    
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
    
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    try:
        response = await chain.ainvoke({
            "specialists": ", ".join(available_specialists),
            "report": medical_report
        })
        
        content = getattr(response, "content", str(response))
        
        import re
        
        # 尝试解析 JSON
        # 1. 移除 <think>...</think> 标签
        clean_content = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL).strip()
        
        # 2. 移除 markdown 代码块标记
        clean_content = clean_content.replace("```json", "").replace("```", "").strip()

        # 3. 寻找 JSON 数组的起始位置
        start_index = clean_content.find('[')
        end_index = clean_content.rfind(']')
        
        if start_index != -1 and end_index != -1 and end_index > start_index:
            try:
                json_str = clean_content[start_index:end_index+1]
                selected_specialists = json.loads(json_str)
            except json.JSONDecodeError:
                # 如果截取后解析失败，尝试整个字符串解析作为兜底
                try:
                    selected_specialists = json.loads(clean_content)
                except json.JSONDecodeError:
                     # 再次尝试修复常见的单引号问题
                    try:
                        import ast
                        selected_specialists = ast.literal_eval(json_str)
                    except:
                        selected_specialists = None
        else:
            # 找不到 []，尝试直接解析
            try:
                selected_specialists = json.loads(clean_content)
            except:
                selected_specialists = None
        
        if selected_specialists is None:
             log_error(f"无法解析分诊结果 JSON: {clean_content}")
             return available_specialists
        
        if isinstance(selected_specialists, list):
            # 过滤掉不在列表里的胡乱输出
            valid_specialists = [s for s in selected_specialists if s in available_specialists]
            log_info(f"分诊结果：{valid_specialists}")
            return valid_specialists
        else:
            log_error("分诊模型返回格式错误，非列表。")
            return available_specialists # 降级：全选
            
    except Exception as e:
        log_error(f"分诊过程发生错误: {e}")
        return available_specialists # 降级：全选
