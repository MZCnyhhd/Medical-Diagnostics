"""
智能分诊模块：根据医疗报告自动选择相关专科医生
==============================================

本模块实现了智能分诊功能，类似于医院的分诊台。

分诊的作用：
在真实医院中，分诊台会根据患者的主诉快速判断应该去哪个科室就诊。
本模块使用 LLM 模拟这个过程：
1. 分析患者的医疗报告内容
2. 理解症状涉及的身体系统
3. 从所有可用专科中选择最相关的几个

为什么需要分诊：
- 提高效率：避免所有专科都参与每个病例的诊断
- 提高相关性：确保参与诊断的专科与病情相关
- 节省资源：减少不必要的 LLM API 调用

分诊策略：
- 选择与症状最直接相关的科室
- 复杂病例可选择多个相关科室（通常 2-5 个）
- 如果分诊失败，降级为使用所有专科（兜底策略）

核心函数：
- triage_specialists：根据医疗报告选择相关专科医生

依赖关系：
- src/services/llm.py：获取 LLM 模型实例
- src/tools/common.py：JSON 响应清洗工具
"""

# ==================== LangChain 导入 ====================
# PromptTemplate：LangChain 的提示词模板类，支持变量插值
from langchain_core.prompts import PromptTemplate

# ==================== 项目内部模块导入 ====================
# LLM 工厂函数：获取配置好的 Chat 模型实例
from src.services.llm import get_chat_model

# ==================== 标准库导入 ====================
# json：用于解析 LLM 返回的 JSON 格式结果
import json
# ast：用于安全地解析 Python 字面量（处理单引号 JSON）
import ast

# ==================== 项目工具导入 ====================
# 日志工具函数
from src.services.logging import log_info, log_error
# LLM JSON 响应清洗工具
from src.tools.common import clean_llm_json_response


async def triage_specialists(medical_report: str, available_specialists: list[str]) -> list[str]:
    """
    智能分诊：根据医疗报告内容，从可用专科医生列表中选择最相关的科室
    
    这是一个关键的预处理步骤，在执行多学科会诊之前：
    1. 分析医疗报告中描述的症状和体征
    2. 判断这些症状涉及哪些身体系统
    3. 选择最相关的专科参与诊断
    
    工作流程详解：
    ==============
    
    第一步：构建分诊 Prompt
    - 将可用专科列表和医疗报告传入 Prompt 模板
    - Prompt 指导 LLM 扮演全科分诊医生的角色
    - 要求输出 JSON 数组格式的专科列表
    
    第二步：调用 LLM 进行分析
    - 使用 LangChain 的 LCEL 语法构建调用链
    - 异步调用 LLM 获取分析结果
    
    第三步：解析和清洗响应
    - LLM 返回的文本可能包含额外内容（Markdown 标记等）
    - 使用多种策略提取 JSON 数组
    - 处理常见的格式问题（单引号、多余文本等）
    
    第四步：验证和过滤
    - 验证提取的是否是有效的列表
    - 过滤掉不在可用列表中的专科名称
    - 防止 LLM 返回不存在的专科
    
    Args:
        medical_report (str): 患者的医疗报告文本
            - 包含患者主诉、症状描述、检查结果等
            - 例如："患者，男，45岁，主诉胸闷、气短一周..."
        
        available_specialists (list[str]): 所有可用的专科医生列表
            - 从 prompts.yaml 配置文件加载
            - 例如：["心脏科医生", "消化科医生", "心理医生", ...]
    
    Returns:
        list[str]: 选中的专科医生列表
            - 通常包含 2-5 个最相关的专科
            - 例如：["心脏科医生", "肺科医生"]
            - 如果分诊失败，返回所有可用专科（兜底策略）
    
    兜底策略：
    如果分诊过程中发生任何错误（LLM 调用失败、JSON 解析失败等），
    会返回所有可用专科，确保诊断流程不会因分诊失败而中断。
    
    使用示例：
    ```python
    # 可用专科列表
    specialists = ["消化科医生", "心理医生", "心脏科医生", "肺科医生"]
    
    # 患者报告
    report = "患者主诉腹痛、焦虑一周，伴有食欲下降"
    
    # 执行分诊
    selected = await triage_specialists(report, specialists)
    print(selected)  # 输出: ['消化科医生', '心理医生']
    ```
    """
    # ========== 获取 LLM 实例 ==========
    # 获取配置好的 Chat 模型（支持 Qwen/OpenAI/Gemini 等）
    # 会自动根据环境变量选择模型并配置容灾
    llm = get_chat_model()
    
    # ========== 构建分诊 Prompt 模板 ==========
    # 这个 Prompt 指导 LLM 如何进行分诊决策
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
    # PromptTemplate | llm 会创建一个 RunnableSequence
    # 调用时会自动：格式化 Prompt → 调用 LLM → 返回结果
    prompt = PromptTemplate.from_template(prompt_template)
    chain = prompt | llm
    
    try:
        # ========== 异步调用 LLM ==========
        # 传入变量：专科列表和医疗报告
        # specialists 用逗号连接成字符串，方便阅读
        response = await chain.ainvoke({
            "specialists": ", ".join(available_specialists),
            "report": medical_report
        })
        
        # 提取 LLM 返回的文本内容
        # LangChain 返回的是消息对象，需要通过 content 属性获取文本
        content = getattr(response, "content", str(response))
        
        # ========== JSON 解析与清洗 ==========
        # LLM 可能返回包含额外文本的 JSON，需要清洗
        # 例如：```json\n["消化科医生"]\n``` 或 "我选择：["消化科医生"]"
        
        # 第一步：使用通用清洗函数
        # 移除 <think> 标签、Markdown 代码块标记等
        clean_content = clean_llm_json_response(content)

        # 第二步：寻找 JSON 数组的起始和结束位置
        # 使用 find() 定位第一个 '['
        start_index = clean_content.find('[')
        # 使用 rfind() 定位最后一个 ']'
        end_index = clean_content.rfind(']')
        
        # 第三步：尝试解析 JSON
        if start_index != -1 and end_index != -1 and end_index > start_index:
            # 找到了数组边界，截取 JSON 字符串
            try:
                json_str = clean_content[start_index:end_index+1]
                # 尝试使用标准 json 库解析
                selected_specialists = json.loads(json_str)
            except json.JSONDecodeError:
                # 截取的部分解析失败，尝试解析整个字符串
                try:
                    selected_specialists = json.loads(clean_content)
                except json.JSONDecodeError:
                    # 再次尝试修复常见的单引号问题
                    # Python 风格的列表使用单引号，而 JSON 要求双引号
                    # 例如：['消化科医生'] → ["消化科医生"]
                    try:
                        # ast.literal_eval 可以安全地解析 Python 字面量
                        selected_specialists = ast.literal_eval(json_str)
                    except:
                        # 所有解析方法都失败
                        selected_specialists = None
        else:
            # 找不到数组边界（没有 '[' 或 ']'）
            # 尝试直接解析整个字符串
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
            # 解析成功，是列表类型
            # 过滤掉不在可用列表中的专科名称
            # 这防止了 LLM 返回不存在的专科（如 "骨科医生" 但配置中没有）
            valid_specialists = [s for s in selected_specialists if s in available_specialists]
            # 记录分诊结果
            log_info(f"分诊结果：{valid_specialists}")
            return valid_specialists
        else:
            # LLM 返回的不是列表格式（可能是字典或字符串）
            log_error("分诊模型返回格式错误，非列表。")
            return available_specialists  # 降级：返回所有专科
            
    except Exception as e:
        # 任何异常都记录并返回所有专科
        # 这确保系统不会因分诊失败而崩溃
        log_error(f"分诊过程发生错误: {e}")
        return available_specialists  # 降级：返回所有专科
