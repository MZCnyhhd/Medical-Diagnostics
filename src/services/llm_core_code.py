# [导入模块] ############################################################################################################
# [标准库 | Standard Libraries] =========================================================================================
import os                                                               # 操作系统接口：环境变量读取
import base64                                                           # Base64 编解码：图片数据传输
# [第三方库 | Third-party Libraries] ====================================================================================
import requests                                                         # HTTP 客户端：API 调用
import streamlit as st                                                  # Streamlit：缓存装饰器
from langchain_community.chat_models import ChatTongyi                  # 通义千问模型
from langchain_openai import ChatOpenAI                                 # OpenAI GPT 模型
from langchain_google_genai import ChatGoogleGenerativeAI               # Google Gemini 模型
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace  # HuggingFace 本地模型
# [内部模块 | Internal Modules] =========================================================================================
from src.services.logging import log_info, log_warn, log_error          # 统一日志服务
# [定义函数] ############################################################################################################
# [内部-加载本地模型] =====================================================================================================
@st.cache_resource
def _load_local_model(model_path: str):
    """
    加载本地 HuggingFace 模型（带 Streamlit 缓存）。
    :param model_path: 模型路径
    :return: ChatHuggingFace 实例，加载失败返回 None
    """
    # [step1] 延迟导入 transformers 依赖
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
    except ImportError as e:
        log_warn(f"本地模型依赖缺失: {e}")
        return None
    # [step2] 加载 tokenizer 和模型
    log_info(f"正在加载本地模型: {model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True
        )
        log_info(f"模型已加载到设备: {model.device}")
    except Exception as e:
        log_warn(f"加载本地模型失败: {e}")
        return None
    # [step3] 构建 pipeline 并包装为 LangChain 兼容对象
    pipe = pipeline(
        "text-generation", model=model, tokenizer=tokenizer,
        max_new_tokens=2048, temperature=0.1, top_p=0.95,
        repetition_penalty=1.15, return_full_text=False
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    log_info("本地模型加载成功！")
    return ChatHuggingFace(llm=local_llm)
# [内部-初始化可用模型] ===================================================================================================
def _init_available_models(temperature: float) -> tuple[dict, dict]:
    """
    初始化所有可用的 LLM 模型。
    :param temperature: 模型温度参数
    :return: (可用模型字典, 初始化错误字典)
    """
    available_models = {}
    init_errors = {}
    # [step1] 尝试加载本地模型
    local_model_path = os.getenv("LOCAL_MODEL_PATH")
    if local_model_path and os.path.exists(local_model_path):
        local_model = _load_local_model(local_model_path)
        if local_model:
            available_models["local"] = local_model
    # [step2] 尝试初始化 Qwen
    if os.getenv("DASHSCOPE_API_KEY"):
        try:
            model_name = os.getenv("QWEN_MODEL", "qwen-max")
            available_models["qwen"] = ChatTongyi(model=model_name, temperature=temperature)
        except Exception as e:
            init_errors["qwen"] = str(e)
            log_warn(f"初始化 Qwen 失败: {e}")
    # [step3] 尝试初始化 OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            available_models["openai"] = ChatOpenAI(
                model=model_name, temperature=temperature, api_key=os.getenv("OPENAI_API_KEY")
            )
        except Exception as e:
            init_errors["openai"] = str(e)
            log_warn(f"初始化 OpenAI 失败: {e}")
    # [step4] 尝试初始化 Gemini
    if os.getenv("GOOGLE_API_KEY"):
        try:
            model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
            available_models["gemini"] = ChatGoogleGenerativeAI(
                model=model_name, temperature=temperature, google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        except Exception as e:
            init_errors["gemini"] = str(e)
            log_warn(f"初始化 Gemini 失败: {e}")
    return available_models, init_errors
# [内部-选择模型提供商] ===================================================================================================
def _select_provider(
    override_provider: str | None,
    available_models: dict,
    init_errors: dict,
    priority_order: list[str]
) -> str:
    """
    根据优先级选择可用的模型提供商。
    :param override_provider: 强制指定的提供商
    :param available_models: 可用模型字典
    :param init_errors: 初始化错误字典
    :param priority_order: 优先级列表
    :return: 选定的提供商名称
    """
    # [step1] 确定目标提供商
    provider = override_provider.lower() if override_provider else os.getenv("LLM_PROVIDER", "qwen").lower()
    # [step2] 若目标可用则直接返回
    if provider in available_models:
        return provider
    # [step3] 强制指定但不可用时抛出异常
    if override_provider:
        error_detail = init_errors.get(provider, "未知的初始化错误 (可能是 Key 未配置)")
        raise ValueError(f"无法加载指定的模型 '{provider}'。原因: {error_detail}")
    # [step4] 按优先级查找可用模型
    log_warn(f"指定的 LLM_PROVIDER='{provider}' 不可用。")
    for p in priority_order:
        if p in available_models:
            log_warn(f"自动切换到优先级最高的可用模型: '{p}'。")
            return p
    # [step5] 兜底：随机选择一个可用模型
    if available_models:
        fallback = list(available_models.keys())[0]
        log_warn(f"优先级列表中的模型均不可用，随机选择可用模型: '{fallback}'。")
        return fallback
    # [step6] 完全无可用模型时返回默认值
    return "qwen"
# [外部-获取聊天模型] ==================================================================================================
def get_chat_model(override_provider: str | None = None):
    """
    获取 LLM 聊天模型（支持多提供商、自动降级和备用链）。
    :param override_provider: 强制指定的提供商名称
    :return: LangChain 兼容的聊天模型实例
    """
    # [step1] 配置温度参数
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    if temperature == 0:
        temperature = 0.01
    # [step2] 初始化所有可用模型
    available_models, init_errors = _init_available_models(temperature)
    priority_order = ["qwen", "openai", "gemini", "local"]
    # [step3] 选择模型提供商
    provider = _select_provider(override_provider, available_models, init_errors, priority_order)
    # [step4] 处理完全无模型可用的情况
    if provider not in available_models:
        log_warn("未检测到任何有效的 API Key 配置！将尝试返回默认 Qwen 配置（可能会报错）。")
        return ChatTongyi(model="qwen-max", temperature=temperature)
    # [step5] 构建主模型和备用链
    primary_model = available_models[provider]
    fallback_models = [available_models[p] for p in priority_order if p != provider and p in available_models]
    # [step6] 记录配置日志（去重）
    if not hasattr(get_chat_model, "_logged_configs"):
        get_chat_model._logged_configs = set()
    fallback_names = [p for p in priority_order if p != provider and p in available_models]
    log_msg = f"LLM 配置: 主模型={provider}, 备用模型链={fallback_names}" if fallback_models else f"LLM 配置: 主模型={provider}, 无可用备用模型。"
    if log_msg not in get_chat_model._logged_configs:
        log_info(log_msg)
        get_chat_model._logged_configs.add(log_msg)
    # [step7] 返回带备用链的模型或单独模型
    return primary_model.with_fallbacks(fallback_models) if fallback_models else primary_model
# [内部-调用视觉模型API] ==================================================================================================
def _call_vision_api(
        api_name: str,
        api_url: str,
        headers: dict,
        payload: dict,
        timeout: int = 60
) -> str | None:
    """
    统一的视觉模型 API 调用封装。
    :param api_name: API 名称（用于日志）
    :param api_url: API 端点 URL
    :param headers: 请求头
    :param payload: 请求体
    :param timeout: 超时时间
    :return: 分析结果文本，失败返回 None
    """
    # [step1] 记录调用日志
    log_info(f"使用 {api_name} 分析医疗图片...")
    try:
        # [step2] 发送 POST 请求
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        # [step3] 检查响应状态码
        if response.status_code != 200:
            log_warn(f"{api_name} 请求失败: {response.status_code} - {response.text}")
            return None
        # [step4] 解析并返回 JSON 响应
        return response.json()
    except Exception as e:
        # [step5] 异常处理：记录警告并返回 None
        log_warn(f"{api_name} 分析失败: {e}")
        return None
# [内部-解析Qwen/OpenAI响应] =============================================================================================
def _parse_openai_style_response(result: dict | None) -> str | None:
    """
    解析 OpenAI 风格的 API 响应。
    :param result: API 响应 JSON
    :return: 内容文本，解析失败返回 None
    """
    # [step1] 卫语句：空结果直接返回 None
    if not result:
        return None
    # [step2] 链式提取：choices[0].message.content
    return result.get("choices", [{}])[0].get("message", {}).get("content", "") or None
# [内部-解析Gemini响应] ==================================================================================================
def _parse_gemini_response(result: dict | None) -> str | None:
    """
    解析 Google Gemini API 响应。
    :param result: API 响应 JSON
    :return: 内容文本，解析失败返回 None
    """
    # [step1] 卫语句：空结果直接返回 None
    if not result:
        return None
    # [step2] 链式提取：candidates[0].content.parts[0].text
    return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "") or None
# [内部-Qwen图片分析] ====================================================================================================
def _analyze_by_qwen(image_base64: str, prompt: str) -> str | None:
    """使用 Qwen-VL 分析图片"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    result = _call_vision_api(
        "Qwen-VL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        {"model": "qwen-vl-max", "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]}], "max_tokens": 2000}
    )
    if content := _parse_openai_style_response(result):
        log_info("Qwen-VL 图片分析成功")
        return content
    return None
# [内部-OpenAI图片分析] ==================================================================================================
def _analyze_by_openai(image_base64: str, prompt: str) -> str | None:
    """使用 OpenAI GPT-4o 分析图片"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    result = _call_vision_api(
        "OpenAI GPT-4 Vision", "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        {"model": "gpt-4o", "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]}], "max_tokens": 2000}
    )
    if content := _parse_openai_style_response(result):
        log_info("OpenAI GPT-4 Vision 图片分析成功")
        return content
    return None
# [内部-Gemini图片分析] ==================================================================================================
def _analyze_by_gemini(image_base64: str, prompt: str) -> str | None:
    """使用 Gemini Vision 分析图片"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    result = _call_vision_api(
        "Google Gemini Vision",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
        {"Content-Type": "application/json"},
        {"contents": [{"parts": [
            {"text": prompt},
            {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
        ]}]}
    )
    if content := _parse_gemini_response(result):
        log_info("Google Gemini Vision 图片分析成功")
        return content
    return None
# [外部-分析医疗图片] =====================================================================================================
def analyze_medical_image(image_bytes: bytes) -> str:
    """
    使用视觉模型分析医疗图片。
    按优先级尝试：Qwen-VL -> GPT-4o -> Gemini Vision。
    :param image_bytes: 图片字节数据
    :return: 分析结果文本，全部失败返回空字符串
    """
    # [step1] 初始化：Base64 编码与提示词定义
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    analysis_prompt = """
    你是一位专业的医学影像分析专家。请仔细分析这张医疗图片，并提供详细的文字描述报告。
    请按以下格式输出：
    ## 图像类型
    ## 图像描述
    ## 关键发现
    ## 初步印象
    请确保描述准确、专业，使用标准的医学术语。
    """
    # [step2] 尝试 Qwen-VL 分析
    if content := _analyze_by_qwen(image_base64, analysis_prompt):
        return content
    # [step3] 尝试 OpenAI 分析
    if content := _analyze_by_openai(image_base64, analysis_prompt):
        return content
    # [step4] 尝试 Gemini 分析
    if content := _analyze_by_gemini(image_base64, analysis_prompt):
        return content
    # [step5] 最终处理：记录失败并返回空
    log_error("所有视觉模型均无法分析图片")
    return ""