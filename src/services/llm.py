# [导入模块] ############################################################################################################
# [标准库 | Standard Libraries] =========================================================================================
import os                                                              # 操作系统接口：环境变量与路径操作
import base64                                                          # Base64 编码：处理图片数据
import requests                                                        # HTTP 客户端：调用视觉 API
# [第三方库 | Third-party Libraries] ====================================================================================
import streamlit as st                                                 # Streamlit：Web 应用框架
from langchain_community.chat_models import ChatTongyi                 # LangChain 模型：通义千问
from langchain_openai import ChatOpenAI                                # LangChain 模型：OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI              # LangChain 模型：Google Gemini
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace # LangChain 模型：HuggingFace
# [内部模块 | Internal Modules] =========================================================================================
from src.services.logging import log_info, log_warn, log_error         # 统一日志服务：日志记录
# [定义函数] ############################################################################################################
# [内部-加载本地模型] =====================================================================================================
@st.cache_resource
def _load_local_model(model_path: str):
    """
    加载本地 HuggingFace 模型（带 Streamlit 缓存）。
    :param model_path: 本地模型路径
    :return: ChatHuggingFace 实例或 None
    """
    # [step1] 延迟导入本地依赖
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
    except ImportError as e:
        log_warn(f"本地模型依赖缺失: {e}")
        return None
    # [step2] 加载分词器和模型
    log_info(f"正在加载本地模型: {model_path} ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
        log_info(f"模型已加载到设备: {model.device}")
    except Exception as e:
        log_warn(f"加载本地模型失败: {e}")
        return None
    # [step3] 创建文本生成 Pipeline
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=2048, temperature=0.1, top_p=0.95, repetition_penalty=1.15, return_full_text=False)
    # [step4] 包装为 LangChain 接口
    local_llm = HuggingFacePipeline(pipeline=pipe)
    log_info("本地模型加载成功！")
    return ChatHuggingFace(llm=local_llm)
# [内部-初始化 Qwen] =====================================================================================================
def _init_qwen(temperature, available_models, init_errors):
    """初始化通义千问模型"""
    if not os.getenv("DASHSCOPE_API_KEY"):
        return
    try:
        model_name = os.getenv("QWEN_MODEL", "qwen-max")
        available_models["qwen"] = ChatTongyi(model=model_name, temperature=temperature)
    except Exception as e:
        init_errors["qwen"] = str(e)
        log_warn(f"初始化 Qwen 失败: {e}")
# [内部-初始化 Baichuan] ===================================================================================================
def _init_baichuan(temperature, available_models, init_errors):
    """初始化百川大模型 (Baichuan)"""
    if not os.getenv("BAICHUAN_API_KEY"):
        return
    try:
        model_name = os.getenv("BAICHUAN_MODEL", "Baichuan2-Turbo")
        # Baichuan API 兼容 OpenAI SDK
        # Base URL: https://api.baichuan-ai.com/v1
        available_models["baichuan"] = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=os.getenv("BAICHUAN_API_KEY"),
            base_url="https://api.baichuan-ai.com/v1"
        )
    except Exception as e:
        init_errors["baichuan"] = str(e)
        log_warn(f"初始化 Baichuan 失败: {e}")
# [内部-初始化可用模型] ===================================================================================================
def _init_available_models(temperature: float) -> tuple[dict, dict]:
    """
    初始化所有配置了 API Key 的模型。
    :param temperature: 模型生成温度
    :return: (可用模型字典, 初始化错误字典)
    """
    available_models, init_errors = {}, {}
    # [step1] 尝试加载本地模型
    local_path = os.getenv("LOCAL_MODEL_PATH")
    if local_path and os.path.exists(local_path):
        local_model = _load_local_model(local_path)
        if local_model:
            available_models["local"] = local_model
    # [step2] 初始化云端模型
    _init_qwen(temperature, available_models, init_errors)
    _init_baichuan(temperature, available_models, init_errors)
    return available_models, init_errors
# [内部-选择模型提供商] ===================================================================================================
def _select_provider(override_provider: str | None, available_models: dict, init_errors: dict, priority_order: list[str]) -> str:
    """
    选择最终使用的模型提供商，支持自动回退。
    :param override_provider: 强制指定的提供商
    :param available_models: 可用模型字典
    :param init_errors: 初始化错误字典
    :param priority_order: 优先级顺序
    :return: 选中的提供商名称
    """
    # [step1] 确定目标提供商
    provider = override_provider.lower() if override_provider else os.getenv("LLM_PROVIDER", "qwen").lower()
    if provider in available_models:
        return provider
    # [step2] 处理强制指定但不可用的情况
    if override_provider:
        error_detail = init_errors.get(provider, "未知的初始化错误 (可能是 Key 未配置)")
        raise ValueError(f"无法加载指定的模型 '{provider}'。原因: {error_detail}")
    # [step3] 自动切换到优先级最高的可用模型
    log_warn(f"指定的 LLM_PROVIDER='{provider}' 不可用。")
    for p in priority_order:
        if p in available_models:
            log_warn(f"自动切换到优先级最高的可用模型: '{p}'。")
            return p
    # [step4] 最后的保底逻辑
    if available_models:
        fallback = list(available_models.keys())[0]
        log_warn(f"优先级列表中的模型均不可用，随机选择可用模型: '{fallback}'。")
        return fallback
    return "qwen"
# [外部-获取对话模型] =====================================================================================================
def get_chat_model(override_provider: str | None = None):
    """
    获取配置好的 Chat 模型实例，支持自动容灾切换。
    :param override_provider: 强制指定的主模型提供商
    :return: 带 Fallback 机制的 Chat 模型实例
    """
    # [step1] 读取生成温度
    temp = float(os.getenv("LLM_TEMPERATURE", "0"))
    temperature = temp if temp != 0 else 0.01
    # [step2] 初始化所有可用模型
    available_models, init_errors = _init_available_models(temperature)
    priority_order = ["qwen", "baichuan", "openai", "gemini", "local"]
    # [step3] 选择主模型提供商
    provider = _select_provider(override_provider, available_models, init_errors, priority_order)
    # [step4] 卫语句：处理没有任何模型可用的极端情况
    if provider not in available_models:
        log_warn("未检测到任何有效的 API Key 配置！将尝试返回默认 Qwen 配置（可能会报错）。")
        return ChatTongyi(model="qwen-max", temperature=temperature)
    # [step5] 配置备用模型链 (Fallback)
    primary_model = available_models[provider]
    fallback_models = [available_models[p] for p in priority_order if p != provider and p in available_models]
    # [step6] 打印配置日志（仅限首次）
    if not hasattr(get_chat_model, "_logged_configs"):
        get_chat_model._logged_configs = set()
    fallback_names = [p for p in priority_order if p != provider and p in available_models]
    log_msg = f"LLM 配置: 主模型={provider}, 备用模型链={fallback_names}" if fallback_models else f"LLM 配置: 主模型={provider}, 无可用备用模型。"
    if log_msg not in get_chat_model._logged_configs:
        log_info(log_msg)
        get_chat_model._logged_configs.add(log_msg)
    # [step7] 返回带 Fallback 的模型
    return primary_model.with_fallbacks(fallback_models) if fallback_models else primary_model
# [内部-调用视觉 API] ====================================================================================================
def _call_vision_api(api_name: str, api_url: str, headers: dict, payload: dict, timeout: int = 60) -> str | None:
    """
    通用的视觉 API 调用封装。
    :param api_name: API 名称
    :param api_url: API 端点
    :param headers: 请求头
    :param payload: 请求体
    :param timeout: 超时时间
    :return: 响应 JSON 或 None
    """
    log_info(f"使用 {api_name} 分析医疗图片...")
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
        if response.status_code != 200:
            log_warn(f"{api_name} 请求失败: {response.status_code} - {response.text}")
            return None
        return response.json()
    except Exception as e:
        log_warn(f"{api_name} 分析失败: {e}")
        return None
# [内部-解析 OpenAI 风格响应] ============================================================================================
def _parse_openai_style_response(result: dict | None) -> str | None:
    """解析 OpenAI/Qwen 格式的响应"""
    if not result:
        return None
    return result.get("choices", [{}])[0].get("message", {}).get("content", "") or None
# [内部-解析 Gemini 响应] ================================================================================================
def _parse_gemini_response(result: dict | None) -> str | None:
    """解析 Gemini 格式的响应"""
    if not result:
        return None
    return result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "") or None
# [内部-使用 Qwen 分析图片] ==============================================================================================
def _analyze_by_qwen(image_base64: str, prompt: str) -> str | None:
    """使用 Qwen-VL 模型分析图片"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    result = _call_vision_api(
        "Qwen-VL", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        {"model": "qwen-vl-max", "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}], "max_tokens": 2000}
    )
    if content := _parse_openai_style_response(result):
        log_info("Qwen-VL 图片分析成功")
        return content
    return None
# [内部-使用 OpenAI 分析图片] ============================================================================================
def _analyze_by_openai(image_base64: str, prompt: str) -> str | None:
    """使用 OpenAI GPT-4o 模型分析图片"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    result = _call_vision_api(
        "OpenAI GPT-4 Vision", "https://api.openai.com/v1/chat/completions",
        {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        {"model": "gpt-4o", "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}], "max_tokens": 2000}
    )
    if content := _parse_openai_style_response(result):
        log_info("OpenAI GPT-4 Vision 图片分析成功")
        return content
    return None
# [内部-使用 Gemini 分析图片] ============================================================================================
def _analyze_by_gemini(image_base64: str, prompt: str) -> str | None:
    """使用 Google Gemini 1.5 Flash 模型分析图片"""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    result = _call_vision_api(
        "Google Gemini Vision",
        f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
        {"Content-Type": "application/json"},
        {"contents": [{"parts": [{"text": prompt}, {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}]}]}
    )
    if content := _parse_gemini_response(result):
        log_info("Google Gemini Vision 图片分析成功")
        return content
    return None
# [外部-分析医疗图片] =====================================================================================================
def analyze_medical_image(image_bytes: bytes) -> str:
    """
    使用视觉模型分析医疗图片，支持多模型回退。
    :param image_bytes: 图片字节数据
    :return: 医学文字描述报告
    """
    # [step1] 图片 Base64 编码
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    analysis_prompt = "请分析这张医疗图片并提供诊断建议。"
    # [step2] 依次尝试不同模型
    if content := _analyze_by_qwen(image_base64, analysis_prompt):
        return content
    if content := _analyze_by_openai(image_base64, analysis_prompt):
        return content
    if content := _analyze_by_gemini(image_base64, analysis_prompt):
        return content
    # [step3] 所有模型均失败
    log_error("所有视觉模型均无法分析图片")
    return ""
