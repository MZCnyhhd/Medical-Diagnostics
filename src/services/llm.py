"""
模块名称: LLM Factory (大模型工厂)
功能描述:

    负责初始化和管理 LangChain 的 ChatModel 实例。
    支持多供应商 (OpenAI, Anthropic, Ollama 等) 和多模型切换。
    提供统一的 `get_llm` 接口，屏蔽底层模型差异。

设计理念:

    1.  **工厂模式**: 根据配置动态创建模型实例，易于扩展新的模型提供商。
    2.  **统一接口**: 返回 LangChain 标准的 `BaseChatModel`，保证上层业务代码的兼容性。
    3.  **配置驱动**: 所有模型参数 (Temperature, API Key) 均从 `settings` 读取。

线程安全性:

    - LangChain 的 LLM 对象通常是线程安全的。

依赖关系:

    - `langchain_openai`, `langchain_community`: 模型实现。
    - `src.core.settings`: 模型配置。
"""

import os
import asyncio
import threading
import base64
import requests
import streamlit as st
from langchain_core.runnables import RunnableLambda
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# [fix] 延迟导入，避免在不支持本地模型的环境（如 Torch 缺失）中启动崩溃
# from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from src.services.logging import log_info, log_warn, log_error
# [定义函数] ############################################################################################################
# [内部-加载本地模型] =====================================================================================================
@st.cache_resource(show_spinner="正在加载本地 AI 模型，请稍候...")
def _load_local_model(model_path: str):
    """
    加载本地 HuggingFace 模型（带 Streamlit 缓存）。
    :param model_path: 本地模型路径
    :return: ChatHuggingFace 实例或 None
    """
    # [step1] 延迟导入本地依赖
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        import torch
    except ImportError as e:
        log_warn(f"本地模型依赖缺失: {e}")
        return None
    # [step2] 加载分词器和模型
    log_info(f"正在加载本地模型: {model_path} ...")
    try:
        # [Fix] HuatuoGPT-7B 等模型包含自定义代码，必须启用 trust_remote_code=True
        # [Fix] 强制 use_fast=False，防止 transformers 尝试将其转换为 FastTokenizer 导致 'vocab_size' 属性冲突
        tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        # [Fix] 某些自定义模型未显式设置 pad_token，会导致生成阶段出现 shape 相关异常
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        
        # [Optimization] 配置 4-bit 量化以适配 RTX 5060 (8GB VRAM)
        # 这将把模型大小从 ~14GB 压缩到 ~4.5GB，无需 offload 到磁盘，大幅提升加载和推理速度
        bnb_config: BitsAndBytesConfig = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # [Fix] 配置模型加载参数
        # 先尝试纯 GPU 4-bit；若显存不足则自动降级为 CPU offload 模式
        try:
            model: AutoModelForCausalLM = AutoModelForCausalLM.from_pretrained(
                model_path,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
            loaded_mode = "4-bit Quantized"
        except Exception as load_err:
            err_text = str(load_err)
            if ("dispatched on the CPU or the disk" in err_text) or ("offload" in err_text.lower()):
                log_warn("检测到显存不足，自动切换为 8-bit + CPU offload 模式加载本地模型...")
                offload_config: BitsAndBytesConfig = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    quantization_config=offload_config,
                    device_map="auto",
                    trust_remote_code=True
                )
                loaded_mode = "8-bit + CPU Offload"
            else:
                raise load_err
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id
        if getattr(model.config, "eos_token_id", None) is None:
            model.config.eos_token_id = tokenizer.eos_token_id
        log_info(f"模型已加载到设备: {model.device} ({loaded_mode})")
    except Exception as e:
        log_warn(f"加载本地模型失败: {e}")
        return None
    # [step3] 直接调用 model.generate（绕过 pipeline，规避部分自定义模型在 pipeline 路径上的 shape 异常）
    max_new_tokens = int(os.getenv("LOCAL_MAX_NEW_TOKENS", "512"))
    temperature = float(os.getenv("LOCAL_TEMPERATURE", "0.1"))
    top_p = float(os.getenv("LOCAL_TOP_P", "0.95"))
    repetition_penalty = float(os.getenv("LOCAL_REPETITION_PENALTY", "1.15"))
    max_input_tokens = int(os.getenv("LOCAL_MAX_INPUT_TOKENS", "2048"))
    generation_lock = threading.Lock()

    def _generate_text(prompt: str) -> str:
        text = str(prompt)
        with generation_lock:
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_input_tokens,
                padding=False
            )
            input_ids = inputs["input_ids"].to(model.device)
            attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(model.device)

            gen_kwargs = {
                "input_ids": input_ids,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
                "do_sample": temperature > 0
            }
            if attention_mask is not None:
                gen_kwargs["attention_mask"] = attention_mask
            if gen_kwargs["do_sample"]:
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = top_p

            with torch.no_grad():
                outputs = model.generate(**gen_kwargs)

            output_ids = outputs[0]
            prompt_len = input_ids.shape[-1]
            generated_ids = output_ids[prompt_len:] if output_ids.shape[-1] > prompt_len else output_ids
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
            if generated_text:
                return generated_text
            return tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    def _invoke(prompt):
        return _generate_text(prompt)

    async def _ainvoke(prompt):
        return await asyncio.to_thread(_generate_text, prompt)

    log_info("本地模型加载成功 (Direct Generate Mode)！")
    return RunnableLambda(_invoke, afunc=_ainvoke)
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

# [内部-初始化 Ollama] =====================================================================================================
def _ollama_generate(prompt: str, model_name: str, base_url: str, temperature: float) -> str:
    """
    直接调用 Ollama /api/generate，绕过 langchain_ollama 在部分环境下的 502 兼容问题。
    """
    api_url = f"{base_url.rstrip('/')}/api/generate"
    timeout = int(os.getenv("OLLAMA_TIMEOUT", "180"))
    payload = {
        "model": model_name,
        "prompt": str(prompt),
        "stream": False,
        "options": {"temperature": temperature}
    }
    resp = requests.post(api_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json() or {}
    return data.get("response", "")


def _init_ollama(temperature, available_models, init_errors):
    """初始化 Ollama 本地模型（HTTP 直连版）"""
    try:
        model_name = os.getenv("OLLAMA_MODEL", "gemma:latest")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        def _invoke(prompt):
            return _ollama_generate(str(prompt), model_name, base_url, temperature)

        async def _ainvoke(prompt):
            return await asyncio.to_thread(_ollama_generate, str(prompt), model_name, base_url, temperature)

        available_models["ollama"] = RunnableLambda(_invoke, afunc=_ainvoke)
        if not hasattr(_init_ollama, "_logged_config") or _init_ollama._logged_config != (model_name, base_url):
            log_info(f"Ollama 模型已配置: {model_name} @ {base_url}")
            _init_ollama._logged_config = (model_name, base_url)
    except Exception as e:
        init_errors["ollama"] = str(e)
        log_warn(f"初始化 Ollama 失败: {e}")

# [内部-初始化可用模型] ===================================================================================================
def _init_available_models(temperature: float) -> tuple[dict, dict]:
    """
    初始化所有配置了 API Key 的模型。
    :param temperature: 模型生成温度
    :return: (可用模型字典, 初始化错误字典)
    """
    available_models: dict = {}
    init_errors: dict = {}
    # [step1] 尝试加载本地模型（仅在用户选择本地模式时加载，避免云端模式下浪费时间）
    current_provider = os.getenv("LLM_PROVIDER", "qwen").lower()
    if current_provider == "local":
        local_path = os.getenv("LOCAL_MODEL_PATH")
        if local_path:
            if not os.path.isabs(local_path):
                local_path = os.path.abspath(local_path)
            if os.path.exists(local_path):
                local_model = _load_local_model(local_path)
                if local_model:
                    available_models["local"] = local_model
            else:
                log_warn(f"配置的本地模型路径不存在: {local_path}")
                init_errors["local"] = f"路径不存在: {local_path}"
    # [step2] 按需初始化模型：云端模式只初始化云端 API，本地模式只初始化本地模型
    # 选 qwen/baichuan → 仅初始化对应云端 API
    # 选 ollama → 仅初始化 Ollama 本地服务
    # 选 local → 仅加载 HuggingFace 本地模型（已在 step1 完成）
    if current_provider in ("qwen", "baichuan"):
        _init_qwen(temperature, available_models, init_errors)
        _init_baichuan(temperature, available_models, init_errors)
    elif current_provider == "ollama":
        _init_ollama(temperature, available_models, init_errors)
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
    priority_order = ["qwen", "baichuan", "openai", "gemini", "ollama", "local"]
    # [step3] 选择主模型提供商
    provider = _select_provider(override_provider, available_models, init_errors, priority_order)
    # [step4] 卫语句：处理没有任何模型可用的极端情况
    if provider not in available_models:
        log_warn("未检测到任何有效的 API Key 配置！将尝试返回默认 Qwen 配置（可能会报错）。")
        return ChatTongyi(model="qwen-max", temperature=temperature)
    # [step5] 配置备用模型链 (Fallback)
    primary_model = available_models[provider]
    fallback_names = [p for p in priority_order if p != provider and p in available_models]
    fallback_models = [available_models[p] for p in fallback_names]
    # [step6] 打印配置日志（仅限首次）
    # 使用 frozenset 确保 hashable，用于缓存键
    current_config_key = (provider, tuple(fallback_names))
    
    if not hasattr(get_chat_model, "_logged_configs"):
        get_chat_model._logged_configs = set()
    
    if current_config_key not in get_chat_model._logged_configs:
        log_msg = f"LLM 配置: 主模型={provider}, 备用模型链={fallback_names}"
        log_info(log_msg)
        get_chat_model._logged_configs.add(current_config_key)
    
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
