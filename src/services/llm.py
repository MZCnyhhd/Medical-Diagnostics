"""
LLM 工厂模块：统一管理多个大语言模型，支持自动容灾切换

本模块实现了 LLM 的统一接口，支持：
1. 多模型后端：Qwen（通义千问）、OpenAI（GPT）、Gemini、本地 HuggingFace 模型
2. 自动容灾：主模型失败时自动切换到备用模型
3. 配置灵活：通过环境变量控制模型选择和行为

核心功能：
- get_chat_model(): 获取配置好的 Chat 模型实例（带自动回退）
- _load_local_model(): 加载本地 HuggingFace 模型（用于离线部署）

容灾策略：
主模型（Qwen）→ 备用1（OpenAI）→ 备用2（Gemini）→ 备用3（本地模型）
"""

import os
from langchain_community.chat_models import ChatTongyi
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from src.services.logging import log_info, log_warn

import streamlit as st
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace


@st.cache_resource
def _load_local_model(model_path):
    """
    加载本地 HuggingFace 模型（带 Streamlit 缓存）
    
    用于离线部署场景，支持加载本地 HuggingFace 模型（如 DeepSeek、ChatGLM 等）。
    使用 Streamlit 缓存避免重复加载，提升性能。
    
    Args:
        model_path (str): 本地模型路径（可以是 HuggingFace Hub 路径或本地目录）
    
    Returns:
        ChatHuggingFace | None: 加载成功的模型实例，失败返回 None
    """
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch

        log_info(f"正在加载本地模型: {model_path} ...")
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        log_info(f"模型已加载到设备: {model.device} (如果是 cuda:0 则代表使用的是 NVIDIA 显卡)")

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=2048,
            temperature=0.1,
            top_p=0.95,
            repetition_penalty=1.15,
            return_full_text=False
        )

        local_llm = HuggingFacePipeline(pipeline=pipe)
        # 使用 ChatHuggingFace 包装，以便自动应用 Chat Template (如 <|im_start|>user...)
        # 这能解决模型复读 Prompt 或无法理解指令的问题
        local_chat_model = ChatHuggingFace(llm=local_llm)
        
        log_info("本地模型加载成功！")
        return local_chat_model
    except Exception as e:
        log_warn(f"加载本地模型失败: {e}")
        return None

def get_chat_model(override_provider=None):
    """根据环境变量返回配置了自动重试与回退机制的 Chat 模型实例。
    
    Args:
        override_provider (str, optional): 强制指定的主模型提供商 (e.g., "qwen", "openai", "local").
                                         如果未指定，则使用环境变量 LLM_PROVIDER。

    逻辑：
    1. 尝试初始化所有配置了 API Key 的模型（Qwen, OpenAI, Gemini）以及本地模型。
    2. 根据 override_provider 或 LLM_PROVIDER 确定主模型。
    3. 按照优先级配置回退（Fallback）模型：Qwen -> OpenAI -> Gemini -> Local。
    """
    
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    if temperature == 0:
        temperature = 0.01 # Prevent validation errors for some providers that require > 0
    available_models = {}
    init_errors = {}

    # ========== 第一步：初始化所有可用的模型 ==========
    # 尝试初始化所有配置了 API Key 的模型，失败不影响其他模型
    
    # 0. 初始化本地模型（如果配置了 LOCAL_MODEL_PATH）
    local_model_path = os.getenv("LOCAL_MODEL_PATH")
    if local_model_path and os.path.exists(local_model_path):
        local_model = _load_local_model(local_model_path)
        if local_model:
            available_models["local"] = local_model

    # 1. 初始化 Qwen（通义千问）- 默认主模型
    if os.getenv("DASHSCOPE_API_KEY"):
        try:
            model_name = os.getenv("QWEN_MODEL", "qwen-max")
            available_models["qwen"] = ChatTongyi(model=model_name, temperature=temperature)
        except Exception as e:
            init_errors["qwen"] = str(e)
            log_warn(f"初始化 Qwen 失败: {e}")

    # 2. 初始化 OpenAI（GPT 系列）
    if os.getenv("OPENAI_API_KEY"):
        try:
            model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            available_models["openai"] = ChatOpenAI(
                model=model_name, 
                temperature=temperature, 
                api_key=os.getenv("OPENAI_API_KEY")
            )
        except Exception as e:
            init_errors["openai"] = str(e)
            log_warn(f"初始化 OpenAI 失败: {e}")

    # 3. 初始化 Gemini（Google）
    if os.getenv("GOOGLE_API_KEY"):
        try:
            model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
            available_models["gemini"] = ChatGoogleGenerativeAI(
                model=model_name, 
                temperature=temperature, 
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        except Exception as e:
            init_errors["gemini"] = str(e)
            log_warn(f"初始化 Gemini 失败: {e}")

    # ========== 第二步：确定主模型 ==========
    # 优先使用用户指定的模型，否则使用环境变量配置
    if override_provider:
        provider = override_provider.lower()
    else:
        provider = os.getenv("LLM_PROVIDER", "qwen").lower()
    
    # 模型优先级顺序（用于自动切换）
    # 如果主模型不可用，按此顺序尝试备用模型
    priority_order = ["qwen", "openai", "gemini", "local"]
    
    # 如果指定的主模型不可用
    if provider not in available_models:
        # 如果是用户强制指定的，直接报错，不要回退
        if override_provider:
            error_detail = init_errors.get(provider, "未知的初始化错误 (可能是 Key 未配置)")
            raise ValueError(f"无法加载指定的模型 '{provider}'。原因: {error_detail}")
            
        log_warn(f"指定的 LLM_PROVIDER='{provider}' 不可用。")
        for p in priority_order:
            if p in available_models:
                provider = p
                log_warn(f"自动切换到优先级最高的可用模型: '{provider}'。")
                break
        else:
            # 如果所有都在列表中找不到（极少见），尝试取 available_models 的第一个
            if available_models:
                provider = list(available_models.keys())[0]
                log_warn(f"优先级列表中的模型均不可用，随机选择可用模型: '{provider}'。")
            else:
                log_warn("未检测到任何有效的 API Key 配置！将尝试返回默认 Qwen 配置（可能会报错）。")
                return ChatTongyi(model="qwen-max", temperature=temperature)

    primary_model = available_models[provider]

    # ========== 第三步：配置备用模型链（Fallback） ==========
    # 按照优先级顺序生成 fallback 列表，排除当前主模型
    # 这样当主模型失败时，LangChain 会自动切换到备用模型
    fallback_models = []
    for p in priority_order:
        if p != provider and p in available_models:
            fallback_models.append(available_models[p])

    if fallback_models:
        # 有备用模型，配置自动切换
        fallback_names = [p for p in priority_order if p != provider and p in available_models]
        
        # 防止重复日志（使用函数属性缓存已记录的配置）
        log_msg = f"LLM 配置: 主模型={provider}, 备用模型链={fallback_names}"
        if not hasattr(get_chat_model, "_logged_configs"):
             get_chat_model._logged_configs = set()
        
        if log_msg not in get_chat_model._logged_configs:
            log_info(log_msg)
            get_chat_model._logged_configs.add(log_msg)

        # 使用 LangChain 的 with_fallbacks() 实现自动容灾切换
        # 当主模型调用失败时，自动尝试备用模型，直到成功或所有模型都失败
        return primary_model.with_fallbacks(fallback_models)
    else:
        # 无备用模型，直接返回主模型
        log_msg = f"LLM 配置: 主模型={provider}, 无可用备用模型。"
        if not hasattr(get_chat_model, "_logged_configs"):
             get_chat_model._logged_configs = set()

        if log_msg not in get_chat_model._logged_configs:
            log_info(log_msg)
            get_chat_model._logged_configs.add(log_msg)

        return primary_model
