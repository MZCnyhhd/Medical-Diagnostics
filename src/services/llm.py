"""
LLM 工厂模块：统一管理多个大语言模型，支持自动容灾切换
======================================================

本模块是整个系统的 AI 引擎层，负责管理和调用各种大语言模型。

支持的模型后端：
1. Qwen（通义千问）：阿里云的大语言模型，默认首选
2. OpenAI（GPT）：OpenAI 的 GPT 系列模型
3. Gemini：Google 的 Gemini 模型
4. 本地模型：HuggingFace 上的开源模型（支持离线部署）

核心功能：
- get_chat_model()：获取配置好的 Chat 模型实例
  - 自动检测可用的模型（基于 API Key 配置）
  - 配置自动容灾：主模型失败时自动切换到备用模型
  - 支持强制指定模型提供商

- _load_local_model()：加载本地 HuggingFace 模型
  - 用于离线部署场景
  - 支持 GPU 加速（自动检测 CUDA）

- analyze_medical_image()：使用视觉模型分析医疗图片
  - 支持 X 光片、CT、MRI、检验报告单等
  - 自动选择可用的视觉模型

容灾策略：
主模型（Qwen）→ 备用1（OpenAI）→ 备用2（Gemini）→ 备用3（本地模型）

如果主模型调用失败（网络错误、API 限流等），会自动切换到下一个可用的备用模型，
确保诊断服务的高可用性。

环境变量配置：
- LLM_PROVIDER：指定默认的模型提供商（qwen/openai/gemini/local）
- LLM_TEMPERATURE：生成温度（0-1，默认 0，越高越有创意）
- DASHSCOPE_API_KEY：通义千问 API Key
- OPENAI_API_KEY：OpenAI API Key
- GOOGLE_API_KEY：Google Gemini API Key
- LOCAL_MODEL_PATH：本地模型路径
- QWEN_MODEL：Qwen 模型名称（默认 qwen-max）
- OPENAI_MODEL：OpenAI 模型名称（默认 gpt-3.5-turbo）
- GEMINI_MODEL：Gemini 模型名称（默认 gemini-pro）
"""

# ==================== 标准库导入 ====================
# os：用于读取环境变量
import os
# base64：用于图片的 Base64 编码（视觉模型需要）
import base64
# requests：用于调用视觉模型的 REST API
import requests
# typing：类型注解
from typing import Optional, List, Any

# ==================== LangChain 模型导入 ====================
# ChatTongyi：LangChain 对通义千问的封装
from langchain_community.chat_models import ChatTongyi
# ChatOpenAI：LangChain 对 OpenAI 的封装
from langchain_openai import ChatOpenAI
# ChatGoogleGenerativeAI：LangChain 对 Google Gemini 的封装
from langchain_google_genai import ChatGoogleGenerativeAI

# ==================== 项目内部模块导入 ====================
# 日志工具函数
from src.services.logging import log_info, log_warn, log_error
# 配置管理
from src.core.settings import get_settings

# ==================== Streamlit 和 HuggingFace 导入 ====================
# streamlit：用于 @st.cache_resource 缓存装饰器
import streamlit as st
# HuggingFacePipeline：LangChain 对 HuggingFace 模型的封装
# ChatHuggingFace：将 HuggingFace 模型包装为 Chat 接口
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace


@st.cache_resource
def _load_local_model(model_path):
    """
    加载本地 HuggingFace 模型（带 Streamlit 缓存）
    
    用于离线部署场景，支持加载本地保存的 HuggingFace 模型。
    常用于：
    - 无网络环境的部署
    - 需要数据隐私保护的场景（数据不出本地）
    - 使用自定义微调的模型
    
    支持的模型类型：
    - DeepSeek 系列
    - ChatGLM 系列
    - Qwen 本地版
    - 其他 HuggingFace 兼容的因果语言模型
    
    缓存机制：
    使用 @st.cache_resource 装饰器，模型只会在首次调用时加载，
    后续调用直接返回缓存的模型实例，避免重复加载。
    
    Args:
        model_path (str): 本地模型路径
            - 可以是 HuggingFace Hub 路径，如 "THUDM/chatglm3-6b"
            - 也可以是本地目录，如 "/home/models/chatglm3-6b"
    
    Returns:
        ChatHuggingFace | None: 
            - 成功：返回包装好的 Chat 模型实例
            - 失败：返回 None（会记录警告日志）
    
    技术细节：
    - 使用 torch.float16 半精度以节省显存
    - device_map="auto" 自动选择设备（GPU 优先）
    - trust_remote_code=True 允许执行模型的自定义代码
    """
    try:
        # 导入 HuggingFace Transformers 库的组件
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        # 导入 PyTorch
        import torch

        # 记录加载开始
        log_info(f"正在加载本地模型: {model_path} ...")
        
        # ========== 第一步：加载分词器 ==========
        # 分词器负责将文本转换为模型可以理解的 token ID
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # ========== 第二步：加载模型 ==========
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            # 使用 float16 半精度，显存占用减半
            torch_dtype=torch.float16,
            # 自动选择设备（如果有 GPU 就用 GPU）
            device_map="auto",
            # 允许执行模型仓库中的自定义代码
            # 某些模型（如 ChatGLM）需要这个选项
            trust_remote_code=True
        )
        # 记录模型加载到的设备
        log_info(f"模型已加载到设备: {model.device} (如果是 cuda:0 则代表使用的是 NVIDIA 显卡)")

        # ========== 第三步：创建文本生成 Pipeline ==========
        # Pipeline 是 HuggingFace 的高级 API，封装了推理的完整流程
        pipe = pipeline(
            # 任务类型：文本生成
            "text-generation",
            # 使用前面加载的模型
            model=model,
            # 使用前面加载的分词器
            tokenizer=tokenizer,
            # 最大生成 token 数
            max_new_tokens=2048,
            # 生成温度（0.1 接近确定性输出）
            temperature=0.1,
            # Top-p 采样参数
            top_p=0.95,
            # 重复惩罚（防止生成重复内容）
            repetition_penalty=1.15,
            # 不返回输入的 prompt（只返回生成的部分）
            return_full_text=False
        )

        # ========== 第四步：包装为 LangChain 接口 ==========
        # 首先包装为 HuggingFacePipeline
        local_llm = HuggingFacePipeline(pipeline=pipe)
        # 然后包装为 ChatHuggingFace
        # 这一步会自动应用 Chat Template（如 <|im_start|>user...）
        # 解决模型复读 Prompt 或无法理解指令的问题
        local_chat_model = ChatHuggingFace(llm=local_llm)
        
        # 记录加载成功
        log_info("本地模型加载成功！")
        return local_chat_model
        
    except Exception as e:
        # 加载失败，记录警告并返回 None
        log_warn(f"加载本地模型失败: {e}")
        return None


def get_chat_model(override_provider=None):
    """
    根据配置返回 Chat 模型实例，支持自动容灾切换
    
    这是获取 LLM 的统一入口函数。会根据环境变量配置：
    1. 初始化所有可用的模型（检测到 API Key 的）
    2. 选择主模型（根据配置或参数）
    3. 配置备用模型链（用于自动容灾）
    
    容灾机制：
    当主模型调用失败时（网络错误、API 限流、服务不可用等），
    LangChain 会自动切换到备用模型，按优先级依次尝试：
    Qwen → OpenAI → Gemini → 本地模型
    
    Args:
        override_provider (str, optional): 强制指定的主模型提供商
            - "qwen"：使用通义千问
            - "openai"：使用 OpenAI GPT
            - "gemini"：使用 Google Gemini
            - "local"：使用本地模型
            - None：使用环境变量 LLM_PROVIDER 的配置
            
            注意：如果指定的模型不可用，会抛出 ValueError
            （与不指定时的自动切换行为不同）
    
    Returns:
        BaseChatModel: LangChain 的 Chat 模型实例
            - 如果有多个模型可用，返回的是带 Fallback 的模型
            - 调用时会自动进行容灾切换
    
    Raises:
        ValueError: 当 override_provider 指定的模型不可用时
    
    使用示例：
    ```python
    # 使用默认配置（自动选择和容灾）
    model = get_chat_model()
    response = model.invoke("你好")
    
    # 强制使用 OpenAI
    model = get_chat_model(override_provider="openai")
    
    # 异步调用
    response = await model.ainvoke("分析这份报告...")
    ```
    """
    
    # ========== 读取配置 ==========
    # 从环境变量读取生成温度
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    # 某些模型不接受 temperature=0，需要设置一个接近 0 的值
    if temperature == 0:
        temperature = 0.01
    
    # 用于存储成功初始化的模型实例
    available_models = {}
    # 用于存储初始化失败的错误信息（用于错误提示）
    init_errors = {}

    # ==================== 第一步：初始化所有可用的模型 ====================
    # 尝试初始化所有配置了 API Key 的模型
    # 失败不影响其他模型的初始化
    
    # ---------- 0. 初始化本地模型 ----------
    # 如果配置了 LOCAL_MODEL_PATH 且路径存在
    local_model_path = os.getenv("LOCAL_MODEL_PATH")
    if local_model_path and os.path.exists(local_model_path):
        # 调用本地模型加载函数
        local_model = _load_local_model(local_model_path)
        if local_model:
            # 加载成功，添加到可用模型字典
            available_models["local"] = local_model

    # ---------- 1. 初始化 Qwen（通义千问）----------
    # Qwen 是默认的主模型
    if os.getenv("DASHSCOPE_API_KEY"):
        try:
            # 从环境变量读取模型名称，默认使用 qwen-max
            model_name = os.getenv("QWEN_MODEL", "qwen-max")
            # 创建 ChatTongyi 实例
            # ChatTongyi 是 LangChain 对通义千问 API 的封装
            available_models["qwen"] = ChatTongyi(model=model_name, temperature=temperature)
        except Exception as e:
            # 初始化失败，记录错误
            init_errors["qwen"] = str(e)
            log_warn(f"初始化 Qwen 失败: {e}")

    # ---------- 2. 初始化 OpenAI（GPT 系列）----------
    if os.getenv("OPENAI_API_KEY"):
        try:
            # 从环境变量读取模型名称，默认使用 gpt-3.5-turbo
            model_name = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
            # 创建 ChatOpenAI 实例
            available_models["openai"] = ChatOpenAI(
                model=model_name, 
                temperature=temperature, 
                # 显式传入 API Key（避免依赖环境变量名）
                api_key=os.getenv("OPENAI_API_KEY")
            )
        except Exception as e:
            # 初始化失败，记录错误
            init_errors["openai"] = str(e)
            log_warn(f"初始化 OpenAI 失败: {e}")

    # ---------- 3. 初始化 Gemini（Google）----------
    if os.getenv("GOOGLE_API_KEY"):
        try:
            # 从环境变量读取模型名称，默认使用 gemini-pro
            model_name = os.getenv("GEMINI_MODEL", "gemini-pro")
            # 创建 ChatGoogleGenerativeAI 实例
            available_models["gemini"] = ChatGoogleGenerativeAI(
                model=model_name, 
                temperature=temperature, 
                # 显式传入 API Key
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        except Exception as e:
            # 初始化失败，记录错误
            init_errors["gemini"] = str(e)
            log_warn(f"初始化 Gemini 失败: {e}")

    # ==================== 第二步：确定主模型 ====================
    # 优先使用用户指定的模型，否则使用环境变量配置
    if override_provider:
        # 用户强制指定了模型
        provider = override_provider.lower()
    else:
        # 从环境变量读取，默认使用 qwen
        provider = os.getenv("LLM_PROVIDER", "qwen").lower()
    
    # 模型优先级顺序（用于自动切换）
    # 当指定的模型不可用时，按此顺序尝试备用模型
    priority_order = ["qwen", "openai", "gemini", "local"]
    
    # 检查指定的主模型是否可用
    if provider not in available_models:
        # 如果是用户强制指定的，直接报错，不要回退
        if override_provider:
            # 构建详细的错误信息
            error_detail = init_errors.get(provider, "未知的初始化错误 (可能是 Key 未配置)")
            raise ValueError(f"无法加载指定的模型 '{provider}'。原因: {error_detail}")
        
        # 不是强制指定的，尝试自动切换到可用的模型
        log_warn(f"指定的 LLM_PROVIDER='{provider}' 不可用。")
        
        # 按优先级顺序查找可用的模型
        for p in priority_order:
            if p in available_models:
                provider = p
                log_warn(f"自动切换到优先级最高的可用模型: '{provider}'。")
                break
        else:
            # 如果优先级列表中的模型都不可用
            if available_models:
                # 但还有其他可用模型，随机选一个
                provider = list(available_models.keys())[0]
                log_warn(f"优先级列表中的模型均不可用，随机选择可用模型: '{provider}'。")
            else:
                # 没有任何可用的模型，返回默认 Qwen 配置
                # 这种情况调用时会报错，但至少不会在初始化阶段崩溃
                log_warn("未检测到任何有效的 API Key 配置！将尝试返回默认 Qwen 配置（可能会报错）。")
                return ChatTongyi(model="qwen-max", temperature=temperature)

    # 获取主模型实例
    primary_model = available_models[provider]

    # ==================== 第三步：配置备用模型链（Fallback）====================
    # 按照优先级顺序生成 fallback 列表，排除当前主模型
    # 这样当主模型失败时，LangChain 会自动切换到备用模型
    fallback_models = []
    for p in priority_order:
        if p != provider and p in available_models:
            fallback_models.append(available_models[p])

    # ---------- 防止重复日志 ----------
    # 使用函数属性缓存已记录的配置
    # 避免每次调用都打印相同的日志
    if not hasattr(get_chat_model, "_logged_configs"):
        get_chat_model._logged_configs = set()

    # ---------- 配置 Fallback ----------
    if fallback_models:
        # 有备用模型，配置自动切换
        # 构建备用模型名称列表（用于日志）
        fallback_names = [p for p in priority_order if p != provider and p in available_models]
        log_msg = f"LLM 配置: 主模型={provider}, 备用模型链={fallback_names}"
        
        # 只在首次配置时打印日志
        if log_msg not in get_chat_model._logged_configs:
            log_info(log_msg)
            get_chat_model._logged_configs.add(log_msg)

        # 使用 LangChain 的 with_fallbacks() 实现自动容灾切换
        # 当主模型调用失败时，自动尝试备用模型
        # 直到成功或所有模型都失败
        return primary_model.with_fallbacks(fallback_models)
    else:
        # 无备用模型，直接返回主模型
        log_msg = f"LLM 配置: 主模型={provider}, 无可用备用模型。"

        # 只在首次配置时打印日志
        if log_msg not in get_chat_model._logged_configs:
            log_info(log_msg)
            get_chat_model._logged_configs.add(log_msg)

        return primary_model


def analyze_medical_image(image_bytes: bytes) -> str:
    """
    使用视觉模型分析医疗图片，提取医学文字描述
    
    支持分析各类医学图像：
    - 医学影像：X 光片、CT 扫描、MRI、超声图像
    - 病理图片：病理切片、显微镜图像
    - 检验报告：化验单、检查报告单的图片
    - 其他医学相关图片
    
    视觉模型优先级：
    1. Qwen-VL（通义千问视觉模型）：阿里云的多模态模型
    2. OpenAI GPT-4 Vision：OpenAI 的多模态模型
    3. Google Gemini Vision：Google 的多模态模型
    
    按优先级依次尝试，直到成功或所有模型都失败。
    
    Args:
        image_bytes (bytes): 图片的字节数据
            - 支持 JPEG、PNG 等常见格式
            - 会自动转换为 Base64 编码
    
    Returns:
        str: 图片的医学文字描述
            - 成功时返回详细的分析报告
            - 失败时返回空字符串（不会抛出异常）
    
    返回的报告格式：
    ```markdown
    ## 图像类型
    X光片 / CT / MRI / 超声 / 病理切片 / 检验报告单等
    
    ## 图像描述
    详细描述图像中可见的医学信息...
    
    ## 关键发现
    - 发现1：...
    - 发现2：...
    
    ## 初步印象
    基于图像分析的医学印象...
    ```
    
    使用示例：
    ```python
    # 读取图片文件
    with open("xray.jpg", "rb") as f:
        image_bytes = f.read()
    
    # 分析图片
    description = analyze_medical_image(image_bytes)
    print(description)
    ```
    """
    # ========== 图片预处理 ==========
    # 将图片字节数据转换为 Base64 编码
    # 视觉模型的 API 通常要求图片以 Base64 格式传输
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # ========== 构建分析 Prompt ==========
    # 详细的系统提示，指导视觉模型如何分析医学图片
    analysis_prompt = """你是一位专业的医学影像分析专家。请仔细分析这张医疗图片，并提供详细的文字描述报告。

请按以下格式输出：

## 图像类型
（说明这是什么类型的医学图像，如：X光片、CT扫描、MRI、超声、病理切片、检验报告单等）

## 图像描述
（详细描述图像中可见的所有医学相关信息，包括解剖结构、异常发现等）

## 关键发现
（列出图像中的关键医学发现，如有异常请详细说明）

## 初步印象
（基于图像分析给出初步的医学印象，供后续诊断参考）

请确保描述准确、专业，使用标准的医学术语。"""

    # ==================== 尝试使用 Qwen-VL ====================
    # Qwen-VL 是阿里云的视觉语言模型，支持图片理解
    if os.getenv("DASHSCOPE_API_KEY"):
        try:
            log_info("使用 Qwen-VL 分析医疗图片...")
            # 获取 API Key
            api_key = os.getenv("DASHSCOPE_API_KEY")
            
            # 调用 Qwen-VL API（使用 OpenAI 兼容接口）
            response = requests.post(
                # DashScope 的 OpenAI 兼容端点
                "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                headers={
                    # Bearer Token 认证
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    # 使用 qwen-vl-max 模型（最强的视觉模型）
                    "model": "qwen-vl-max",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                # 文本部分：分析指令
                                {"type": "text", "text": analysis_prompt},
                                # 图片部分：Base64 编码的图片
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                            ]
                        }
                    ],
                    # 最大输出 token 数
                    "max_tokens": 2000
                },
                # 超时时间 60 秒（视觉模型处理较慢）
                timeout=60
            )
            
            # 检查响应状态
            if response.status_code == 200:
                # 解析响应 JSON
                result = response.json()
                # 提取生成的文本内容
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    log_info("Qwen-VL 图片分析成功")
                    return content
            else:
                # API 调用失败
                log_warn(f"Qwen-VL 请求失败: {response.status_code} - {response.text}")
        except Exception as e:
            # 异常处理
            log_warn(f"Qwen-VL 分析失败: {e}")
    
    # ==================== 尝试使用 OpenAI GPT-4 Vision ====================
    # 如果 Qwen-VL 失败或不可用，尝试 OpenAI
    if os.getenv("OPENAI_API_KEY"):
        try:
            log_info("使用 OpenAI GPT-4 Vision 分析医疗图片...")
            api_key = os.getenv("OPENAI_API_KEY")
            
            # 调用 OpenAI API
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    # 使用 gpt-4o 模型（支持视觉）
                    "model": "gpt-4o",
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": analysis_prompt},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                            ]
                        }
                    ],
                    "max_tokens": 2000
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                if content:
                    log_info("OpenAI GPT-4 Vision 图片分析成功")
                    return content
            else:
                log_warn(f"OpenAI 请求失败: {response.status_code} - {response.text}")
        except Exception as e:
            log_warn(f"OpenAI Vision 分析失败: {e}")
    
    # ==================== 尝试使用 Google Gemini Vision ====================
    # 如果前面的模型都失败，尝试 Gemini
    if os.getenv("GOOGLE_API_KEY"):
        try:
            log_info("使用 Google Gemini Vision 分析医疗图片...")
            api_key = os.getenv("GOOGLE_API_KEY")
            
            # 调用 Gemini API（Google AI Studio 接口）
            response = requests.post(
                # Gemini API 端点（API Key 作为查询参数）
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
                headers={"Content-Type": "application/json"},
                json={
                    "contents": [
                        {
                            "parts": [
                                # 文本部分
                                {"text": analysis_prompt},
                                # 图片部分（Gemini 使用 inline_data 格式）
                                {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
                            ]
                        }
                    ]
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                # Gemini 的响应结构与 OpenAI 不同
                content = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                if content:
                    log_info("Google Gemini Vision 图片分析成功")
                    return content
            else:
                log_warn(f"Gemini 请求失败: {response.status_code} - {response.text}")
        except Exception as e:
            log_warn(f"Gemini Vision 分析失败: {e}")
    
    # ==================== 所有模型都失败 ====================
    # 记录错误日志并返回空字符串
    log_error("所有视觉模型均无法分析图片")
    return ""
