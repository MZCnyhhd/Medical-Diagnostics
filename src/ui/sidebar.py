"""
模块名称: Sidebar Component (侧边栏组件)
功能描述:

    渲染应用的左侧控制面板。
    包含模型选择、知识库管理 (上传/重建)、缓存清理等系统级操作入口。

设计理念:

    1.  **功能聚合**: 将配置和管理类功能集中在侧边栏，保持主界面 (Main Content) 专注于诊断业务。
    2.  **即时反馈**: 操作 (如切换模型) 立即生效，通常通过修改环境变量或 Session State 实现。
    3.  **状态可视**: 显示当前连接的模型、数据库状态等信息。

线程安全性:

    - 依赖 Streamlit 的渲染线程，操作 Session State 需注意并发 (但在 Streamlit 中通常是单线程模型)。

依赖关系:

    - `streamlit`: UI 框架。
    - `src.core.settings`: 读取和修改配置。
"""

import os
import streamlit as st

# [定义函数] ############################################################################################################
# [UI-渲染侧边栏] =========================================================================================================
def render_sidebar():
    """渲染侧边栏组件"""
    with st.sidebar:
        st.subheader("🤖 选择大模型")
        
        # [step1] 模型切换功能
        model_options = {
            "Qwen-Turbo (通义千问)": "qwen",
            "Baichuan M2 (百川)": "baichuan",
            "Ollama Service (本地服务)": "ollama",
            "HuggingFace Native (原生加载)": "local"
        }
        
        # 获取当前环境变量中的默认值
        current_provider = os.getenv("LLM_PROVIDER", "qwen")
        # 反向查找对应的 index
        default_index = 0
        for idx, (name, key) in enumerate(model_options.items()):
            if key == current_provider:
                default_index = idx
                break
        
        selected_model_name = st.selectbox(
            "AI后端",
            options=list(model_options.keys()),
            index=default_index,
            key="model_selector",
            help="选择用于诊断的底层大语言模型",
            label_visibility="collapsed"
        )
        
        # 更新环境变量
        selected_key = model_options[selected_model_name]
        os.environ["LLM_PROVIDER"] = selected_key

        # [step2-1] Ollama 模型配置
        if selected_key == "ollama":
            ollama_base = st.text_input(
                "Ollama 地址",
                value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
                help="Ollama 服务的 API 地址"
            )
            os.environ["OLLAMA_BASE_URL"] = ollama_base
            
            ollama_model = st.text_input(
                "Ollama 模型名称",
                value=os.getenv("OLLAMA_MODEL", "FreedomIntelligence/HuatuGPT-7B"),
                placeholder="例如: llama3, gemma:latest",
                help="请输入已在 Ollama 中下载的模型名称"
            )
            os.environ["OLLAMA_MODEL"] = ollama_model
            
            # 显示状态检查
            if st.button("测试 Ollama 连接", use_container_width=True):
                try:
                    import requests
                    # 临时清除代理环境变量以避免 localhost 连接问题
                    proxies = {"http": None, "https": None}
                    resp = requests.get(ollama_base, timeout=2, proxies=proxies)
                    if resp.status_code == 200:
                        st.success("✅ 服务连接成功")
                        # 检查模型
                        try:
                            tags = requests.get(f"{ollama_base}/api/tags", timeout=2, proxies=proxies).json()
                            models = [m['name'] for m in tags.get('models', [])]
                            # 不区分大小写匹配
                            target = ollama_model.lower()
                            # 处理 :latest 后缀
                            if ":" not in target:
                                target += ":latest"
                            
                            found = False
                            for m in models:
                                m_lower = m.lower()
                                if target == m_lower:
                                    found = True
                                    break
                                # 尝试如果不带 latest
                                if target.replace(":latest", "") == m_lower:
                                    found = True
                                    break
                                    
                            if found:
                                st.success(f"✅ 模型 {ollama_model} 已就绪")
                            else:
                                st.warning(f"⚠️ 未找到模型 {ollama_model}，请先执行 pull")
                                st.info(f"可用模型: {', '.join(models)}")
                        except:
                            pass
                    else:
                        st.error(f"❌ 服务异常: {resp.status_code}")
                except Exception as e:
                    st.error(f"❌ 无法连接到 Ollama: {str(e)}")

        # [step2-2] HuggingFace 本地模型路径配置
        if selected_key == "local":
            local_path = st.text_input(
                "大语言模型路径 (LLM)",
                value=os.getenv("LOCAL_MODEL_PATH", ""),
                placeholder="例如: models/qwen-7b-chat",
                help="请输入本地 HuggingFace 模型目录的绝对路径"
            )
            if local_path:
                os.environ["LOCAL_MODEL_PATH"] = local_path
            else:
                st.warning("请设置本地模型路径")

            local_embedding = st.text_input(
                "Embedding 模型路径",
                value=os.getenv("LOCAL_EMBEDDING_MODEL", ""),
                placeholder="例如: models/bge-small-zh",
                help="请输入本地 Embedding 模型目录的绝对路径"
            )
            if local_embedding:
                os.environ["LOCAL_EMBEDDING_MODEL"] = local_embedding
        
        st.subheader("📚 知识库管理")
        
        # [step3] 知识库管理按钮
        # 分两个按钮，明确功能区分
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 更新向量库", use_container_width=True, 
                        help="更新 RAG 检索用的向量索引 (Pinecone/FAISS)"):
                with st.spinner("正在处理文档..."):
                    from src.scripts.ingest_knowledge import ingest_docs
                    status = ingest_docs()
                    if "成功" in status:
                        st.toast(status, icon="✅")
                    else:
                        st.error(status)
        
        with col2:
            # Neo4j 按钮（如果启用）
            from src.core.settings import get_settings
            settings = get_settings()
            
            if settings.enable_neo4j:
                if st.button("🕸️ 更新图谱", use_container_width=True, 
                            help="更新 Neo4j 知识图谱（需要较长时间）"):
                    with st.spinner("正在构建知识图谱..."):
                        try:
                            from src.scripts.build_kg import build_knowledge_graph
                            result = build_knowledge_graph()
                            if result and "成功" in result:
                                st.toast(result, icon="✅")
                            else:
                                st.toast("知识图谱构建完成", icon="✅")
                        except Exception as e:
                            st.error(f"构建失败: {str(e)}")
            else:
                st.button("🕸️ 图谱未启用", use_container_width=True, disabled=True,
                         help="在配置中设置 ENABLE_NEO4J=true 以启用")
        
        # [step4] 缓存清理
        if st.button("🗑️ 清除缓存", use_container_width=True,
                    help="清除诊断结果缓存，释放存储空间"):
            from src.services.cache import get_cache
            cache = get_cache()
            deleted_count = cache.clear_all()
            if deleted_count > 0:
                st.toast(f"已清除 {deleted_count} 条缓存记录", icon="🗑️")
            else:
                st.toast("缓存已清空", icon="✅")
        
