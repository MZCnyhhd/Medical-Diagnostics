import streamlit as st
import os

def render_sidebar():
    with st.sidebar:
        st.subheader("🤖 选择大模型")
        
        # --- 模型切换功能 ---
        model_options = {
            "Qwen (通义千问)": "qwen",
            "OpenAI (GPT-3.5/4)": "openai",
            "Gemini (Google)": "gemini",
            "Local Model (本地模型)": "local"
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

        # --- 本地模型路径配置 ---
        if selected_key == "local":
            local_path = st.text_input(
                "本地模型路径",
                value=os.getenv("LOCAL_MODEL_PATH", ""),
                placeholder="例如: models/qwen-7b-chat",
                help="请输入本地 HuggingFace 模型目录的绝对路径"
            )
            if local_path:
                os.environ["LOCAL_MODEL_PATH"] = local_path
            else:
                st.warning("请设置本地模型路径")
        
        st.subheader("📚 知识库管理")
        
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
        
        # 清除缓存按钮
        if st.button("🗑️ 清除缓存", use_container_width=True,
                    help="清除诊断结果缓存，释放存储空间"):
            from src.services.cache import get_cache
            cache = get_cache()
            deleted_count = cache.clear_all()
            if deleted_count > 0:
                st.toast(f"已清除 {deleted_count} 条缓存记录", icon="🗑️")
            else:
                st.toast("缓存已清空", icon="✅")
        
