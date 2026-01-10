import streamlit as st
import os

def render_sidebar():
    with st.sidebar:
        # Logo Area
        st.markdown(
            """
            <div style="text-align: center; padding: 1rem 0;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">🏥</div>
                <div style="font-weight: 800; font-size: 1.2rem; color: #2c3e50;">智能医疗诊断系统</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.divider()
        
        # 系统介绍
        st.markdown(
            """
            <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; border: 1px solid #bce3eb; color: #315e6b; margin-bottom: 1rem;">
                <div style="text-align: center; font-weight: bold; font-size: 16px; margin-bottom: 8px;">
                    智能多学科会诊系统 (MDT) v1.0.0
                </div>
                <div style="font-size: 14px; line-height: 1.5;">
                    模拟真实医院的 MDT 流程，由多个 AI 专科医生协同工作，提供全面的诊断建议。
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        
        st.subheader("🤖 选择大模型")
        
        # --- 模型切换功能 ---
        model_options = {
            "Qwen (通义千问)": "qwen",
            "OpenAI (GPT-3.5/4)": "openai",
            "Gemini (Google)": "gemini"
        }
        
        # 获取当前环境变量中的默认值
        current_provider = os.getenv("LLM_PROVIDER", "local")
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
        
        # 显示当前 API Key 状态 (Compact status)
        status_cols = st.columns([1, 4])
        with status_cols[0]:
            st.markdown("🔑")
        with status_cols[1]:
            if selected_key == "qwen":
                if not os.getenv("DASHSCOPE_API_KEY"):
                    st.caption("🔴 :red[未配置 Key]")
                else:
                    st.caption("🟢 :green[Ready]")
            elif selected_key == "openai":
                if not os.getenv("OPENAI_API_KEY"):
                    st.caption("🔴 :red[未配置 Key]")
                else:
                    st.caption("🟢 :green[Ready]")
            elif selected_key == "gemini":
                if not os.getenv("GOOGLE_API_KEY"):
                    st.caption("🔴 :red[未配置 Key]")
                else:
                    st.caption("🟢 :green[Ready]")
            
        
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
        
