import streamlit as st
import os

def render_sidebar():
    with st.sidebar:
        st.header("⚙️ 设置")
        
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
            "选择 AI 模型",
            options=list(model_options.keys()),
            index=default_index,
            key="model_selector"
        )
        
        # 更新环境变量 (注意：os.environ 的修改只在当前进程有效，
        # 如果需要持久化，通常需要写入 .env 文件，或者每次启动时读取)
        # 这里我们简单地更新 os.environ，以便后续 get_chat_model 读取
        selected_key = model_options[selected_model_name]
        os.environ["LLM_PROVIDER"] = selected_key
        
        # 显示当前 API Key 状态 (脱敏显示)
        st.caption(f"当前选择: {selected_key}")
        
        if selected_key == "qwen":
            if not os.getenv("DASHSCOPE_API_KEY"):
                st.error("未检测到 DASHSCOPE_API_KEY")
            else:
                st.success("DashScope API Key 已配置")
        elif selected_key == "openai":
            if not os.getenv("OPENAI_API_KEY"):
                st.error("未检测到 OPENAI_API_KEY")
            else:
                st.success("OpenAI API Key 已配置")
        elif selected_key == "gemini":
            if not os.getenv("GOOGLE_API_KEY"):
                st.error("未检测到 GOOGLE_API_KEY")
            else:
                st.success("Google API Key 已配置")
            
        st.divider()
        if st.button("🔄 更新知识库", help="将 data/knowledge_base 目录下的文档重新写入向量库"):
            with st.spinner("正在更新知识库..."):
                from src.scripts.ingest_knowledge import ingest_docs
                status = ingest_docs()
                if "成功" in status:
                    st.success(status)
                else:
                    st.error(status)
        
        st.divider()
        st.markdown("### 关于")
        st.info(
            "这是一个基于多智能体协作的医疗诊断系统。\n\n"
            "它模拟了多学科会诊流程，由不同专科的 AI 医生共同分析病例，"
            "并由主治医生汇总最终诊断意见。"
        )
