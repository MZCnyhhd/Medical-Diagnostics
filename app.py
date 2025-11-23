import streamlit as st
import os
import sys

# 设置页面配置必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="医疗诊断 AI 智能体",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    import asyncio
    from dotenv import load_dotenv
    from Main import generate_diagnosis
    from Utils.config import APIKEY_ENV_PATH
    import Utils.db as db
    
    # 加载环境变量 (强制覆盖，确保读取最新配置)
    load_dotenv(dotenv_path=APIKEY_ENV_PATH, override=True)
    
except Exception as e:
    st.error(f"应用启动失败，发生严重错误：\n{e}")
    st.stop()

# 自定义 CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6
    }
    .main-header {
        font-size: 2.5rem;
        color: #0e1117;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #262730;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .stTextArea textarea {
        font-size: 1rem;
    }
    /* 卡片样式 */
    .specialist-card {
        background-color: #ffffff;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #4CAF50;
    }
    .specialist-title {
        font-weight: bold;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # 初始化数据库
    import Utils.db as db
    db.init_db()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "diagnosis_result" not in st.session_state:
        st.session_state.diagnosis_result = None

    st.markdown('<h1 class="main-header">🏥 医疗诊断 AI 智能体</h1>', unsafe_allow_html=True)
    
    with st.sidebar:
        st.header("⚙️ 设置")
        
        # --- 模型切换功能 ---
        model_options = {
            "Local (DeepSeek-R1-Distill)": "local",
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
            help="选择用于诊断和对话的底层大模型"
        )
        
        # 更新环境变量
        selected_provider = model_options[selected_model_name]
        os.environ["LLM_PROVIDER"] = selected_provider
        if "llm_provider" not in st.session_state or st.session_state["llm_provider"] != selected_provider:
            st.session_state["llm_provider"] = selected_provider

        st.divider()

        st.header("配置与说明")
        st.info(
            """
            **工作原理：**
            系统将医疗报告分发给多个专科 AI 智能体（心脏科、心理科、肺科等）并行分析，
            最后由多学科团队整合意见，给出综合诊断建议。
            """
        )
        
        api_key = os.getenv("DASHSCOPE_API_KEY")
        if not api_key:
            st.warning("未检测到 DASHSCOPE_API_KEY 环境变量。")
            user_api_key = st.text_input("请输入 DashScope API Key:", type="password")
            if user_api_key:
                os.environ["DASHSCOPE_API_KEY"] = user_api_key
        else:
            st.success("DashScope API Key 已配置")
            
        st.divider()
        if st.button("🔄 更新知识库", help="将 KnowledgeBase 目录下的文档重新写入向量库"):
            with st.spinner("正在更新知识库..."):
                from ingest_knowledge import ingest_docs
                status = ingest_docs()
                if "失败" in status:
                    st.error(status)
                else:
                    st.success(status)

        st.divider()
        st.markdown("### 📜 历史诊断记录")
        history = db.get_history()
        if history:
            selected_history = st.selectbox(
                "查看过往病例",
                options=history,
                format_func=lambda x: f"{x['timestamp']} (ID: {x['id']})"
            )
            if selected_history:
                with st.expander("查看详情", expanded=False):
                    st.markdown("**原始报告**:")
                    st.text(selected_history['report_content'][:100] + "...")
                    st.markdown("**诊断结果**:")
                    st.markdown(selected_history['diagnosis_result'])
        else:
            st.info("暂无历史记录")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<h2 class="sub-header">📄 输入医疗报告</h2>', unsafe_allow_html=True)
        
        input_method = st.radio("选择输入方式", ["直接粘贴文本", "上传 TXT 文件", "选择示例报告"])
        
        medical_report = ""
        
        if input_method == "直接粘贴文本":
            medical_report = st.text_area("在此处粘贴医疗报告内容...", height=400)
        elif input_method == "上传 TXT 文件":
            uploaded_file = st.file_uploader("上传医疗报告 (.txt)", type=["txt"])
            if uploaded_file is not None:
                medical_report = uploaded_file.read().decode("utf-8")
                st.text_area("文件内容预览", value=medical_report, height=400, disabled=True)
        elif input_method == "选择示例报告":
            example_dir = os.path.join("Medical Reports", "Examples")
            if os.path.exists(example_dir):
                example_files = [f for f in os.listdir(example_dir) if f.endswith(".txt")]
                if example_files:
                    selected_example = st.selectbox("请选择一个示例报告", example_files)
                    if selected_example:
                        with open(os.path.join(example_dir, selected_example), "r", encoding="utf-8") as f:
                            medical_report = f.read()
                        st.text_area("示例报告内容", value=medical_report, height=400)
                else:
                    st.warning("未找到示例报告文件。")
            else:
                st.warning("示例报告目录不存在。")

        start_btn = st.button("🚀 开始多学科会诊", type="primary", use_container_width=True)

    # 定义聊天区域容器（放在底部，但提前定义以便引用）
    st.divider()
    st.markdown('<h2 class="sub-header">💬 专家咨询</h2>', unsafe_allow_html=True)
    chat_container = st.container()

    with col2:
        st.markdown('<h2 class="sub-header">🩺 诊断过程</h2>', unsafe_allow_html=True)
        
        # 占位符：用于显示各专科医生的分析过程
        specialist_placeholder = st.empty()
        
        if start_btn and medical_report:
             # 清空之前的会话和结果
            st.session_state.messages = []
            st.session_state.diagnosis_result = None

            if not os.getenv("DASHSCOPE_API_KEY") and not os.getenv("OPENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
                st.error("请先配置 API Key！")
            else:
                async def run_async_diagnosis():
                    gen = generate_diagnosis(medical_report)
                    try:
                        async for agent_name, response in gen:
                            if agent_name == "Final Diagnosis":
                                st.success("会诊完成！")
                                
                                # --- 修复：双重输出问题 ---
                                # 不再在 col2 中显示最终结果，而是直接流式输出到底部的 chat_container
                                
                                full_diagnosis = response
                                
                                # 在聊天区域显示最终诊断
                                with chat_container:
                                    with st.chat_message("assistant"):
                                        st.markdown("### 📋 多学科团队综合诊断")
                                        message_placeholder = st.empty()
                                        
                                        # 模拟流式打字机效果
                                        displayed_text = ""
                                        chunk_size = 10
                                        for i in range(0, len(full_diagnosis), chunk_size):
                                            chunk = full_diagnosis[i:i+chunk_size]
                                            displayed_text += chunk
                                            message_placeholder.markdown(displayed_text + "▌")
                                            await asyncio.sleep(0.02)
                                        message_placeholder.markdown(displayed_text)

                                # 保存结果到 Session State
                                st.session_state.diagnosis_result = full_diagnosis
                                st.session_state.messages.append({"role": "assistant", "content": f"### 📋 多学科团队综合诊断\n\n{full_diagnosis}"})
                                
                                # --- 新增：数据持久化 ---
                                db.save_consultation(medical_report, full_diagnosis)
                                
                            else:
                                # 显示专科医生的分析过程（保持在 col2）
                                with specialist_placeholder.container():
                                    st.markdown(f"""
                                    <div class="specialist-card">
                                        <div class="specialist-header">{agent_name} 正在分析...</div>
                                        <div class="specialist-content">{response}</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    await asyncio.sleep(0.5)

                    except Exception as e:
                        st.error(f"发生错误: {e}")

                asyncio.run(run_async_diagnosis())

    # ---------------------------------------------------------
    # 聊天问答区域内容渲染
    # ---------------------------------------------------------
    # 如果刚点击了开始按钮，说明上面已经流式输出了诊断结果，这里就不需要再渲染历史记录了（否则会重复）
    if not start_btn:
        with chat_container:
            # 显示聊天记录
            for message in st.session_state.messages:
                if message["role"] != "system":
                     with st.chat_message(message["role"]):
                        st.markdown(message["content"])

    if prompt := st.chat_input("对诊断结果有疑问？请在此提问..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # 生成回复
        if st.session_state.diagnosis_result:
            # 重新构建 prompt
            context = f"基于以下诊断结果：\n{st.session_state.diagnosis_result}\n\n用户问题：{prompt}"
            
            # 获取模型
            from Utils.llm_factory import get_chat_model
            # 使用默认配置的模型 (通常是 local 或 qwen)
            chat_model = get_chat_model()
            
            with chat_container:
                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    try:
                        # 使用 stream 实现流式输出
                        for chunk in chat_model.stream(context):
                            content = getattr(chunk, "content", str(chunk))
                            full_response += content
                            # 实时显示（带光标）
                            response_placeholder.markdown(full_response + "▌")
                        
                        # 生成完成后，进行后处理（折叠思考过程、清理 token）
                        
                        # --- 优化输出显示 ---
                        import re
                        # 1. 提取思考过程
                        thought_content = None
                        
                        # 尝试匹配标准的 <think>...</think>
                        think_match = re.search(r'<think>(.*?)</think>', full_response, re.DOTALL)
                        if think_match:
                            thought_content = think_match.group(1).strip()
                            full_response = full_response.replace(think_match.group(0), '').strip()
                        else:
                            # 处理只有 </think> 的情况
                            end_think_match = re.search(r'(.*?)</think>', full_response, re.DOTALL)
                            if end_think_match:
                                thought_content = end_think_match.group(1).strip()
                                full_response = full_response.replace(end_think_match.group(0), '').strip()

                        if thought_content:
                            with st.expander("💭 思考过程"):
                                st.markdown(thought_content)
                        
                        # 2. 清理可能残留的特殊 token
                        full_response = re.sub(r'<\|.*?\|>', '', full_response).strip()

                        # 显示最终处理后的结果（不带光标）
                        response_placeholder.markdown(full_response)
                        
                        # --- 新增：导出功能 ---
                        st.divider()
                        col_pdf, col_docx = st.columns(2)
                        from Utils.export_utils import generate_pdf, generate_docx
                        
                        with col_pdf:
                            pdf_file = generate_pdf(full_response)
                            st.download_button(
                                label="📄 下载 PDF 报告",
                                data=pdf_file,
                                file_name="diagnosis_report.pdf",
                                mime="application/pdf"
                            )
                            
                        with col_docx:
                            docx_file = generate_docx(full_response)
                            st.download_button(
                                label="📝 下载 Word 报告",
                                data=docx_file,
                                file_name="diagnosis_report.docx",
                                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                            )

                    except Exception as e:
                        st.error(f"回复生成失败: {e}")
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.warning("请先完成诊断再提问。")

if __name__ == "__main__":
    main()
