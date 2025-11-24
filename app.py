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
    from src.core.orchestrator import generate_diagnosis
    from src.core.config import APIKEY_ENV_PATH
    import src.services.db as db
    
    # 加载环境变量 (强制覆盖，确保读取最新配置)
    load_dotenv(dotenv_path=APIKEY_ENV_PATH, override=True)
    
except Exception as e:
    st.error(f"应用启动失败，发生严重错误：\n{e}")
    st.stop()

# 自定义 CSS
from src.ui.styles import get_css
st.markdown(get_css(), unsafe_allow_html=True)

from src.ui.sidebar import render_sidebar

def main():
    # 初始化数据库
    db.init_db()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "diagnosis_result" not in st.session_state:
        st.session_state.diagnosis_result = None

    if "specialist_logs" not in st.session_state:
        st.session_state.specialist_logs = []

    st.markdown('<h1 class="main-header">🏥 医疗诊断 AI 智能体</h1>', unsafe_allow_html=True)
    
    # 渲染侧边栏
    render_sidebar()
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
            # --- 新增：语音输入演示按钮 ---
            if st.button("🎙️ 模拟语音输入 (演示用)", help="点击模拟患者口述病情"):
                st.session_state.voice_input = "医生你好，我最近一周总是感觉头痛，尤其是下午的时候，太阳穴这边跳着疼。而且有时候会恶心，想吐但吐不出来。以前有高血压，不知道有没有关系。"
            
            default_text = st.session_state.get("voice_input", "")
            medical_report = st.text_area("在此处粘贴医疗报告内容...", value=default_text, height=400)
        elif input_method == "上传 TXT 文件":
            uploaded_file = st.file_uploader("上传医疗报告 (.txt)", type=["txt"])
            if uploaded_file is not None:
                medical_report = uploaded_file.read().decode("utf-8")
                st.text_area("文件内容预览", value=medical_report, height=400, disabled=True)
        elif input_method == "选择示例报告":
            example_dir = os.path.join("data", "medical_reports", "Examples")
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
        process_container = st.container()
        
        # 渲染历史日志
        with process_container:
            for log_html in st.session_state.specialist_logs:
                st.markdown(log_html, unsafe_allow_html=True)
        
        if start_btn and medical_report:
             # 清空之前的会话、结果和日志
            st.session_state.messages = []
            st.session_state.diagnosis_result = None
            st.session_state.specialist_logs = []

            if not os.getenv("DASHSCOPE_API_KEY") and not os.getenv("OPENAI_API_KEY") and not os.getenv("GOOGLE_API_KEY"):
                st.error("请先配置 API Key！")
            else:
                async def run_async_diagnosis():
                    # 使用 st.status 显示整体进度
                    with st.status("🚀 正在启动多学科会诊系统...", expanded=True) as status_container:
                        gen = generate_diagnosis(medical_report)
                        try:
                            async for agent_name, response in gen:
                                if agent_name == "Status":
                                    # 更新状态容器的标题
                                    status_container.update(label=response, state="running")
                                    # 也可以在内部打印日志
                                    st.write(f"ℹ️ {response}")
                                
                                elif agent_name == "Final Diagnosis":
                                    status_container.update(label="✅ 会诊完成！", state="complete", expanded=False)
                                    
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
                                    # 在 status 内部显示简略信息
                                    st.markdown(f"**{agent_name}** 正在分析...")
                                    
                                    # 在外部 container 显示详细卡片
                                    log_html = f"""
                                    <div class="specialist-card">
                                        <div class="specialist-header">{agent_name} 正在分析...</div>
                                        <div class="specialist-content">{response}</div>
                                    </div>
                                    """
                                    # 保存到 session state
                                    st.session_state.specialist_logs.append(log_html)
                                    
                                    # 实时渲染
                                    with process_container:
                                        st.markdown(log_html, unsafe_allow_html=True)
                                        
                                    await asyncio.sleep(0.5)

                        except Exception as e:
                            status_container.update(label="❌ 发生错误", state="error")
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
            from src.services.llm import get_chat_model
            # 强制使用当前选择的模型
            current_provider = st.session_state.get("llm_provider", "qwen")
            chat_model = get_chat_model(override_provider=current_provider)
            
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
                        from src.tools.export import generate_pdf, generate_docx
                        
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
