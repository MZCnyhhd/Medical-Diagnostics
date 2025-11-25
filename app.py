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

def clear_results():
    """清空诊断结果和日志"""
    st.session_state.diagnosis_result = None
    st.session_state.specialist_logs = []
    st.session_state.messages = []

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
    
    # --- 历史记录区域美化 ---
    st.markdown('<h3 style="color: #2c3e50; font-weight: 600;">📜 历史诊断记录</h3>', unsafe_allow_html=True)
    history = db.get_history()
    if history:
        selected_history = st.selectbox(
            "查看过往病例",
            options=history,
            format_func=lambda x: f"🕒 {x['timestamp']} (ID: {x['id']})",
            label_visibility="collapsed"
        )
        if selected_history:
            with st.expander("📋 查看详情", expanded=False):
                st.markdown("### 📄 原始报告")
                # --- 修改：显示完整报告内容 ---
                st.markdown(f"```\n{selected_history['report_content']}\n```")
                st.markdown("### 🩺 诊断结果")
                st.markdown(selected_history['diagnosis_result'])
    else:
        st.info("暂无历史记录")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown('<h2 class="sub-header">📄 输入医疗报告</h2>', unsafe_allow_html=True)
        
        # 添加 on_change 回调以在切换输入方式时清空结果
        # 修改：横向排列，隐藏标签，移除直接粘贴选项
        input_method = st.radio(
            "选择输入方式", 
            ["上传 TXT 文件", "选择示例报告"], 
            on_change=clear_results,
            horizontal=True,
            label_visibility="collapsed"
        )
        
        medical_report = ""
        
        if input_method == "上传 TXT 文件":
            uploaded_file = st.file_uploader("上传医疗报告 (.txt)", type=["txt"], on_change=clear_results)
            if uploaded_file is not None:
                medical_report = uploaded_file.read().decode("utf-8")
        elif input_method == "选择示例报告":
            example_dir = os.path.join("data", "medical_reports", "Examples")
            if os.path.exists(example_dir):
                example_files = [f for f in os.listdir(example_dir) if f.endswith(".txt")]
                if example_files:
                    # --- 修改：添加 on_change 回调，隐藏标签 ---
                    selected_example = st.selectbox(
                        "请选择一个示例报告", 
                        example_files, 
                        on_change=clear_results,
                        label_visibility="collapsed"
                    )
                    if selected_example:
                        with open(os.path.join(example_dir, selected_example), "r", encoding="utf-8") as f:
                            medical_report = f.read()
                else:
                    st.warning("未找到示例报告文件。")
            else:
                st.warning("示例报告目录不存在。")

    with col2:
        st.markdown('<h2 class="sub-header">🩺 诊断过程</h2>', unsafe_allow_html=True)
        
        # --- 新增：状态显示区域 (位于折叠面板上方) ---
        status_placeholder = st.empty()

        # 如果已有诊断结果且不在运行中，显示完成状态
        if st.session_state.diagnosis_result:
            with status_placeholder:
                st.success("✅ 多学科会诊已完成")

    # --- 第二行：内容展示区域 (对齐) ---
    col3, col4 = st.columns([1, 1])

    # 右侧：详细诊断过程 (先定义以便函数可用)
    with col4:
        # --- 修改：使用 scrollable container ---
        with st.expander("🩺 详细诊断过程", expanded=True):
            # 设置固定高度，使其可滚动
            process_container = st.container(height=400)
            
            # --- 修复：重新渲染历史日志 ---
            for log in st.session_state.specialist_logs:
                with process_container:
                    with st.chat_message(log["agent"], avatar="👨‍⚕️"):
                        st.write(f"**{log['agent']}**: {log['content']}")

            # --- 定义异步诊断任务 ---
            async def run_async_diagnosis():
                # 使用 status_placeholder 显示整体进度
                with status_placeholder:
                    with st.status("🚀 正在启动多学科会诊系统...", expanded=True) as status_container:
                        gen = generate_diagnosis(medical_report)
                        full_diagnosis = None
                        try:
                            async for agent_name, response in gen:
                                if agent_name == "Status":
                                    status_container.update(label=response, state="running")
                                elif agent_name == "Final Diagnosis":
                                    full_diagnosis = response
                                    st.session_state.diagnosis_result = full_diagnosis
                                    # 保存到数据库
                                    db.save_consultation(medical_report, full_diagnosis)
                                    status_container.update(label="✅ 会诊完成", state="complete", expanded=False)
                                else:
                                    # 专家意见
                                    st.session_state.specialist_logs.append({
                                        "agent": agent_name,
                                        "content": response
                                    })
                                    with process_container:
                                        with st.chat_message(agent_name, avatar="👨‍⚕️"):
                                            st.write(f"**{agent_name}**: {response}")
                        except Exception as e:
                            st.error(f"诊断过程中发生错误: {e}")
                            status_container.update(label="❌ 诊断失败", state="error")

                if full_diagnosis:
                    # 诊断完成后，强制刷新页面以进入持久化显示模式
                    st.rerun()

    # 左侧：报告内容 + 按钮 + 结果
    with col3:
        # 显示报告内容 (如果有)
        if medical_report:
            with st.expander("📄 报告内容", expanded=True):
                st.text_area("文件内容预览", value=medical_report, height=400, disabled=True, label_visibility="collapsed")
        else:
            # 占位符，保持对齐 (可选，或者直接显示空 expander)
            with st.expander("📄 报告内容", expanded=True):
                st.info("请在上方选择或上传报告")

        # 确保 start_btn 点击时能读取到 report
        start_btn = st.button("🚀 开始诊断", type="primary", use_container_width=True)
        
        if start_btn and not medical_report:
            st.error("请先上传或选择一份医疗报告！")

        # --- 执行诊断 ---
        if start_btn and medical_report:
            # 检查 API Key
            if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
                st.error("未检测到有效的 API Key，请先配置环境变量！")
            else:
                # 运行异步任务
                asyncio.run(run_async_diagnosis())

        # --- 结果显示区域 (持久化) ---
        # 移到 col1 中，按钮下方
        if st.session_state.diagnosis_result and not start_btn:
            st.divider()
            with st.chat_message("assistant"):
                st.markdown(f"### 📋 诊断结果\n\n{st.session_state.diagnosis_result}")
        
            col_pdf, col_docx = st.columns(2)
            from src.tools.export import generate_pdf, generate_docx
            
            # 重新构建报告内容用于下载
            report_content = f"【病例报告】\n{medical_report}\n\n【诊断结果】\n{st.session_state.diagnosis_result}"

            with col_pdf:
                pdf_file = generate_pdf(report_content)
                st.download_button(
                    label="📄 下载 PDF 报告",
                    data=pdf_file,
                    file_name="diagnosis_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="download_pdf_btn_persistent"
                )
                
            with col_docx:
                docx_file = generate_docx(report_content)
                st.download_button(
                    label="📝 下载 Word 报告",
                    data=docx_file,
                    file_name="diagnosis_report.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    key="download_docx_btn_persistent"
                )


    # Floating chat assistant popover
    with st.popover(" ", help="咨询专家助手"):
        # 准备 Chat Component 所需的参数
        from src.ui.chat_component import render_chat_component
        
        # 1. 获取 API Key 和 Base URL
        # 默认使用 Qwen (DashScope)
        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        model = os.getenv("QWEN_MODEL", "qwen-max")
        
        # 如果配置了 OpenAI 且没有 DashScope，或者用户强制指定了 OpenAI (这里简化逻辑，优先 DashScope 因为项目默认是 Qwen)
        # 实际项目中可以根据 st.session_state.get("llm_provider") 来判断
        provider = st.session_state.get("llm_provider", "qwen")
        
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = "https://api.openai.com/v1/chat/completions"
            model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        # 2. 准备 System Prompt (包含诊断结果)
        system_prompt = "你是一个专业的医疗 AI 助手。请根据用户的提问进行解答。"
        if st.session_state.diagnosis_result:
            system_prompt += f"\n\n以下是该患者的多学科综合诊断结果，请以此为依据回答用户问题：\n{st.session_state.diagnosis_result}"
        
        if api_key:
            render_chat_component(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=system_prompt
            )
        else:
            st.error("未配置 API Key，无法启动聊天助手。")

if __name__ == "__main__":
    main()
