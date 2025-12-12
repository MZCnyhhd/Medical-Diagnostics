import streamlit as st
import os
import sys
import base64
from io import BytesIO

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
    from src.core.settings import get_settings
    from src.services.cache import get_cache
    import src.services.db as db
    
    # 加载环境变量 (强制覆盖，确保读取最新配置)
    load_dotenv(dotenv_path=APIKEY_ENV_PATH, override=True)
    
    # 初始化配置
    settings = get_settings()
    
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
    # 清空上传的图片
    if "uploaded_image" in st.session_state:
        st.session_state.uploaded_image = None


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """从 PDF 文件中提取文本"""
    try:
        import pypdf
        pdf_reader = pypdf.PdfReader(BytesIO(file_bytes))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
        return text.strip()
    except ImportError:
        st.error("请安装 pypdf 库: pip install pypdf")
        return ""
    except Exception as e:
        st.error(f"PDF 解析失败: {e}")
        return ""


def process_uploaded_file(uploaded_file) -> tuple[str, bytes | None]:
    """
    处理上传的文件，返回 (文本内容, 图片字节数据)
    - 文本文件返回 (text, None)
    - 图片文件返回 ("", image_bytes)
    """
    if uploaded_file is None:
        return "", None
    
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.read()
    
    # 文本文件: txt, md
    if file_name.endswith(('.txt', '.md', '.markdown')):
        try:
            return file_bytes.decode("utf-8"), None
        except UnicodeDecodeError:
            return file_bytes.decode("gbk", errors="ignore"), None
    
    # PDF 文件
    elif file_name.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes), None
    
    # 图片文件: png, jpg, jpeg
    elif file_name.endswith(('.png', '.jpg', '.jpeg')):
        return "", file_bytes
    
    else:
        st.warning(f"不支持的文件格式: {file_name}")
        return "", None

def main():
    # 初始化数据库
    db.init_db()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "diagnosis_result" not in st.session_state:
        st.session_state.diagnosis_result = None

    if "specialist_logs" not in st.session_state:
        st.session_state.specialist_logs = []

    # 渲染侧边栏
    render_sidebar()
    
    # --- 历史记录区域美化 ---
    st.markdown('<div class="sub-header">📜 历史诊断记录</div>', unsafe_allow_html=True)
    history = db.get_history()
    if history:
        selected_history = st.selectbox(
            "查看过往病例",
            options=history,
            format_func=lambda x: f"🕒 {x['timestamp']} (ID: {x['id']})",
            label_visibility="collapsed"
        )
        if selected_history:
            with st.expander("诊断记录-内容提取", expanded=False):
                st.markdown("### 📄 病例报告")
                # 使用 markdown 引用块统一格式
                st.markdown(f"> {selected_history['report_content']}")
                
                st.markdown("### 🩺 诊断结果")
                # 统一使用标准文本格式
                st.markdown(selected_history['diagnosis_result'])
    else:
        st.info("暂无历史记录")

    # --- 第一部分：输入病例报告（上下布局，单栏） ---
    st.markdown('<div class="sub-header">📄 输入病例报告</div>', unsafe_allow_html=True)
    
    # 添加 on_change 回调以在切换输入方式时清空结果
    # 修改：横向排列，隐藏标签，移除直接粘贴选项
    input_method = st.radio(
        "选择输入方式", 
        ["上传文件", "选择示例报告"], 
        on_change=clear_results,
        horizontal=True,
        label_visibility="collapsed"
    )
    
    medical_report = ""
    uploaded_image_bytes = None
    
    if input_method == "上传文件":
        st.caption("📎 支持格式：TXT、PDF、Markdown、PNG、JPG")
        uploaded_file = st.file_uploader(
            "", 
            type=["txt", "pdf", "md", "markdown", "png", "jpg", "jpeg"], 
            on_change=clear_results,
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            medical_report, uploaded_image_bytes = process_uploaded_file(uploaded_file)
            # 保存图片到 session_state 供后续使用
            if uploaded_image_bytes:
                st.session_state.uploaded_image = uploaded_image_bytes
    elif input_method == "选择示例报告":
        example_dir = os.path.join("data", "medical_reports", "Examples")
        if os.path.exists(example_dir):
            example_files = [f for f in os.listdir(example_dir) if f.endswith(".txt")]
            if example_files:
                # 文件名到中文名的映射
                file_display_names = {
                    "example_01_diarrhea.txt": "腹泻病例",
                    "example_02_asthma.txt": "哮喘病例",
                    "example_03_headache.txt": "头痛病例"
                }
                # --- 修改：添加 on_change 回调，隐藏标签，使用中文显示 ---
                selected_example = st.selectbox(
                    "请选择一个示例报告", 
                    example_files, 
                    format_func=lambda x: file_display_names.get(x, x),
                    on_change=clear_results,
                    label_visibility="collapsed"
                )
                # 修复：确保读取文件
                if selected_example:
                    with open(os.path.join(example_dir, selected_example), "r", encoding="utf-8") as f:
                        medical_report = f.read()
                else:
                    st.warning("未找到示例报告文件。")
            else:
                st.warning("示例报告目录不存在。")

    # --- 第二部分：报告内容（用可折叠下拉框） ---
    
    # 获取 session_state 中的图片数据
    has_image = st.session_state.get("uploaded_image") is not None
    
    # 使用可折叠的下拉框显示报告内容
    with st.expander("病例报告-内容提取", expanded=False):
        if medical_report:
            st.text_area("文件内容预览", value=medical_report, height=300, disabled=True, label_visibility="collapsed")
        elif has_image:
            st.image(st.session_state.uploaded_image, caption="上传的医疗图片", use_container_width=True)
            st.info("💡 图片将通过视觉模型进行分析")
        else:
            st.info("请在上方选择或上传报告")
    
    # 开始诊断按钮
    start_btn = st.button("🚀 开始诊断", type="primary", use_container_width=True)
        
    # --- 状态显示区域 ---
    status_placeholder = st.empty()

    # 如果已有诊断结果且不在运行中，显示完成状态
    if st.session_state.diagnosis_result:
        with status_placeholder:
            st.success("✅ 多学科会诊已完成")

    # --- 第三部分：诊断过程区域（移到下方，全宽） ---
    with st.expander("诊断过程-内容记录", expanded=False):
        # 诊断过程容器（带边框）
        process_container = st.container(height=400, border=True)
            
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
    
    # 验证是否有输入内容
    has_valid_input = medical_report or has_image
    
    if start_btn and not has_valid_input:
        st.error("请先上传或选择一份医疗报告/图片！")

    # --- 执行诊断 ---
    if start_btn and has_valid_input:
        # 检查 API Key
        if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            st.error("未检测到有效的 API Key，请先配置环境变量！")
        else:
            # 如果是图片，先用视觉模型提取文本描述
            if has_image and not medical_report:
                with st.spinner("🔍 正在分析医疗图片..."):
                    from src.services.llm import analyze_medical_image
                    medical_report = analyze_medical_image(st.session_state.uploaded_image)
                    if not medical_report:
                        st.error("图片分析失败，请重试或上传文本格式的报告")
                    else:
                        st.success("✅ 图片分析完成")
            
            # 运行异步诊断任务
            if medical_report:
                asyncio.run(run_async_diagnosis())

        # 移除原有的结果显示区域，移到页面底部居中显示


    # 诊断结果居中显示区域
    if st.session_state.diagnosis_result and not start_btn:
        st.markdown('<div class="sub-header">📋 输出诊断结果</div>', unsafe_allow_html=True)
        
        # 诊断结果使用可折叠下拉框
        with st.expander("诊断结果-内容提取", expanded=True):
            st.markdown(f"{st.session_state.diagnosis_result}")
        
        from src.tools.export import generate_markdown
        
        # 重新构建报告内容用于下载
        report_content = f"# 医疗诊断报告\n\n## 病例报告\n{medical_report}\n\n## 诊断结果\n{st.session_state.diagnosis_result}"

        md_file = generate_markdown(report_content)
        st.download_button(
            label="📝 下载 Markdown 文件报告",
            data=md_file,
            file_name="diagnosis_report.md",
            mime="text/markdown",
            use_container_width=True,
            key="download_md_btn_persistent"
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
