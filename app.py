"""
医疗诊断 AI 智能体 - Streamlit 应用入口
========================================

本文件是整个医疗诊断系统的前端入口，基于 Streamlit 框架构建。

应用功能概述：
1. 用户认证：安全的登录系统，支持多角色权限管理
2. 病例输入：支持文件上传（TXT/PDF/MD/图片）和示例选择
3. 智能诊断：调用多学科会诊系统进行 AI 诊断
4. 结果展示：实时显示诊断进度和各专科意见
5. 历史管理：查看和管理历史诊断记录
6. 报告导出：支持 Markdown 格式导出
7. 智能问答：悬浮聊天助手，针对诊断结果进行追问
8. 用户管理：管理员可添加/删除用户

用户角色：
- admin：系统管理员，拥有所有权限
- doctor：医生，可以进行诊断和查看历史
- nurse：护士，可以上传报告和查看历史

技术栈：
- Streamlit：Python Web 应用框架，快速构建数据应用
- streamlit-authenticator：用户认证库
- asyncio：Python 异步编程，支持并发诊断
- SQLite：轻量级数据库，存储历史记录
- bcrypt：密码哈希加密

页面布局：
```
+------------------+------------------------+
|                  |   📜 历史诊断记录      |
|    侧边栏        +------------------------+
|  - 用户信息      |   📄 输入病例报告      |
|  - 系统介绍      |   [上传/示例选择]      |
|  - 模型选择      +------------------------+
|  - 知识库管理    |   🚀 开始诊断          |
|  - 用户管理      +------------------------+
|                  |   诊断过程记录         |
|                  +------------------------+
|                  |   📋 诊断结果          |
+------------------+------------------------+
                                    [💬 聊天助手]
```

启动方式：
```bash
streamlit run app.py
```

默认账户：
- 管理员：admin / admin123
- 医生：doctor / doctor123
- 护士：nurse / nurse123
"""

# ==================== 标准库导入 ====================
# streamlit：Web 应用框架
import streamlit as st
# os：文件和环境变量操作
import os
# sys：系统相关操作
import sys
# base64：Base64 编码（图片处理）
import base64
# BytesIO：内存中的字节流（文件处理）
from io import BytesIO

# ==================== Streamlit 页面配置 ====================
# 注意：set_page_config 必须是第一个 Streamlit 命令
# 否则会报错 "set_page_config() can only be called once per app"
st.set_page_config(
    # 页面标题（浏览器标签页显示）
    page_title="医疗诊断 AI 智能体",
    # 页面图标（浏览器标签页显示）
    page_icon="🏥",
    # 页面布局：wide 表示使用全屏宽度
    layout="wide",
    # 侧边栏初始状态：expanded 表示默认展开
    initial_sidebar_state="expanded",
)

# ==================== 核心模块导入（带错误处理）====================
# 将核心导入放在 try-except 中，确保即使导入失败也能显示友好的错误信息
try:
    # asyncio：Python 异步编程库
    import asyncio
    # dotenv：加载 .env 文件中的环境变量
    from dotenv import load_dotenv
    # 诊断流程编排器：执行多学科会诊
    from src.core.orchestrator import generate_diagnosis
    # 配置管理：API Key 路径和全局设置
    from src.core.settings import get_settings, APIKEY_ENV_PATH
    # 缓存服务
    from src.services.cache import get_cache
    # 数据库服务：存储历史诊断记录
    import src.services.db as db
    # 用户认证服务
    from src.services.auth import (
        get_authenticator,
        render_login_page,
        render_user_info_sidebar,
        get_user_role,
        render_user_management
    )
    
    # 加载环境变量配置文件
    # override=True 表示强制覆盖已存在的环境变量
    # 确保每次启动都读取最新的配置
    load_dotenv(dotenv_path=APIKEY_ENV_PATH, override=True)
    
    # 初始化全局配置
    # 会验证 API Key 配置、检查路径等
    settings = get_settings()
    
except Exception as e:
    # 如果核心模块导入失败，显示错误信息并停止应用
    st.error(f"应用启动失败，发生严重错误：\n{e}")
    st.stop()

# ==================== 自定义 CSS 样式 ====================
# 从 UI 模块导入样式表
from src.ui.styles import get_css
# 注入自定义 CSS
# unsafe_allow_html=True 允许渲染 HTML 和 CSS
st.markdown(get_css(), unsafe_allow_html=True)

# ==================== 侧边栏组件导入 ====================
from src.ui.sidebar import render_sidebar


def clear_results():
    """
    清空诊断结果和日志
    
    当用户切换输入方式或上传新文件时调用此函数，
    清除之前的诊断结果，确保界面显示正确。
    
    清空的内容：
    - diagnosis_result：最终诊断结果
    - specialist_logs：各专科医生的日志
    - messages：聊天记录
    - uploaded_image：上传的图片
    """
    # 清空诊断结果
    st.session_state.diagnosis_result = None
    # 清空专科医生日志
    st.session_state.specialist_logs = []
    # 清空聊天消息
    st.session_state.messages = []
    # 清空上传的图片（如果存在）
    if "uploaded_image" in st.session_state:
        st.session_state.uploaded_image = None


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """
    从 PDF 文件中提取文本
    
    使用 pypdf 库解析 PDF 文件，提取其中的文本内容。
    支持多页 PDF，会将所有页面的文本合并。
    
    Args:
        file_bytes (bytes): PDF 文件的字节数据
    
    Returns:
        str: 提取的文本内容
            - 成功：返回 PDF 中的文本
            - 失败：返回空字符串并显示错误信息
    
    依赖：
        需要安装 pypdf 库：pip install pypdf
    """
    try:
        # 导入 pypdf 库
        import pypdf
        # 从字节数据创建 PDF Reader
        # BytesIO 将字节转换为文件对象
        pdf_reader = pypdf.PdfReader(BytesIO(file_bytes))
        # 用于存储提取的文本
        text = ""
        # 遍历所有页面
        for page in pdf_reader.pages:
            # 提取当前页的文本，如果为 None 则使用空字符串
            text += page.extract_text() or ""
        # 返回去除首尾空白的文本
        return text.strip()
    except ImportError:
        # pypdf 未安装
        st.error("请安装 pypdf 库: pip install pypdf")
        return ""
    except Exception as e:
        # 其他解析错误
        st.error(f"PDF 解析失败: {e}")
        return ""


def process_uploaded_file(uploaded_file) -> tuple[str, bytes | None]:
    """
    处理上传的文件，返回文本内容或图片数据
    
    支持的文件类型：
    - 文本文件：.txt, .md, .markdown
    - PDF 文件：.pdf
    - 图片文件：.png, .jpg, .jpeg
    
    Args:
        uploaded_file: Streamlit 的 UploadedFile 对象
            - 包含文件名、文件内容等信息
            - 如果为 None，表示没有上传文件
    
    Returns:
        tuple[str, bytes | None]: (文本内容, 图片字节数据)
            - 文本文件：返回 (文本内容, None)
            - 图片文件：返回 ("", 图片字节数据)
            - 不支持的格式：返回 ("", None)
    """
    # 检查是否有上传文件
    if uploaded_file is None:
        return "", None
    
    # 获取文件名（转为小写以便比较）
    file_name = uploaded_file.name.lower()
    # 读取文件内容
    file_bytes = uploaded_file.read()
    
    # ========== 处理文本文件：txt, md, markdown ==========
    if file_name.endswith(('.txt', '.md', '.markdown')):
        try:
            # 尝试使用 UTF-8 编码解码
            return file_bytes.decode("utf-8"), None
        except UnicodeDecodeError:
            # 如果 UTF-8 解码失败，尝试 GBK 编码（中文 Windows 常用）
            return file_bytes.decode("gbk", errors="ignore"), None
    
    # ========== 处理 PDF 文件 ==========
    elif file_name.endswith('.pdf'):
        # 调用 PDF 解析函数
        return extract_text_from_pdf(file_bytes), None
    
    # ========== 处理图片文件：png, jpg, jpeg ==========
    elif file_name.endswith(('.png', '.jpg', '.jpeg')):
        # 图片文件直接返回字节数据，后续由视觉模型处理
        return "", file_bytes
    
    # ========== 不支持的文件格式 ==========
    else:
        st.warning(f"不支持的文件格式: {file_name}")
        return "", None


def main():
    """
    应用主函数
    
    这是 Streamlit 应用的主入口，负责：
    1. 用户认证检查
    2. 初始化数据库和会话状态
    3. 渲染侧边栏
    4. 渲染主界面各个区域
    5. 处理用户交互和诊断流程
    """
    # ==================== 用户认证 ====================
    # 检查用户是否已登录
    username, authentication_status, name = render_login_page()
    
    # 如果未登录，停止执行后续代码
    if not authentication_status:
        return
    
    # 获取认证器实例（用于登出等操作）
    authenticator = get_authenticator()
    
    # ==================== 初始化 ====================
    # 初始化数据库（创建表，如果不存在）
    db.init_db()

    # 初始化 Streamlit 会话状态
    # session_state 是 Streamlit 的全局状态存储
    # 用于在页面刷新之间保持数据
    
    # 聊天消息列表
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # 诊断结果
    if "diagnosis_result" not in st.session_state:
        st.session_state.diagnosis_result = None

    # 专科医生日志列表
    if "specialist_logs" not in st.session_state:
        st.session_state.specialist_logs = []

    # ==================== 渲染侧边栏 ====================
    # 侧边栏包含：用户信息、系统介绍、模型选择、知识库管理等
    render_sidebar()
    
    # 在侧边栏显示用户信息和登出按钮
    render_user_info_sidebar(authenticator, username)

    # ==================== 顶部入口：用户管理（方案A） ====================
    if "active_page" not in st.session_state:
        st.session_state.active_page = "main"

    is_admin = get_user_role(username) == "admin"

    try:
        active_page = st.query_params.get("page", "main")
    except Exception:
        active_page = st.experimental_get_query_params().get("page", ["main"])[0]

    st.session_state.active_page = active_page

    if is_admin:
        st.markdown(
            """
            <style>
            a.user-mgmt-top-link {
                position: fixed;
                top: 4.2rem;
                right: 1.0rem;
                z-index: 10000;
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.55rem 0.9rem;
                border-radius: 10px;
                background: #ffffff;
                border: 1px solid #e2e8f0;
                box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
                color: #0f172a;
                font-weight: 600;
                text-decoration: none;
                user-select: none;
            }
            a.user-mgmt-top-link:hover {
                border-color: #cbd5e1;
                box-shadow: 0 10px 22px rgba(0, 0, 0, 0.12);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.active_page == "user_management":
            st.markdown(
                '<a class="user-mgmt-top-link" href="?page=main" target="_self">← 返回</a>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<a class="user-mgmt-top-link" href="?page=user_management" target="_self">👥 用户管理</a>',
                unsafe_allow_html=True,
            )

    # 用户管理页（主区域渲染）
    if st.session_state.active_page == "user_management":
        render_user_management()
        return
    
    # ==================== 历史记录区域 ====================
    # 使用自定义样式的子标题
    st.markdown('<div class="sub-header">📜 历史诊断记录</div>', unsafe_allow_html=True)
    # 从数据库获取历史记录
    history = db.get_history()
    
    if history:
        # 有历史记录，显示选择器
        selected_history = st.selectbox(
            "查看过往病例",
            options=history,
            # 自定义显示格式：时间戳 + ID
            format_func=lambda x: f"🕒 {x['timestamp']} (ID: {x['id']})",
            # 隐藏标签
            label_visibility="collapsed"
        )
        if selected_history:
            # 显示选中的历史记录详情
            with st.expander("诊断记录-内容提取", expanded=False):
                st.markdown("### 📄 病例报告")
                # 使用 Markdown 引用块显示报告内容
                st.markdown(f"> {selected_history['report_content']}")
                
                st.markdown("### 🩺 诊断结果")
                st.markdown(selected_history['diagnosis_result'])
    else:
        # 没有历史记录
        st.info("暂无历史记录")

    # ==================== 病例输入区域 ====================
    st.markdown('<div class="sub-header">📄 输入病例报告</div>', unsafe_allow_html=True)
    
    # 输入方式选择（横向排列的单选按钮）
    # on_change 回调：切换时清空之前的结果
    input_method = st.radio(
        "选择输入方式", 
        ["上传病例报告", "示例病例报告"], 
        on_change=clear_results,
        horizontal=True,  # 横向排列
        label_visibility="collapsed"  # 隐藏标签
    )
    
    # 用于存储医疗报告文本和图片数据
    medical_report = ""
    uploaded_image_bytes = None
    
    # ---------- 上传文件模式 ----------
    if input_method == "上传病例报告":
        # 显示支持的格式提示
        st.caption("📎 文本类文件（TXT、PDF、Markdown）、图像格式文件（PNG、JPG）、最多10个文件")
        # 文件上传组件（支持多文件）
        uploaded_files = st.file_uploader(
            "上传医疗报告文件",  # 必须提供非空标签
            type=["txt", "pdf", "md", "markdown", "png", "jpg", "jpeg"], 
            accept_multiple_files=True,  # 启用多文件上传
            on_change=clear_results,  # 文件变化时清空结果
            label_visibility="collapsed"  # 隐藏标签但保持可访问性
        )
        # 处理上传的文件
        if uploaded_files:
            # 检查文件数量限制
            if len(uploaded_files) > 10:
                st.error("⚠️ 最多支持上传 10 个文件，请减少文件数量")
            else:
                # 合并所有文件内容
                all_texts = []
                all_images = []
                for uploaded_file in uploaded_files:
                    text, image_bytes = process_uploaded_file(uploaded_file)
                    if text:
                        # 添加文件名标识
                        all_texts.append(f"【文件：{uploaded_file.name}】\n{text}")
                    if image_bytes:
                        all_images.append(image_bytes)
                # 合并文本内容
                if all_texts:
                    separator = "\n\n" + "="*50 + "\n\n"
                    medical_report = separator.join(all_texts)
                # 保存第一张图片到 session_state（如果有多张图片，优先处理第一张）
                if all_images:
                    st.session_state.uploaded_image = all_images[0]
                    if len(all_images) > 1:
                        st.info(f"📷 检测到 {len(all_images)} 张图片，将优先分析第一张")
                
    # ---------- 选择示例报告模式 ----------
    elif input_method == "示例病例报告":
        # 示例文件目录
        example_dir = os.path.join("data", "medical_reports", "Examples")
        if os.path.exists(example_dir):
            # 获取所有 .txt 文件
            example_files = [f for f in os.listdir(example_dir) if f.endswith(".txt")]
            if example_files:
                # 文件名到中文名的映射（美化显示）
                file_display_names = {
                    "example_01_diarrhea.txt": "腹泻",
                    "example_02_asthma.txt": "哮喘",
                    "example_03_headache.txt": "头痛"
                }
                # 示例选择下拉框
                selected_example = st.selectbox(
                    "请选择一个示例报告", 
                    example_files, 
                    # 使用中文名显示
                    format_func=lambda x: file_display_names.get(x, x),
                    on_change=clear_results,
                    label_visibility="collapsed"
                )
                # 读取选中的示例文件
                if selected_example:
                    with open(os.path.join(example_dir, selected_example), "r", encoding="utf-8") as f:
                        medical_report = f.read()
                else:
                    st.warning("未找到示例报告文件。")
            else:
                st.warning("示例报告目录不存在。")

    # ==================== 报告内容预览区域 ====================
    # 检查是否有上传的图片
    has_image = st.session_state.get("uploaded_image") is not None
    
    # 使用可折叠的下拉框显示报告内容
    with st.expander("病例报告-内容提取", expanded=False):
        if medical_report:
            # 文本报告：显示在禁用的文本框中
            st.text_area("文件内容预览", value=medical_report, height=300, disabled=True, label_visibility="collapsed")
        elif has_image:
            # 图片报告：显示图片预览
            st.image(st.session_state.uploaded_image, caption="上传的医疗图片", use_container_width=True)
            st.info("💡 图片将通过视觉模型进行分析")
        else:
            # 无内容
            st.info("请在上方选择或上传报告")
    
    # ==================== 开始诊断按钮 ====================
    start_btn = st.button("开始诊断", type="primary", use_container_width=True)
        
    # ==================== 状态显示区域 ====================
    # 创建一个占位符，用于动态更新状态
    status_placeholder = st.empty()

    # 如果已有诊断结果且不在运行中，显示完成状态
    if st.session_state.diagnosis_result:
        with status_placeholder:
            st.success("✅ 多学科会诊已完成")

    # ==================== 诊断过程区域 ====================
    with st.expander("诊断过程-内容记录", expanded=False):
        # 只有在有日志时才创建带边框且固定高度的容器，避免未启动时显示巨大的空白框
        if not st.session_state.specialist_logs:
            st.info("诊断启动后，各专科医生的会诊意见将在此处实时显示")
            process_container = st.container()
        else:
            process_container = st.container(height=400, border=True)
            
        # 重新渲染历史日志（页面刷新后恢复显示）
        for log in st.session_state.specialist_logs:
            with process_container:
                # 使用聊天消息样式显示每个专科的意见
                with st.chat_message(log["agent"], avatar="👨‍⚕️"):
                    st.write(f"**{log['agent']}**: {log['content']}")

        # ========== 异步诊断任务定义 ==========
        async def run_async_diagnosis():
            """
            执行异步诊断流程
            
            这是诊断的核心异步函数，负责：
            1. 显示诊断进度状态
            2. 调用 generate_diagnosis 执行多学科会诊
            3. 实时更新各专科医生的诊断意见
            4. 显示最终诊断结果
            5. 保存到数据库
            """
            # 使用 status_placeholder 显示整体进度
            with status_placeholder:
                # Streamlit 的 status 组件，支持展开/折叠和状态更新
                with st.status("🚀 正在启动多学科会诊系统...", expanded=True) as status_container:
                    # 调用诊断流程编排器
                    # generate_diagnosis 是一个异步生成器
                    gen = generate_diagnosis(medical_report)
                    full_diagnosis = None
                    
                    try:
                        # 遍历异步生成器的输出
                        async for agent_name, response in gen:
                            if agent_name == "Status":
                                # 状态更新：更新进度显示
                                status_container.update(label=response, state="running")
                            elif agent_name == "Final Diagnosis":
                                # 最终诊断：保存结果
                                full_diagnosis = response
                                st.session_state.diagnosis_result = full_diagnosis
                                # 保存到数据库
                                db.save_consultation(medical_report, full_diagnosis)
                                # 更新状态为完成
                                status_container.update(label="✅ 会诊完成", state="complete", expanded=False)
                            else:
                                # 专家意见：添加到日志
                                st.session_state.specialist_logs.append({
                                    "agent": agent_name,
                                    "content": response
                                })
                                # 实时显示在诊断过程容器中
                                with process_container:
                                    with st.chat_message(agent_name, avatar="👨‍⚕️"):
                                        st.write(f"**{agent_name}**: {response}")
                    except Exception as e:
                        # 诊断过程中发生错误
                        st.error(f"诊断过程中发生错误: {e}")
                        status_container.update(label="❌ 诊断失败", state="error")

            if full_diagnosis:
                # 诊断完成后，刷新页面以进入持久化显示模式
                st.rerun()
    
    # ==================== 输入验证 ====================
    # 检查是否有有效的输入（文本或图片）
    has_valid_input = medical_report or has_image
    
    # 点击按钮但没有输入
    if start_btn and not has_valid_input:
        st.error("请先上传或选择一份医疗报告/图片！")

    # ==================== 执行诊断 ====================
    if start_btn and has_valid_input:
        # 检查 API Key 是否配置
        if not (os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
            st.error("未检测到有效的 API Key，请先配置环境变量！")
        else:
            # 如果是图片且没有文本，先用视觉模型提取描述
            if has_image and not medical_report:
                with st.spinner("🔍 正在分析医疗图片..."):
                    # 导入视觉分析函数
                    from src.services.llm import analyze_medical_image
                    # 分析图片，获取医学描述
                    medical_report = analyze_medical_image(st.session_state.uploaded_image)
                    if not medical_report:
                        st.error("图片分析失败，请重试或上传文本格式的报告")
                    else:
                        st.success("✅ 图片分析完成")
            
            # 运行异步诊断任务
            if medical_report:
                # asyncio.run 在同步环境中运行异步函数
                asyncio.run(run_async_diagnosis())

    # ==================== 诊断结果显示区域 ====================
    # 只有在有诊断结果且不是刚点击按钮时显示
    if st.session_state.diagnosis_result and not start_btn:
        st.markdown('<div class="sub-header">📋 输出诊断结果</div>', unsafe_allow_html=True)
        
        # 使用可折叠的下拉框显示结果
        with st.expander("诊断结果-内容提取", expanded=True):
            st.markdown(f"{st.session_state.diagnosis_result}")
        
        # 导入报告生成函数
        from src.tools.export import generate_markdown
        
        # 重新构建报告内容用于下载
        report_content = f"# 医疗诊断报告\n\n## 病例报告\n{medical_report}\n\n## 诊断结果\n{st.session_state.diagnosis_result}"

        # 生成 Markdown 文件
        md_file = generate_markdown(report_content)
        
        # 下载按钮
        st.download_button(
            label="📝 下载 Markdown 文件报告",
            data=md_file,
            file_name="diagnosis_report.md",
            mime="text/markdown",
            use_container_width=True,
            key="download_md_btn_persistent"
        )

    # ==================== 悬浮聊天助手 ====================
    # 使用 Streamlit 的 popover 组件创建悬浮聊天窗口
    with st.popover(" ", help="咨询专家助手"):
        # 导入聊天组件
        from src.ui.chat_component import render_chat_component
        
        # 获取 API Key 和配置
        # 默认使用 Qwen (DashScope)
        api_key = os.getenv("DASHSCOPE_API_KEY")
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        model = os.getenv("QWEN_MODEL", "qwen-max")
        
        # 检查用户是否选择了 OpenAI
        provider = st.session_state.get("llm_provider", "qwen")
        
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            # 使用 OpenAI 配置
            api_key = os.getenv("OPENAI_API_KEY")
            base_url = "https://api.openai.com/v1/chat/completions"
            model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")

        # 构建系统提示词
        system_prompt = "你是一个专业的医疗 AI 助手。请根据用户的提问进行解答。"
        # 如果有诊断结果，将其加入系统提示
        if st.session_state.diagnosis_result:
            system_prompt += f"\n\n以下是该患者的多学科综合诊断结果，请以此为依据回答用户问题：\n{st.session_state.diagnosis_result}"
        
        # 渲染聊天组件
        if api_key:
            render_chat_component(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=system_prompt
            )
        else:
            st.error("未配置 API Key，无法启动聊天助手。")


# ==================== 应用入口 ====================
if __name__ == "__main__":
    # 运行主函数
    main()
