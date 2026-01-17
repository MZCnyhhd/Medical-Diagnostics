"""
模块名称: Medical Diagnostics App (医疗诊断应用入口)
功能描述:
    本文件是整个医疗诊断系统的前端入口，基于 Streamlit 框架构建。
    集成用户认证、病例输入、智能诊断流程控制、结果展示及历史记录管理。
    提供悬浮聊天助手，支持针对诊断结果的追问。
设计理念:
    1.  **单页应用 (SPA)**: 利用 Streamlit 的响应式布局，在一个页面内完成所有交互。
    2.  **状态驱动 UI**: 广泛使用 `st.session_state` 管理用户登录、诊断进度、聊天记录等状态。
    3.  **异步集成**: 在同步的 Streamlit 渲染循环中嵌入 `asyncio` 事件循环，以驱动后端的并发诊断逻辑。
    4.  **角色权限**: 基于角色的访问控制 (RBAC)，区分 Admin/Doctor/Nurse 的操作权限。
页面布局:
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
线程安全性:
    - Streamlit 为每个用户会话运行独立的脚本线程。
    - 全局资源 (如 DB 连接) 需保证线程安全 (已在 Services 层处理)。
依赖关系:
    - `streamlit`: Web 框架。
    - `streamlit-authenticator`: 认证组件。
    - `src.core.orchestrator`: 诊断业务入口。
    - `src.ui`: 各类 UI 组件。
"""
# [导入模块] ############################################################################################################
# [标准库 | Standard Libraries] =========================================================================================
import os                                                              # 操作系统接口：文件和环境变量操作
from typing import Any, cast                                           # 类型注解：通用类型
# [第三方库 | Third-party Libraries] ====================================================================================
import streamlit as st                                                  # Web 应用框架
# [Streamlit 页面配置] ===================================================================================================
# 注意：set_page_config 必须是第一个 Streamlit 命令
st.set_page_config(
    page_title="医疗诊断 AI 智能体",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# [关键修复] 必须在导入任何 src.* 模块之前加载环境变量
# 否则 settings.py 在导入时初始化的单例将无法获取 Key
from dotenv import load_dotenv
try:
    # 硬编码路径以避免循环依赖，与 settings.py 默认值保持一致
    env_path = "config/apikey.env"
    # 强制 UTF-8 加载
    load_dotenv(dotenv_path=env_path, override=True, encoding="utf-8")
    
    # 二次检查关键 Key 是否加载成功
    if not os.getenv("DASHSCOPE_API_KEY"):
         # 如果 UTF-8 加载后 Key 仍为空，可能是文件 BOM 头问题，尝试 GBK
        load_dotenv(dotenv_path=env_path, override=True, encoding="gbk")
        
except Exception as e:
    st.error(f"环境变量加载异常: {e}")

# [内部模块 | Internal Modules] =========================================================================================
try:
    import asyncio                                                         # 异步编程
    from src.core.orchestrator import generate_diagnosis                   # 诊断编排器
    from src.core.settings import get_settings, APIKEY_ENV_PATH            # 系统配置
    from src.services.cache import get_cache                               # 缓存服务
    import src.services.db as db                                           # 数据库服务
    from src.services.auth import (
        get_authenticator,                                                 # 核心认证器对象
        render_login_page,                                                 # 登录页渲染函数
        render_user_info_sidebar,                                          # 侧边栏用户信息展示
        get_user_role,                                                     # 获取用户角色 (admin/doctor/nurse)
        render_user_management                                             # 用户管理界面 (仅管理员)
    )
    from src.ui.styles import get_css                                      # UI 样式
    from src.ui.sidebar import render_sidebar                              # 侧边栏组件
    from src.services.logging import log_info, log_error                   # 日志服务
    from src.utils.file_processors import (                                # 文件处理工具
        process_uploaded_file as process_file_core,
        FileProcessingError,
        UnsupportedFileFormatError
    )
    # [初始化全局配置]
    # settings 已经在导入时自动加载了最新的环境变量
    settings = get_settings()
except Exception as e:
    # 这里不能用 log_error，因为可能 logging 还没初始化成功，或者是在最外层捕获
    st.error(f"应用启动失败，发生严重错误：\n{e}")
    st.stop()
# [注入自定义样式] =======================================================================================================
st.markdown(get_css(), unsafe_allow_html=True)
# [定义函数] ############################################################################################################
# [外部-UI清空结果] ======================================================================================================
def clear_results() -> None:
    """
    清空诊断结果和日志。
    当用户切换输入方式或上传新文件时调用。
    """
    # [step1] 清空诊断结果
    st.session_state.diagnosis_result = None
    # [step2] 清空专科医生日志
    st.session_state.specialist_logs = []
    # [step3] 清空聊天消息
    st.session_state.messages = []
    # [step4] 清空上传的图片
    if "uploaded_image" in st.session_state:
        st.session_state.uploaded_image = None
# [外部-处理上传文件] =====================================================================================================
def process_uploaded_file(uploaded_file) -> tuple[str, bytes | None]:
    """
    处理上传的文件，返回文本内容或图片数据。
    :param uploaded_file: Streamlit UploadedFile 对象
    :return: (文本内容, 图片字节数据)
    """
    # [step1] 检查文件是否存在
    if uploaded_file is None:
        return "", None
    try:
        # [step2] 调用核心处理逻辑
        return process_file_core(uploaded_file.name, uploaded_file.read())
    except UnsupportedFileFormatError as e:
        st.warning(str(e))
        log_info(f"用户上传了不支持的文件格式: {uploaded_file.name}")
        return "", None
    except FileProcessingError as e:
        st.error(f"文件处理失败: {e}")
        log_error(f"文件处理异常: {e}")
        return "", None
    except Exception as e:
        st.error(f"未知错误: {e}")
        log_error(f"处理文件 {uploaded_file.name} 时发生未知错误: {e}", exc_info=True)
        return "", None
# [外部-初始化会话状态] ====================================================================================================
def init_session_state() -> None:
    """初始化会话状态"""
    # [step1] 初始化数据库
    db.init_db()
    # [step2] 设置默认会话状态
    defaults = {
        "messages": [],
        "diagnosis_result": None,
        "specialist_logs": [],
        "active_page": "main"
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    return None
# [内部-获取页面参数] =====================================================================================================
def _get_page_param() -> str:
    # [step1] 获取 query 参数中的 page
    try:
        return st.query_params.get("page", "main")
    except (AttributeError, KeyError):
        return st.experimental_get_query_params().get("page", ["main"])[0]
# [内部-设置页面参数] =====================================================================================================
def _set_page_param(page: str) -> None:
    # [step1] 写入 query 参数中的 page
    try:
        st.query_params["page"] = page
    except (AttributeError, KeyError):
        st.experimental_set_query_params(page=page)
    return None
# [内部-显示医生消息] =====================================================================================================
def _render_doctor_message(target: Any, agent: str, content: str) -> None:
    # [step1] 输出专科医生的对话气泡
    msg = target.chat_message(agent, avatar="👨‍⚕️")
    msg.write(f"**{agent}**: {content}")
    return None
# [内部-追加专家日志] =====================================================================================================
def _append_specialist_log(agent: str, content: str) -> None:
    # [step1] 追加专科医生日志到会话状态
    st.session_state.specialist_logs.append({"agent": agent, "content": content})
    return None
# [内部-最终诊断] ========================================================================================================
def _finalize_diagnosis(report: str, diagnosis: str, status_container: Any) -> str:
    # [step1] 写入最终诊断到会话状态
    st.session_state.diagnosis_result = diagnosis
    # [step2] 持久化诊断结果到历史记录
    db.save_consultation(report, diagnosis)
    # [step3] 更新状态为完成
    status_container.update(label="✅ 会诊完成", state="complete", expanded=False)
    return diagnosis
# [外部-处理导航] ========================================================================================================
def handle_navigation(username: str) -> bool:
    """处理页面导航"""
    # [step1] 获取当前页面参数
    page = _get_page_param()
    # [step2] 更新活动页面状态
    st.session_state.active_page = page
    # [step3] 处理用户管理页面的特殊导航逻辑
    if st.session_state.active_page != "user_management":
        return False
    is_admin = get_user_role(username) == "admin"
    if not is_admin:
        st.session_state.active_page = "main"
        _set_page_param("main")
        st.rerun()
    go_back = st.button("← 返回主页")
    if go_back:
        st.session_state.active_page = "main"
        _set_page_param("main")
        st.rerun()
    render_user_management()
    return True
# [外部-渲染历史记录部分] ==================================================================================================
def render_history_section() -> None:
    """渲染历史记录区域"""
    st.markdown('<div class="sub-header">📜 历史诊断记录</div>', unsafe_allow_html=True)
    # [step1] 获取历史记录
    history = db.get_history()
    if not history:
        st.info("暂无历史记录")
        return None
    # [step2] 显示选择框
    selected = st.selectbox(
        "查看过往病例", history,
        format_func=lambda x: f"🕒 {x['timestamp']} (ID: {x['id']})",
        label_visibility="collapsed"
    )
    # [step3] 显示选中记录的详细信息
    if selected:
        with st.expander("诊断记录-内容提取", expanded=False):
            st.markdown("### 📄 病例报告")
            st.markdown(f"> {selected['report_content']}")
            st.markdown("### 🩺 诊断结果")
            st.markdown(selected['diagnosis_result'])
    return None
# [内部-处理文件上传] =====================================================================================================
def _handle_file_upload() -> str:
    # [step1] 渲染文件上传组件
    st.caption("📎 文本类文件（TXT、PDF、Markdown）、图像格式文件（PNG、JPG）")
    f = st.file_uploader(
        "上传医疗报告文件",
        type=["txt", "pdf", "md", "markdown", "png", "jpg", "jpeg"],
        accept_multiple_files=False,
        on_change=clear_results,
        label_visibility="collapsed"
    )
    # [step2] 处理上传的文件内容
    if f:
        text, img = process_uploaded_file(f)
        if img: st.session_state.uploaded_image = img
        return text if text else ""
    return ""
# [内部-处理示例选择] =====================================================================================================
def _handle_example_selection() -> str:
    # [step1] 检查示例目录是否存在
    example_dir = os.path.join("data", "medical_reports", "Examples")
    if not os.path.exists(example_dir):
        st.warning("示例报告目录不存在。")
        return ""
    # [step2] 获取示例文件列表
    files = [f for f in os.listdir(example_dir) if f.endswith(".txt")]
    if not files:
        st.warning("示例报告目录不存在。")
        return ""
    names = {"example_01_diarrhea.txt": "腹泻", "example_02_asthma.txt": "哮喘", "example_03_headache.txt": "头痛"}
    # [step3] 渲染选择框并读取选中文件
    selected = st.selectbox(
        "请选择一个示例报告",
        files,
        format_func=lambda x: names.get(x, x),
        on_change=clear_results,
        label_visibility="collapsed"
    )
    if selected:
        with open(os.path.join(example_dir, selected), "r", encoding="utf-8") as f:
            return f.read()
    return ""
# [外部-末端输入部分] =====================================================================================================
def render_input_section() -> str:
    """渲染病例输入区域"""
    st.markdown('<div class="sub-header">📄 输入病例报告</div>', unsafe_allow_html=True)
    # [step1] 渲染输入方式选择
    method = st.radio(
        "选择输入方式",
        ["上传病例报告",
         "示例病例报告"],
        on_change=clear_results,
        horizontal=True,
        label_visibility="collapsed"
    )
    # [step2] 根据选择调用相应的处理函数
    if method == "上传病例报告":
        return _handle_file_upload()
    else:
        return _handle_example_selection()
# [外部-渲染预览部分] =====================================================================================================
def render_preview_section(report: str) -> None:
    """渲染报告预览区域"""
    has_img = st.session_state.get("uploaded_image") is not None
    with st.expander("病例报告-内容提取", expanded=False):
        # [step1] 优先显示文本报告
        if report:
            st.text_area("文件内容预览", value=report, height=300, disabled=True, label_visibility="collapsed")
        # [step2] 其次显示上传的图片
        elif has_img:
            st.image(st.session_state.uploaded_image, caption="上传的医疗图片", use_container_width=True)
            st.info("💡 图片将通过视觉模型进行分析")
        # [step3] 提示用户未上传
        else:
            st.info("请在上方选择或上传报告")
    return None
# [外部-渲染日志部分] =====================================================================================================
def render_logs_section() -> object:
    """渲染诊断过程区域"""
    with st.expander("诊断过程-内容记录", expanded=False):
        # [step1] 初始化日志容器
        if not st.session_state.specialist_logs:
            st.info("诊断启动后，各专科医生的会诊意见将在此处实时显示")
            container = st.container()
        else:
            container = st.container(height=400, border=True)
        # [step2] 渲染已有日志
        for log in st.session_state.specialist_logs:
            _render_doctor_message(container, log["agent"], log["content"])
        return container
# [内部-处理诊断流] =========================================================================================================
async def _process_diagnosis_stream(gen, report: str, status_container: Any, log_container: Any) -> str | None:
    """处理诊断生成器流"""
    full_diagnosis = None
    # [step1] 遍历生成器结果
    async for agent, response in gen:
        if agent == "Status":
            # [step2] 更新状态
            status_container.update(label=response, state="running")
        elif agent == "Final Diagnosis":
            # [step3] 处理最终诊断
            full_diagnosis = _finalize_diagnosis(report, response, status_container)
        else:
            # [step4] 记录专家日志
            _append_specialist_log(agent, response)
            _render_doctor_message(log_container, agent, response)
    return full_diagnosis
# [异步-外部-运行诊断流程] =================================================================================================
async def run_diagnosis_flow(report: str, status_container: Any, log_container: Any) -> str | None:
    """执行诊断流程"""
    log_info("开始新的诊断流程")
    # [step1] 初始化诊断生成器
    gen = generate_diagnosis(report)
    try:
        # [step2] 处理诊断流
        result = await _process_diagnosis_stream(gen, report, status_container, log_container)
        log_info("诊断流程完成")
        return result
    except Exception as ex:
        # [step3] 错误处理
        st.error(f"诊断过程中发生错误: {ex}")
        status_container.update(label="❌ 诊断失败", state="error")
        log_error(f"诊断流程异常: {ex}", exc_info=True)
        return None
# [内部-检查API密钥] ======================================================================================================
def _check_api_keys() -> bool:
    """检查必要的 API Key 是否存在"""
    # [step1] 校验环境变量
    # 只要存在任意一个配置的 API Key 即可（支持多模型切换）
    available_keys = [k for k in settings.required_api_keys if os.getenv(k)]
    if not available_keys:
        st.error(f"未检测到有效的 API Key，请至少配置以下之一: {', '.join(settings.required_api_keys)}")
        log_error("未检测到任何有效的 API Key")
        return False
    return True
# [内部-分析上传图片] =====================================================================================================
def _analyze_uploaded_image(image_bytes: bytes) -> str:
    """调用视觉模型分析图片"""
    # [step1] 执行分析
    with cast(Any, st.spinner("🔍 正在分析医疗图片...")):
        from src.services.llm import analyze_medical_image
        report = analyze_medical_image(image_bytes)
    # [step2] 检查结果
    if not report:
        st.error("图片分析失败，请重试或上传文本格式的报告")
        return ""
    # [step3] 返回报告
    st.success("✅ 图片分析完成")
    return report
# [外部-执行诊断] ========================================================================================================
def execute_diagnosis(report: str, status_ph: Any, logs_container: Any) -> None:
    """处理诊断执行逻辑"""
    has_img = st.session_state.get("uploaded_image") is not None
    valid_input = report or has_img
    # [step1] 校验输入
    if not valid_input:
        st.error("请先上传或选择一份医疗报告/图片！")
        return None
    # [step2] 校验 API Key
    if not _check_api_keys():
        return None
    # [step3] 处理纯图片输入
    if has_img and not report:
        report = _analyze_uploaded_image(st.session_state.uploaded_image)
        if not report:
            return None
    # [step4] 启动诊断流程
    if report:
        status_container = status_ph.status("🚀 正在启动多学科会诊系统...", expanded=True)
        done = asyncio.run(run_diagnosis_flow(report, status_container, logs_container))
        if done:
            st.rerun()
    return None
# [外部-渲染结果区域] =====================================================================================================
def render_results_section(report: str) -> None:
    """渲染诊断结果和导出"""
    st.markdown('<div class="sub-header">📋 输出诊断结果</div>', unsafe_allow_html=True)
    # [step1] 展示诊断结果
    with st.expander("诊断结果-内容提取", expanded=True):
        st.markdown(f"{st.session_state.diagnosis_result}")
    # [step2] 提供下载按钮
    from src.tools.export import generate_markdown
    content = f"# 医疗诊断报告\n\n## 病例报告\n{report}\n\n## 诊断结果\n{st.session_state.diagnosis_result}"
    st.download_button(
        label="📝 下载 Markdown 文件报告",
        data=generate_markdown(content),
        file_name="diagnosis_report.md",
        mime="text/markdown",
        use_container_width=True,
        key="download_md_btn_persistent"
    )
    return None
# [内部-获取聊天配置] =====================================================================================================
def _get_chat_config() -> dict[str, str | None]:
    """获取 LLM 聊天配置"""
    provider = st.session_state.get("llm_provider", "qwen")
    # [step1] 默认配置
    cfg = {
        "api_key": os.getenv("DASHSCOPE_API_KEY"),
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
        "model": os.getenv("QWEN_MODEL", "qwen-max")
    }
    # [step2] OpenAI 配置覆盖
    if provider == "openai" and os.getenv("OPENAI_API_KEY"):
        cfg.update({
            "api_key": os.getenv("OPENAI_API_KEY"),
            "base_url": "https://api.openai.com/v1/chat/completions",
            "model": os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
        })
    return cfg
# [内部-构建系统提示词] ====================================================================================================
def _build_chat_system_prompt(diagnosis_result: str | None) -> str:
    """构建聊天助手 System Prompt"""
    # [step1] 基础提示词
    sys_prompt = "你是一个专业的医疗 AI 助手。请根据用户的提问进行解答。"
    # [step2] 注入诊断上下文
    if diagnosis_result:
        sys_prompt += (
            "\n\n以下是该患者的多学科综合诊断结果，请以此为依据回答用户问题：\n"
            f"{diagnosis_result}"
        )
    return sys_prompt
# [内部-渲染聊天内容] =====================================================================================================
def _render_chat_content() -> None:
    """渲染聊天组件内容"""
    from src.ui.chat_component import render_chat_component
    # [step1] 获取配置
    cfg = _get_chat_config()
    
    # [step2] 渲染组件或显示错误
    if cfg["api_key"]:
        sys_prompt = _build_chat_system_prompt(st.session_state.diagnosis_result)
        render_chat_component(
            api_key=cfg["api_key"], 
            base_url=cfg["base_url"], 
            model=cfg["model"], 
            system_prompt=sys_prompt
        )
    else:
        st.error("未配置 API Key，无法启动聊天助手。")
    return None
# [外部-渲染聊天助手] =====================================================================================================
def render_chat_assistant() -> None:
    """渲染悬浮聊天助手"""
    # [step1] 渲染弹出框
    with st.popover(" ", help="咨询专家助手"):
        _render_chat_content()
    return None
# [主函数] ==============================================================================================================
def main() -> None:
    """应用主函数"""
    # [step1] 用户认证
    username, auth_status, _ = render_login_page()
    if not auth_status:
        return None
    # [step2] 初始化应用状态
    init_session_state()
    # [step3] 渲染侧边栏
    render_sidebar()
    render_user_info_sidebar(get_authenticator(), username)
    # [step4] 处理页面导航
    if handle_navigation(username):
        return None
    # [step5] 渲染主要内容区域
    render_history_section()
    report = render_input_section()
    render_preview_section(report)
    # [step6] 渲染操作按钮和状态区
    start_btn = st.button("开始诊断", type="primary", use_container_width=True)
    status_ph = st.empty()
    if st.session_state.diagnosis_result:
        status_ph.success("✅ 多学科会诊已完成")
    # [step7] 渲染日志区并处理诊断逻辑
    logs_container = render_logs_section()
    if start_btn:
        execute_diagnosis(report, status_ph, logs_container)
    # [step8] 渲染结果和聊天助手
    if st.session_state.diagnosis_result and not start_btn:
        render_results_section(report)
    render_chat_assistant()
    return None
# [应用入口] ############################################################################################################
if __name__ == "__main__":
    main()