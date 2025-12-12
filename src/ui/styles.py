
def get_css():
    return """
<style>
    /* 引入 Google Fonts (可选，如果网络允许) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* 全局样式 */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1e293b; /* Slate-800 */
    }

    /* 优化原生标题样式 */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #1e293b;
        letter-spacing: -0.01em;
    }
    
    h1 { font-size: 2.25rem; }
    h2 { font-size: 1.8rem; }
    h3 { font-size: 1.5rem; }

    /* 主背景 */
    .stApp {
        background-color: #f8fafc; /* Slate-50 */
        background-image: radial-gradient(#e2e8f0 1px, transparent 1px);
        background-size: 20px 20px;
    }
    
    /* 限制主内容区域宽度 (变窄 1/3) 并增加底部留白 */
    .block-container {
        max-width: 66% !important;
        margin-left: auto !important;
        margin-right: auto !important;
        padding-top: 2rem !important;
        padding-bottom: 150px !important; /* 增加底部留白，防止被悬浮元素遮挡 */
    }

    /* 侧边栏样式 */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0; /* Slate-200 */
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.02);
    }
    
    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* 标题样式 */
    .main-header {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(120deg, #2563eb, #0891b2); /* Blue-600 to Cyan-600 */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2.5rem;
        letter-spacing: -0.025em;
        padding: 1rem 0;
    }
    
    .sub-header {
        font-size: 1.0rem;
        font-weight: 600;
        color: #334155; /* Slate-700 */
        margin-top: 1.0rem;
        margin-bottom: 0.75rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* 装饰性下划线 */
    .sub-header::after {
        content: "";
        flex: 1;
        height: 1px;
        background: #e2e8f0;
        margin-left: 1rem;
    }

    /* 输入框和下拉框优化 */
    .stTextInput input, .stSelectbox select, .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #cbd5e1; /* Slate-300 */
        background-color: #ffffff;
        padding: 0.75rem;
        transition: all 0.2s;
        font-size: 0.95rem;
    }

    .stTextInput input:focus, .stSelectbox select:focus, .stTextArea textarea:focus {
        border-color: #3b82f6; /* Blue-500 */
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        outline: none;
    }

    /* 按钮样式优化 */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: none;
        padding: 0.6rem 1.2rem;
        transition: all 0.2s ease;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* 主要按钮 (Primary) */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.3);
    }
    
    /* 次要按钮 (Secondary/Default) */
    .stButton > button[kind="secondary"] {
        background-color: #ffffff;
        color: #475569;
        border: 1px solid #e2e8f0;
    }

    .stButton > button[kind="secondary"]:hover {
        background-color: #f8fafc;
        border-color: #cbd5e1;
        color: #1e293b;
    }

    /* Expander 样式 */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        color: #334155;
        font-weight: 600;
        padding: 1rem;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-top: none;
        border-bottom-left-radius: 8px;
        border-bottom-right-radius: 8px;
        padding: 1.5rem;
    }
    
    /* File Uploader 区域 */
    .stFileUploader {
        background-color: transparent;
        border: none;
        border-radius: 12px;
        padding: 0;
        text-align: center;
        transition: border-color 0.2s;
    }
    
    .stFileUploader:hover {
        border-color: #3b82f6;
    }

    /* 聊天消息气泡 */
    .stChatMessage {
        background-color: #ffffff;
        border: 1px solid #f1f5f9;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        margin-bottom: 1rem;
        animation: fadeIn 0.3s ease-in-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(5px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stChatMessage[data-testid="stChatMessageAvatar"] {
        background-color: #eff6ff;
        border: 1px solid #dbeafe;
    }

    /* 状态容器 */
    .stStatus {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }

    /* 浮动聊天按钮优化 */
    div[data-testid="stPopover"] {
        position: fixed !important;
        bottom: 2rem !important;
        right: 2rem !important;
        z-index: 9999 !important;
    }

    div[data-testid="stPopover"] button {
        width: 3.5rem !important;
        height: 3.5rem !important;
        border-radius: 50% !important;
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important; /* Sky Blue */
        border: none !important;
        box-shadow: 0 10px 15px -3px rgba(14, 165, 233, 0.4), 0 4px 6px -2px rgba(14, 165, 233, 0.2) !important;
        transition: transform 0.2s, box-shadow 0.2s !important;
    }

    div[data-testid="stPopover"] button:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 20px 25px -5px rgba(14, 165, 233, 0.5), 0 10px 10px -5px rgba(14, 165, 233, 0.2) !important;
    }

    /* 浮动按钮图标替换为 Chat 图标 */
    div[data-testid="stPopover"] button::after {
        content: "";
        display: block;
        width: 24px;
        height: 24px;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z'%3E%3C/path%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: center;
    }
    
    /* 隐藏原始文本 */
    div[data-testid="stPopover"] button > div {
        display: none !important;
    }

    /* 自定义滚动条 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #cbd5e1;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #94a3b8;
    }
    
    /* Dividers */
    hr {
        border-color: #e2e8f0;
        opacity: 0.6;
    }

    /* Info/Success/Warning/Error Messages */
    .stAlert {
        border-radius: 8px;
        border: none;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    
    /* --- Popover 聊天窗口优化 --- */
    
    /* Popover 内容容器 - 增加宽度，去除内边距 */
    /* Popover 内容容器 - 强制固定在右下角 */
    div[data-testid="stPopoverBody"] {
        position: fixed !important;
        top: auto !important;
        left: auto !important;
        bottom: 90px !important; /* 按钮位置 + 高度 */
        right: 2rem !important;
        width: 450px !important;
        max-width: 90vw !important;
        max-height: 70vh !important;
        padding: 0 !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        border: 1px solid #e2e8f0 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1) !important;
        z-index: 10000 !important;
        transform: none !important;
    }
    
    /* 确保 iframe 填满容器 */
    div[data-testid="stPopoverBody"] > div {
        height: 100% !important;
        min-height: 500px !important;
    }
    
    div[data-testid="stPopoverBody"] iframe {
        width: 100% !important;
        min-height: 500px !important;
        display: block !important;
        margin: 0 !important;
    }

    /* --- 文件上传组件美化与汉化 --- */
    
    /* 1. 拖拽区域样式优化 - 扩展至填满整个上传区域 */
    section[data-testid="stFileUploaderDropzone"] {
        background-color: #ffffff;
        border: 2px dashed #3b82f6;
        border-radius: 12px;
        padding: 2rem;
        transition: all 0.3s ease;
        position: relative; /* 为按钮绝对定位做准备 */
        width: 100%;
        box-sizing: border-box;
    }
    
    section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: #2563eb;
        background-color: #eff6ff;
    }
    
    /* 2. 隐藏原有的 "Drag and drop file here" 文本 */
    section[data-testid="stFileUploaderDropzone"] > div > div > span {
        display: none;
    }
    
    /* 3. 添加自定义中文提示 */
    section[data-testid="stFileUploaderDropzone"] > div > div::after {
        content: "拖拽文件到此处，或点击浏览文件";
        display: block;
        font-size: 1rem;
        font-weight: 500;
        color: #475569;
        margin-top: 8px;
        text-align: center;
    }
    
    /* 4. 隐藏 "Limit 200MB per file" 文本 (通常是 small 标签) */
    section[data-testid="stFileUploaderDropzone"] small {
        display: none;
    }
    
    /* 5. 添加自定义大小限制提示 */
    section[data-testid="stFileUploaderDropzone"] > div > div::before {
        content: "支持单个文件最大 200MB";
        display: block;
        font-size: 0.85rem;
        color: #94a3b8;
        order: 2; /* 尝试放在下面 */
        margin-top: 4px;
        text-align: center;
    }
    
    /* 6. 将 "Browse files" 按钮改为全屏透明覆盖，实现“点击任意处上传”，视觉上删除按钮 */
    section[data-testid="stFileUploaderDropzone"] button {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: 0;
        z-index: 10;
        cursor: pointer;
    }

    /* 辅助样式：确保内容居中 */
    section[data-testid="stFileUploaderDropzone"] > div {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
    }

    section[data-testid="stFileUploaderDropzone"] > div > div {
        display: flex;
        flex-direction: column;
        align-items: center;
        width: 100%;
    }

</style>
"""

