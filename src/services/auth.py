"""
用户认证服务模块
================

本模块提供医疗诊断系统的用户认证功能，包括：
1. 用户登录/登出
2. 用户注册（仅管理员）
3. 密码重置
4. 用户角色管理

角色说明：
- admin：系统管理员，拥有所有权限
- doctor：医生，可以进行诊断和查看历史
- nurse：护士，可以上传报告和查看历史

技术实现：
- 使用 streamlit-authenticator 库
- 密码使用 bcrypt 哈希存储
- 用户数据存储在 config/auth.yaml 文件中
"""

import os
import yaml
import bcrypt
import streamlit as st
import streamlit_authenticator as stauth
from typing import Optional, Tuple, Dict, Any

# 配置文件路径
AUTH_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "auth.yaml")


def load_auth_config() -> Dict[str, Any]:
    """
    加载认证配置文件
    
    Returns:
        Dict: 认证配置字典，包含用户信息和 cookie 设置
    """
    config_path = os.path.abspath(AUTH_CONFIG_PATH)
    
    if not os.path.exists(config_path):
        # 如果配置文件不存在，创建默认配置
        default_config = create_default_config()
        save_auth_config(default_config)
        return default_config
    
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_auth_config(config: Dict[str, Any]) -> None:
    """
    保存认证配置到文件
    
    Args:
        config: 认证配置字典
    """
    config_path = os.path.abspath(AUTH_CONFIG_PATH)
    
    # 确保目录存在
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def create_default_config() -> Dict[str, Any]:
    """
    创建默认认证配置
    
    Returns:
        Dict: 默认配置字典
    """
    # 生成默认密码的哈希值
    admin_hash = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
    doctor_hash = bcrypt.hashpw("doctor123".encode(), bcrypt.gensalt()).decode()
    nurse_hash = bcrypt.hashpw("nurse123".encode(), bcrypt.gensalt()).decode()
    
    return {
        "cookie": {
            "expiry_days": 30,
            "key": "medical_diagnosis_system_secret_key_2024",
            "name": "medical_auth_cookie"
        },
        "credentials": {
            "usernames": {
                "admin": {
                    "email": "admin@hospital.com",
                    "failed_login_attempts": 0,
                    "logged_in": False,
                    "name": "系统管理员",
                    "password": admin_hash,
                    "role": "admin"
                },
                "doctor": {
                    "email": "doctor@hospital.com",
                    "failed_login_attempts": 0,
                    "logged_in": False,
                    "name": "张医生",
                    "password": doctor_hash,
                    "role": "doctor"
                },
                "nurse": {
                    "email": "nurse@hospital.com",
                    "failed_login_attempts": 0,
                    "logged_in": False,
                    "name": "李护士",
                    "password": nurse_hash,
                    "role": "nurse"
                }
            }
        },
        "pre-authorized": {
            "emails": ["newuser@hospital.com"]
        }
    }


def hash_password(password: str) -> str:
    """
    对密码进行哈希处理
    
    Args:
        password: 明文密码
    
    Returns:
        str: bcrypt 哈希后的密码
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def get_authenticator() -> stauth.Authenticate:
    """
    获取认证器实例（使用缓存避免重复创建）
    
    Returns:
        stauth.Authenticate: Streamlit 认证器实例
    """
    # 使用 session_state 缓存认证器实例，避免重复创建导致 key 冲突
    if "authenticator" not in st.session_state:
        config = load_auth_config()
        
        # streamlit-authenticator 0.4.x 版本移除了 pre_authorized 参数
        st.session_state.authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days']
        )
    
    return st.session_state.authenticator


def get_user_role(username: str) -> Optional[str]:
    """
    获取用户角色
    
    Args:
        username: 用户名
    
    Returns:
        Optional[str]: 用户角色，如果用户不存在则返回 None
    """
    config = load_auth_config()
    user_data = config.get('credentials', {}).get('usernames', {}).get(username)
    
    if user_data:
        return user_data.get('role', 'user')
    return None


def get_user_display_name(username: str) -> str:
    """
    获取用户显示名称
    
    Args:
        username: 用户名
    
    Returns:
        str: 用户显示名称
    """
    config = load_auth_config()
    user_data = config.get('credentials', {}).get('usernames', {}).get(username)
    
    if user_data:
        return user_data.get('name', username)
    return username


def add_user(username: str, name: str, email: str, password: str, role: str = "nurse") -> bool:
    """
    添加新用户
    
    Args:
        username: 用户名
        name: 显示名称
        email: 邮箱
        password: 明文密码
        role: 角色 (admin/doctor/nurse)
    
    Returns:
        bool: 是否添加成功
    """
    config = load_auth_config()
    
    # 检查用户是否已存在
    if username in config['credentials']['usernames']:
        return False
    
    # 添加新用户
    config['credentials']['usernames'][username] = {
        "email": email,
        "failed_login_attempts": 0,
        "logged_in": False,
        "name": name,
        "password": hash_password(password),
        "role": role
    }
    
    save_auth_config(config)
    return True


def delete_user(username: str) -> bool:
    """
    删除用户
    
    Args:
        username: 用户名
    
    Returns:
        bool: 是否删除成功
    """
    config = load_auth_config()
    
    # 不允许删除 admin 用户
    if username == "admin":
        return False
    
    if username in config['credentials']['usernames']:
        del config['credentials']['usernames'][username]
        save_auth_config(config)
        return True
    
    return False


def update_user_password(username: str, new_password: str) -> bool:
    """
    更新用户密码
    
    Args:
        username: 用户名
        new_password: 新密码（明文）
    
    Returns:
        bool: 是否更新成功
    """
    config = load_auth_config()
    
    if username in config['credentials']['usernames']:
        config['credentials']['usernames'][username]['password'] = hash_password(new_password)
        save_auth_config(config)
        return True
    
    return False


def get_all_users() -> Dict[str, Dict[str, Any]]:
    """
    获取所有用户信息
    
    Returns:
        Dict: 用户信息字典（不包含密码）
    """
    config = load_auth_config()
    users = {}
    
    for username, data in config['credentials']['usernames'].items():
        users[username] = {
            "name": data.get('name', username),
            "email": data.get('email', ''),
            "role": data.get('role', 'user')
        }
    
    return users


def render_login_page() -> Tuple[Optional[str], bool, Optional[str]]:
    """
    渲染登录页面
    
    Returns:
        Tuple[Optional[str], bool, Optional[str]]: (用户名, 认证状态, 显示名称)
    """
    authenticator = get_authenticator()
    
    # 设置登录页面样式
    st.markdown("""
    <style>
    .login-container {
        max-width: 400px;
        margin: 0 auto;
        padding: 2rem;
    }
    .login-header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .login-header h1 {
        color: #1e3a5f;
        font-size: 1.8rem;
        margin-bottom: 0.5rem;
    }
    .login-header p {
        color: #64748b;
        font-size: 0.9rem;
    }

    /* 极简输入框 - 长度对齐并居中 */
    .stTextInput input {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        background-color: #ffffff;
        padding: 0.8rem 1rem;
        font-size: 1rem;
        transition: all 0.2s ease;
        color: #334155;
        width: 100%; /* 强制宽度 100% 占满容器 */
    }
    
    .stTextInput input:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
    }
    
    /* 登录按钮优化 - 变长并居中 */
    div[data-testid="stForm"] .stButton {
        margin: 1rem auto 0 auto !important;
        width: 100% !important;
        display: flex;
        justify-content: center;
    }
    
    /* 使用子选择器 > 确保优先级高于全局样式 */
    div[data-testid="stForm"] .stButton > button {
        width: 100% !important; /* 强制宽度 100% 占满容器 */
        display: block;
        border-radius: 8px;
        background: linear-gradient(90deg, #2563eb 0%, #3b82f6 100%); 
        color: white;
        font-weight: 600;
        font-size: 1.05rem;
        padding: 0.8rem 1rem;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2), 0 2px 4px -1px rgba(37, 99, 235, 0.1);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        letter-spacing: 0.05em;
    }
    
    div[data-testid="stForm"] .stButton > button:hover {
        background: linear-gradient(90deg, #1d4ed8 0%, #2563eb 100%);
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.4), 0 4px 6px -2px rgba(37, 99, 235, 0.2);
        color: white;
        border-color: transparent;
    }
    
    div[data-testid="stForm"] .stButton > button:active {
        transform: translateY(0);
        box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.06);
        background-color: #1e3a8a;
    }
    
    /* 覆盖 Streamlit 默认按钮 focus 状态 */
    div[data-testid="stForm"] .stButton > button:focus {
        color: white;
        border-color: transparent;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.4);
    }
    
    /* 隐藏 Streamlit Authenticator 默认标题 */
    div[data-testid="stForm"] > div:first-child h3 {
        display: none;
    }
    
    /* 确保表单内容居中 */
    div[data-testid="stForm"] {
        text-align: center;
    }
    
    /* 确保表单内部元素宽度一致 */
    div[data-testid="stForm"] > div:not(:first-child) {
        margin-bottom: 1rem;
    }
    
    /* 确保输入框容器宽度一致 */
    div[data-testid="stForm"] .stTextInput, 
    div[data-testid="stForm"] .stPasswordInput {
        margin-left: auto;
        margin-right: auto;
        max-width: 100%;
    }

    /* streamlit-authenticator 版本差异：提交按钮可能渲染为 stFormSubmitButton */
    div[data-testid="stFormSubmitButton"] {
        margin: 1rem auto 0 auto !important;
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        text-align: center !important;
    }

    /* 某些 Streamlit 版本会在 stFormSubmitButton 下额外包一层 div */
    div[data-testid="stFormSubmitButton"] > div {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        text-align: center !important;
    }

    div[data-testid="stFormSubmitButton"] > button {
        width: 100% !important;
        display: block;
    }

    /* Streamlit 新按钮结构：button[data-testid="stBaseButton-secondaryFormSubmit"] */
    div[data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-secondaryFormSubmit"],
    div[data-testid="stFormSubmitButton"] button[kind="secondaryFormSubmit"] {
        float: none !important;
        display: block !important;
        align-self: center !important;
        margin-left: auto !important;
        margin-right: auto !important;
        width: auto !important;
        min-width: 120px;
    }

    button[data-testid="stBaseButton-secondaryFormSubmit"],
    button[kind="secondaryFormSubmit"] {
        float: none !important;
        display: block !important;
        margin-left: auto !important;
        margin-right: auto !important;
        align-self: center !important;
    }

    div[data-testid="stForm"] div:has(> button[kind="secondaryFormSubmit"]),
    div[data-testid="stForm"] div:has(> button[data-testid="stBaseButton-secondaryFormSubmit"]),
    div[data-testid="stForm"] div:has(button[kind="secondaryFormSubmit"]),
    div[data-testid="stForm"] div:has(button[data-testid="stBaseButton-secondaryFormSubmit"]) {
        width: 100% !important;
        display: flex !important;
        justify-content: center !important;
        text-align: center !important;
    }
    
    /* 确保登录按钮居中且宽度100% */
    div[data-testid="stForm"] .stButton {
        margin-left: auto;
        margin-right: auto;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # 先检查是否已经登录（通过 session_state）
    if st.session_state.get("authentication_status") == True:
        return (
            st.session_state.get("username"),
            True,
            st.session_state.get("name")
        )
    
    # 显示登录表单
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="login-header">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🏥</div>
            <h1>智能医疗诊断系统</h1>
        </div>
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; border: 1px solid #bce3eb; color: #315e6b; margin-bottom: 2rem; text-align: left;">
            <div style="text-align: center; font-weight: bold; font-size: 16px; margin-bottom: 8px;">
                智能多学科会诊系统 (MDT) v1.0.0
            </div>
            <div style="font-size: 14px; line-height: 1.5;">
                模拟真实医院的 MDT 流程，由多个 AI 专科医生协同工作，提供全面的诊断建议。
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 使用 streamlit-authenticator 的登录表单
        try:
            # 0.4.x 版本 API
            authenticator.login(location='main', fields={
                'Form name': '用户登录',
                'Username': '用户名',
                'Password': '密码',
                'Login': '登录'
            })
        except TypeError:
            # 兼容旧版本 API
            try:
                authenticator.login('用户登录', 'main')
            except Exception as e:
                st.error(f"登录组件加载失败: {e}")
                return None, False, None
        except Exception as e:
            st.error(f"登录组件加载失败: {e}")
            return None, False, None
        
        # 获取认证状态
        name = st.session_state.get("name")
        authentication_status = st.session_state.get("authentication_status")
        username = st.session_state.get("username")
        
        if authentication_status == False:
            st.error("用户名或密码错误")
        
        return username, authentication_status == True, name
    
    return None, False, None


def render_user_info_sidebar(authenticator: stauth.Authenticate, username: str) -> None:
    """
    在侧边栏渲染用户信息和登出按钮
    
    Args:
        authenticator: 认证器实例
        username: 当前登录的用户名
    """
    role = get_user_role(username)
    name = get_user_display_name(username)
    
    # 角色显示映射
    role_display = {
        "admin": "👑 管理员",
        "doctor": "👨‍⚕️ 医生",
        "nurse": "👩‍⚕️ 护士"
    }
    
    st.sidebar.markdown("---")
    
    # 使用自定义 HTML 卡片样式美化用户信息
    st.sidebar.markdown(f"""
        <div style="
            background-color: white; 
            padding: 1.2rem; 
            border-radius: 10px; 
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
            border: 1px solid #f0f2f6;
            text-align: center;
        ">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">👤</div>
            <div style="font-weight: 600; font-size: 1.1rem; color: #1f2937; margin-bottom: 0.2rem;">{name}</div>
            <div style="
                display: inline-block;
                background-color: #f3f4f6; 
                color: #4b5563; 
                padding: 0.2rem 0.8rem; 
                border-radius: 9999px; 
                font-size: 0.8rem;
                margin-bottom: 1rem;
            ">{role_display.get(role, '用户')}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # 用户管理按钮（仅管理员可见）
    if role == "admin":
        if st.sidebar.button("👥 用户管理", use_container_width=True):
            st.query_params["page"] = "user_management"
            st.rerun()
    
    # 登出按钮 - 兼容不同版本的 API
    try:
        # 0.4.x 版本 API - 设置 key 避免重复
        authenticator.logout(button_name="🚪 退出登录", location="sidebar", key="logout_btn")
    except TypeError:
        # 旧版本 API
        authenticator.logout("🚪 退出登录", "sidebar", key="logout_btn")


def render_user_management() -> None:
    """
    渲染用户管理界面（仅管理员可用）
    """
    st.subheader("👥 用户管理")
    
    # 获取当前用户角色
    current_user = st.session_state.get("username")
    current_role = get_user_role(current_user)
    
    if current_role != "admin":
        st.warning("⚠️ 仅管理员可以管理用户")
        return
    
    # 显示现有用户
    users = get_all_users()
    
    st.markdown("### 现有用户")
    for username, data in users.items():
        role_emoji = {"admin": "👑", "doctor": "👨‍⚕️", "nurse": "👩‍⚕️"}.get(data['role'], "👤")
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            st.text(f"{role_emoji} {data['name']}")
        with col2:
            st.text(f"@{username}")
        with col3:
            st.text(data['role'])
        with col4:
            if username != "admin" and username != current_user:
                if st.button("🗑️", key=f"del_{username}", help="删除用户"):
                    if delete_user(username):
                        st.success(f"已删除用户 {username}")
                        st.rerun()
    
    st.markdown("---")
    
    # 添加新用户
    st.markdown("### 添加新用户")
    
    with st.form("add_user_form"):
        new_username = st.text_input("用户名", placeholder="例如：zhangsan")
        new_name = st.text_input("姓名", placeholder="例如：张三")
        new_email = st.text_input("邮箱", placeholder="例如：zhangsan@hospital.com")
        new_password = st.text_input("密码", type="password", placeholder="至少6位")
        new_role = st.selectbox("角色", ["nurse", "doctor", "admin"], 
                               format_func=lambda x: {"admin": "管理员", "doctor": "医生", "nurse": "护士"}[x])
        
        submitted = st.form_submit_button("➕ 添加用户", use_container_width=True)
        
        if submitted:
            if not all([new_username, new_name, new_email, new_password]):
                st.error("请填写所有字段")
            elif len(new_password) < 6:
                st.error("密码至少需要6位")
            elif new_username in users:
                st.error("用户名已存在")
            else:
                if add_user(new_username, new_name, new_email, new_password, new_role):
                    st.success(f"成功添加用户：{new_name}")
                    st.rerun()
                else:
                    st.error("添加用户失败")