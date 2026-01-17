"""
模块名称: Authentication Service (认证服务)
功能描述:

    管理用户登录状态和 UI 界面渲染。
    提供基于 Session 的简单认证机制，以及 Streamlit 侧边栏的登录表单渲染。

设计理念:

    1.  **轻量级认证**: 适用于原型系统的 Session 状态管理，非 OAuth/JWT 复杂鉴权。
    2.  **UI/Logic 耦合**: 针对 Streamlit 特性，将渲染逻辑 (`render_login_sidebar`) 与状态检查 (`check_password`) 结合。
    3.  **安全性**: 密码哈希存储 (TODO)，当前版本主要用于访问控制演示。

线程安全性:

    - 依赖 Streamlit 的 `st.session_state`，线程安全性由 Streamlit 框架保证。

依赖关系:

    - `streamlit`: 用于 UI 渲染和 Session 管理。
    - `src.core.settings`: 获取预设的用户名密码配置。
"""

import os
import yaml
import bcrypt
import streamlit as st
import streamlit_authenticator as stauth
from typing import Dict, Any, Optional, Tuple
from src.core.settings import settings

# [全局变量] ============================================================================================================
# 配置文件路径
AUTH_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config", "auth.yaml")

# [定义函数] ############################################################################################################
# [配置管理-加载配置] =====================================================================================================
def load_auth_config() -> Dict[str, Any]:
    """
    加载认证配置文件。
    如果文件不存在，会自动创建默认配置。
    :return: 认证配置字典
    """
    # [step1] 获取绝对路径
    config_path = os.path.abspath(AUTH_CONFIG_PATH)
    
    # [step2] 文件不存在时创建默认配置
    if not os.path.exists(config_path):
        default_config = create_default_config()
        save_auth_config(default_config)
        return default_config
    
    # [step3] 读取并解析 YAML 文件
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# [配置管理-保存配置] =====================================================================================================
def save_auth_config(config: Dict[str, Any]) -> None:
    """
    保存认证配置到文件。
    :param config: 认证配置字典
    """
    # [step1] 获取绝对路径并确保目录存在
    config_path = os.path.abspath(AUTH_CONFIG_PATH)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # [step2] 写入 YAML 文件
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

# [配置管理-创建默认] =====================================================================================================
def create_default_config() -> Dict[str, Any]:
    """
    创建默认认证配置。
    包含 admin, doctor, nurse 三个默认用户。
    :return: 默认配置字典
    """
    # [step1] 生成默认密码哈希 (bcrypt)
    admin_hash = bcrypt.hashpw("admin123".encode(), bcrypt.gensalt()).decode()
    doctor_hash = bcrypt.hashpw("doctor123".encode(), bcrypt.gensalt()).decode()
    nurse_hash = bcrypt.hashpw("nurse123".encode(), bcrypt.gensalt()).decode()
    
    # [step2] 构建配置字典
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

# [核心认证-密码哈希] =====================================================================================================
def hash_password(password: str) -> str:
    """
    对密码进行 bcrypt 哈希处理。
    :param password: 明文密码
    :return: 哈希后的密码字符串
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# [核心认证-获取认证器] ===================================================================================================
def get_authenticator() -> stauth.Authenticate:
    """
    获取 Streamlit 认证器实例。
    使用 session_state 缓存以避免重复创建。
    :return: Authenticate 实例
    """
    # [step1] 检查缓存，如果存在直接返回（隐式逻辑）
    
    # [step2] 初始化认证器（如果缓存中没有）
    if "authenticator" not in st.session_state:
        config = load_auth_config()
        # streamlit-authenticator 0.4.x 版本参数
        st.session_state.authenticator = stauth.Authenticate(
            config['credentials'],
            config['cookie']['name'],
            config['cookie']['key'],
            config['cookie']['expiry_days']
        )
    
    return st.session_state.authenticator

# [用户信息-获取角色] =====================================================================================================
def get_user_role(username: str) -> Optional[str]:
    """
    根据用户名获取用户角色。
    :param username: 用户名
    :return: 角色名称 (admin/doctor/nurse) 或 None
    """
    # [step1] 加载配置
    config = load_auth_config()
    
    # [step2] 查找用户数据并返回角色
    user_data = config.get('credentials', {}).get('usernames', {}).get(username)
    if user_data:
        return user_data.get('role', 'user')
    return None

# [用户信息-获取显示名] ===================================================================================================
def get_user_display_name(username: str) -> str:
    """
    获取用户的显示名称。
    :param username: 用户名
    :return: 显示名称或原用户名
    """
    # [step1] 加载配置
    config = load_auth_config()
    
    # [step2] 查找用户数据并返回名称
    user_data = config.get('credentials', {}).get('usernames', {}).get(username)
    if user_data:
        return user_data.get('name', username)
    return username

# [内部-清除缓存] =========================================================================================================
def _clear_authenticator_cache() -> None:
    """
    清除认证器缓存，强制下次重新加载配置。
    通常在用户信息变更后调用。
    """
    if "authenticator" in st.session_state:
        del st.session_state["authenticator"]

# [用户管理-添加用户] =====================================================================================================
def add_user(username: str, name: str, email: str, password: str, role: str = "nurse") -> bool:
    """
    添加新用户。
    :param username: 用户名
    :param name: 显示名称
    :param email: 邮箱
    :param password: 明文密码
    :param role: 角色
    :return: 是否成功
    """
    # [step1] 加载当前配置
    config = load_auth_config()
    
    # [step2] 检查用户名是否已存在
    if username in config['credentials']['usernames']:
        return False
    
    # [step3] 添加用户数据
    config['credentials']['usernames'][username] = {
        "email": email,
        "failed_login_attempts": 0,
        "logged_in": False,
        "name": name,
        "password": hash_password(password),
        "role": role
    }
    
    # [step4] 保存配置并刷新缓存
    save_auth_config(config)
    _clear_authenticator_cache()
    return True

# [用户管理-删除用户] =====================================================================================================
def delete_user(username: str) -> bool:
    """
    删除指定用户。
    :param username: 用户名
    :return: 是否成功
    """
    # [step1] 加载配置
    config = load_auth_config()
    
    # [step2] 禁止删除 admin
    if username == "admin":
        return False
    
    # [step3] 删除用户并保存
    if username in config['credentials']['usernames']:
        del config['credentials']['usernames'][username]
        save_auth_config(config)
        _clear_authenticator_cache()
        return True
    
    return False

# [用户管理-更新密码] =====================================================================================================
def update_user_password(username: str, new_password: str) -> bool:
    """
    更新用户密码。
    :param username: 用户名
    :param new_password: 新明文密码
    :return: 是否成功
    """
    # [step1] 加载配置
    config = load_auth_config()
    
    # [step2] 更新密码并保存
    if username in config['credentials']['usernames']:
        config['credentials']['usernames'][username]['password'] = hash_password(new_password)
        save_auth_config(config)
        _clear_authenticator_cache()
        return True
    
    return False

# [用户信息-获取所有] =====================================================================================================
def get_all_users() -> Dict[str, Dict[str, Any]]:
    """
    获取所有用户信息（脱敏）。
    :return: 用户信息字典
    """
    # [step1] 加载配置
    config = load_auth_config()
    users = {}
    
    # [step2] 遍历并重组数据（排除敏感信息）
    for username, data in config['credentials']['usernames'].items():
        users[username] = {
            "name": data.get('name', username),
            "email": data.get('email', ''),
            "role": data.get('role', 'user')
        }
    return users

# [界面-渲染登录页] =======================================================================================================
def render_login_page() -> Tuple[Optional[str], bool, Optional[str]]:
    """
    渲染登录页面。
    :return: (用户名, 认证状态, 显示名称)
    """
    authenticator = get_authenticator()
    
    # [step1] 注入自定义 CSS 样式
    st.markdown("""
    <style>
    .login-container { max-width: 400px; margin: 0 auto; padding: 2rem; }
    .login-header { text-align: center; margin-bottom: 2rem; }
    .login-header h1 { color: #1e3a5f; font-size: 1.8rem; margin-bottom: 0.5rem; }
    .login-header p { color: #64748b; font-size: 0.9rem; }
    .stTextInput input { border-radius: 8px; border: 2px solid #e2e8f0; padding: 0.8rem 1rem; width: 100%; }

    div[data-testid="stForm"] > div:first-child h3 { display: none; }
    div[data-testid="stForm"] { text-align: center; }
    div[data-testid="stForm"] .stTextInput, div[data-testid="stForm"] .stPasswordInput { margin-left: auto; margin-right: auto; max-width: 100%; }
    
    div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
        width: 100% !important;
        margin: 0 auto !important;
    }

    div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] > button {
        width: auto !important;
        min-width: 120px !important;
        border-radius: 8px !important;
        background-color: #ffffff !important;
        color: #475569 !important;
        border: 1px solid #e2e8f0 !important;
        padding: 0.7rem 1.4rem !important;
        margin: 0 auto !important;
        display: block !important;
        align-items: center !important;
        justify-content: center !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
    }
    
    div[data-testid="stForm"] div[data-testid="stFormSubmitButton"] > button:hover {
        background-color: #f8fafc !important;
        border-color: #cbd5e1 !important;
        color: #1e293b !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    /* 确保表单容器内的按钮完全居中 */
    div[data-testid="stForm"] {
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    
    div[data-testid="stForm"] > div {
        width: 100% !important;
        display: flex !important;
        flex-direction: column !important;
        align-items: center !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # [step2] 检查现有会话状态
    if st.session_state.get("authentication_status") == True:
        return (
            st.session_state.get("username"),
            True,
            st.session_state.get("name")
        )
    
    # [step3] 渲染登录表单
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="login-header">
            <div style="font-size: 4rem; margin-bottom: 1rem;">🏥</div>
            <h1>智能医疗诊断系统</h1>
        </div>
        <div style="background-color: #e8f4f8; padding: 15px; border-radius: 5px; border: 1px solid #bce3eb; color: #315e6b; margin-bottom: 2rem; text-align: left;">
            <div style="text-align: center; font-weight: bold; font-size: 16px; margin-bottom: 8px;">智能多学科会诊系统 (MDT) v1.0.0</div>
            <div style="font-size: 14px; line-height: 1.5;">模拟真实医院的 MDT 流程，由多个 AI 专科医生协同工作，提供全面的诊断建议。</div>
        </div>
        """, unsafe_allow_html=True)
        
        # [step4] 调用 authenticator.login
        try:
            # 尝试新版 API
            authenticator.login(location='main', fields={'Form name': '用户登录', 'Username': '用户名', 'Password': '密码', 'Login': '登录'})
        except TypeError:
            # 回退旧版 API
            try:
                authenticator.login('用户登录', 'main')
            except Exception as e:
                st.error(f"登录组件加载失败: {e}")
                return None, False, None
        except Exception as e:
            st.error(f"登录组件加载失败: {e}")
            return None, False, None
        
        # [step5] 处理登录结果
        name = st.session_state.get("name")
        authentication_status = st.session_state.get("authentication_status")
        username = st.session_state.get("username")
        
        if authentication_status == False:
            st.error("用户名或密码错误")
        
        return username, authentication_status == True, name
    
    return None, False, None

# [界面-渲染侧边栏] =======================================================================================================
def render_user_info_sidebar(authenticator: stauth.Authenticate, username: str) -> None:
    """
    在侧边栏渲染用户信息和功能按钮。
    :param authenticator: 认证器实例
    :param username: 当前用户名
    """
    # [step1] 获取用户信息
    role = get_user_role(username)
    name = get_user_display_name(username)
    role_display = {"admin": "👑 管理员", "doctor": "👨‍⚕️ 医生", "nurse": "👩‍⚕️ 护士"}
    
    st.sidebar.markdown("---")
    
    # [step2] 显示用户信息卡片
    st.sidebar.markdown(f"""
        <div style="background-color: white; padding: 1.2rem; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 1rem; border: 1px solid #f0f2f6; text-align: center;">
            <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">👤</div>
            <div style="font-weight: 600; font-size: 1.1rem; color: #1f2937; margin-bottom: 0.2rem;">{name}</div>
            <div style="display: inline-block; background-color: #f3f4f6; color: #4b5563; padding: 0.2rem 0.8rem; border-radius: 9999px; font-size: 0.8rem; margin-bottom: 1rem;">{role_display.get(role, '用户')}</div>
        </div>
    """, unsafe_allow_html=True)
    
    # [step3] 显示管理入口（仅管理员）
    if role == "admin":
        if st.sidebar.button("👥 用户管理", use_container_width=True):
            st.query_params["page"] = "user_management"
            st.rerun()
    
    # [step4] 显示登出按钮
    try:
        authenticator.logout(button_name="🚪 退出登录", location="sidebar", key="logout_btn")
    except TypeError:
        authenticator.logout("🚪 退出登录", "sidebar", key="logout_btn")

# [界面-渲染用户管理] =====================================================================================================
def render_user_management() -> None:
    """
    渲染用户管理界面（仅管理员可用）。
    """
    st.markdown("<h2 style='text-align: center;'>👥 用户管理</h2>", unsafe_allow_html=True)
    
    # [step1] 权限校验
    current_user = st.session_state.get("username")
    current_role = get_user_role(current_user)
    if current_role != "admin":
        st.warning("⚠️ 仅管理员可以管理用户")
        return
    
    # [step2] 获取并筛选用户列表
    users = get_all_users()
    filter_col1, filter_col2, filter_col3 = st.columns([1, 2, 1])
    with filter_col2:
        selected_role_filter = st.selectbox("筛选用户角色", ["全部", "护士", "医生", "管理员"], key="user_filter_role", label_visibility="collapsed")
    
    role_map_cn = {"全部": "all", "护士": "nurse", "医生": "doctor", "管理员": "admin"}
    filter_role_code = role_map_cn[selected_role_filter]
    
    # [step3] 渲染用户列表
    for username, data in users.items():
        if filter_role_code != "all" and data['role'] != filter_role_code:
            continue
            
        role_emoji = {"admin": "👑", "doctor": "👨‍⚕️", "nurse": "👩‍⚕️"}.get(data['role'], "👤")
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        with col1: st.text(f"{role_emoji} {data['name']}")
        with col2: st.text(f"@{username}")
        with col3: st.text(data['role'])
        with col4:
            if username != "admin" and username != current_user:
                if st.button("🗑️", key=f"del_{username}", help="删除用户"):
                    if delete_user(username):
                        st.success(f"已删除用户 {username}")
                        st.rerun()
    
    st.markdown("---")
    
    # [step4] 渲染添加用户表单
    st.markdown("<h3 style='text-align: center;'>添加新用户</h3>", unsafe_allow_html=True)
    with st.form("add_user_form"):
        new_username = st.text_input("用户名", placeholder="例如：zhangsan")
        new_name = st.text_input("姓名", placeholder="例如：张三")
        new_email = st.text_input("邮箱", placeholder="例如：zhangsan@hospital.com")
        new_password = st.text_input("密码", type="password", placeholder="至少6位")
        new_role = st.selectbox("角色", ["nurse", "doctor", "admin"], format_func=lambda x: {"admin": "管理员", "doctor": "医生", "nurse": "护士"}[x])
        
        if st.form_submit_button("➕ 添加用户", use_container_width=True):
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
