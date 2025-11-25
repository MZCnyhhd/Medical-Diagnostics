def get_css():
    return """
<style>
    /* Global Styles */
    .reportview-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    }
    
    /* Main Header - More Prominent */
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.1);
        letter-spacing: -0.5px;
    }
    
    /* Sub Headers - Better Hierarchy */
    .sub-header {
        font-size: 1.75rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 0.5rem;
    }
    
    /* Text Areas - Better Readability */
    .stTextArea textarea {
        font-size: 1rem;
        line-height: 1.6;
        border-radius: 8px;
        border: 2px solid #e0e6ed;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #3498db;
        box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background-color: #ffffff;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.75rem 1rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
        transition: all 0.3s ease;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #f8f9fa;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Button Enhancements */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        border: none;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15);
    }
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Download Buttons */
    .stDownloadButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stError {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stWarning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        border-radius: 8px;
        padding: 1rem;
    }
    
    .stInfo {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        border-radius: 8px;
        padding: 1rem;
    }
    
    /* Specialist Cards - Modern Look */
    .specialist-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 1.75rem;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        margin-bottom: 1.5rem;
        border-left: 5px solid #28a745;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .specialist-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12);
    }
    
    .specialist-header {
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 0.75rem;
        color: #2e7d32;
    }
    
    .specialist-content {
        color: #495057;
        line-height: 1.7;
        font-size: 1rem;
    }
    
    .specialist-title {
        font-weight: 700;
        font-size: 1.3rem;
        margin-bottom: 0.75rem;
        color: #212529;
    }
    
    /* Sidebar Enhancements */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #ffffff 0%, #f8f9fa 100%);
        border-right: 1px solid #e0e6ed;
    }
    
    section[data-testid="stSidebar"] .stSelectbox {
        margin-bottom: 1rem;
    }
    
    /* Radio Button Styling */
    .stRadio > label {
        font-weight: 600;
        color: #2c3e50;
    }
    
    /* Divider Enhancement */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid #e0e6ed;
    }
    
    /* Chat Message Styling */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Status Container */
    .stStatus {
        border-radius: 12px;
        padding: 1rem;
        margin: 1rem 0;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Selectbox Styling */
    .stSelectbox select {
        border-radius: 8px;
        border: 2px solid #667eea;
        font-weight: 500;
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        color: #2c3e50;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
    }
    
    .stSelectbox select:hover {
        border-color: #764ba2;
        background: linear-gradient(135deg, #f0f4ff 0%, #e8eeff 100%);
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.15);
    }
    
    .stSelectbox select:focus {
        border-color: #667eea;
        background: linear-gradient(135deg, #ffffff 0%, #f0f4ff 100%);
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.2);
        outline: none;
    }
    
    /* File Uploader Styling */
    .stFileUploader {
        border-radius: 12px;
        background-color: #ffffff;
        padding: 1.5rem;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* Container Enhancements */
    .element-container {
        margin-bottom: 1rem;
    }
    
    /* Smooth Animations */
    * {
        transition: background-color 0.3s ease, color 0.3s ease;
    }
    
    /* Floating Chat Button - Custom Green Squircle */
    div[data-testid="stPopover"] {
        position: fixed !important;
        bottom: 40px !important;
        right: 40px !important;
        z-index: 99999 !important;
        width: 60px !important;
        height: 60px !important;
    }
    
    div[data-testid="stPopover"] button {
        width: 100% !important;
        height: 100% !important;
        border-radius: 16px !important;
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%) !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(46, 204, 113, 0.4) !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    }
    
    div[data-testid="stPopover"] button:hover {
        transform: scale(1.05) translateY(-3px);
        box-shadow: 0 8px 25px rgba(46, 204, 113, 0.6) !important;
        background: linear-gradient(135deg, #4cd137 0%, #44bd32 100%) !important;
    }
    
    div[data-testid="stPopover"] button:active {
        transform: scale(0.95);
    }
    
    div[data-testid="stPopover"] button > * {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
    }
    
    div[data-testid="stPopover"] button::after {
        content: "";
        display: block !important;
        width: 32px !important;
        height: 32px !important;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='white'%3E%3Cpath d='M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z'/%3E%3C/svg%3E") !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
        background-size: contain !important;
        visibility: visible !important;
        opacity: 1 !important;
    }
</style>
"""
