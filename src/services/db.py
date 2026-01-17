"""
模块名称: Consultation Database (会诊数据库)
功能描述:

    管理全科会诊记录的持久化存储。
    保存用户的医疗报告、诊断结果、专科医生建议等完整会诊历史。

设计理念:

    1.  **结构化存储**: 将非结构化的医疗文本转化为结构化的数据库记录 (Consultation -> Diagnosis)。
    2.  **简单易用**: 封装 SQLite 初始化、插入和查询操作，屏蔽 SQL 细节。
    3.  **数据完整性**: 使用事务保证数据写入的一致性。

线程安全性:

    - 类似 `cache.py`，依赖 SQLite 的并发控制机制。

依赖关系:

    - `sqlite3`: 嵌入式数据库。
    - `src.core.settings`: 获取主数据库路径。
"""

import sqlite3
import json
import os
from pathlib import Path
from datetime import datetime

# [全局变量] ============================================================================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "data" / "medical_diagnostics.db"

# [定义函数] ############################################################################################################
# [初始化-数据库] ========================================================================================================
def init_db():
    """
    初始化数据库表。
    自动创建数据库目录和表结构。
    """
    # [step1] 确保数据库目录存在（Streamlit Cloud 兼容性修复）
    db_dir = Path(DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    # [step2] 连接数据库
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # [step3] 创建问诊记录表 (consultations)
    c.execute('''
        CREATE TABLE IF NOT EXISTS consultations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            report_content TEXT,
            diagnosis_result TEXT
        )
    ''')
    
    # [step4] 提交并关闭
    conn.commit()
    conn.close()

# [操作-保存记录] ========================================================================================================
def save_consultation(report_content, diagnosis_result):
    """
    保存一次问诊记录。
    :param report_content: 医疗报告内容
    :param diagnosis_result: 诊断结果
    """
    # [step1] 连接数据库
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # [step2] 获取当前时间
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    # [step3] 插入记录
    c.execute('''
        INSERT INTO consultations (timestamp, report_content, diagnosis_result)
        VALUES (?, ?, ?)
    ''', (timestamp, report_content, diagnosis_result))
    
    # [step4] 提交并关闭
    conn.commit()
    conn.close()

# [操作-获取历史] ========================================================================================================
def get_history():
    """
    获取所有历史记录（按时间倒序）。
    :return: 历史记录列表字典
    """
    # [step1] 检查数据库文件是否存在
    if not os.path.exists(DB_PATH):
        return []
    
    # [step2] 查询所有记录
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, timestamp, report_content, diagnosis_result FROM consultations ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    
    # [step3] 转换为字典列表
    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "timestamp": row[1],
            "report_content": row[2],
            "diagnosis_result": row[3]
        })
    return history
