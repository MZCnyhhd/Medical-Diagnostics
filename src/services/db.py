import sqlite3
from datetime import datetime
import os
from pathlib import Path

DB_PATH = "data/medical_diagnostics.db"

def init_db():
    """初始化数据库表"""
    # 确保数据库目录存在（Streamlit Cloud 兼容性修复）
    db_dir = Path(DB_PATH).parent
    db_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 创建问诊记录表
    c.execute('''
        CREATE TABLE IF NOT EXISTS consultations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            report_content TEXT,
            diagnosis_result TEXT
        )
    ''')
    
    conn.commit()
    conn.close()

def save_consultation(report_content, diagnosis_result):
    """保存一次问诊记录"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    c.execute('''
        INSERT INTO consultations (timestamp, report_content, diagnosis_result)
        VALUES (?, ?, ?)
    ''', (timestamp, report_content, diagnosis_result))
    conn.commit()
    conn.close()

def get_history():
    """获取所有历史记录（按时间倒序）"""
    if not os.path.exists(DB_PATH):
        return []
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT id, timestamp, report_content, diagnosis_result FROM consultations ORDER BY id DESC')
    rows = c.fetchall()
    conn.close()
    
    history = []
    for row in rows:
        history.append({
            "id": row[0],
            "timestamp": row[1],
            "report_content": row[2],
            "diagnosis_result": row[3]
        })
    return history
