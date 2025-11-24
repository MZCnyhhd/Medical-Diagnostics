import sqlite3
import json
from datetime import datetime
import os

DB_PATH = "data/medical_diagnostics.db"

def init_db():
    """初始化数据库表"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 创建问诊记录表
    c.execute('''
        CREATE TABLE IF NOT EXISTS consultations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            report_content TEXT,
            diagnosis_result TEXT,
            structured_data TEXT
        )
    ''')
    
    # 检查 structured_data 列是否存在（针对旧表迁移）
    try:
        c.execute('SELECT structured_data FROM consultations LIMIT 1')
    except sqlite3.OperationalError:
        c.execute('ALTER TABLE consultations ADD COLUMN structured_data TEXT')
    
    conn.commit()
    conn.close()

def save_consultation(report_content, diagnosis_result, structured_data=None):
    """保存一次问诊记录"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 确保 structured_data 是 JSON 字符串
    if isinstance(structured_data, (dict, list)):
        structured_data = json.dumps(structured_data, ensure_ascii=False)
        
    c.execute('''
        INSERT INTO consultations (timestamp, report_content, diagnosis_result, structured_data)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, report_content, diagnosis_result, structured_data))
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
