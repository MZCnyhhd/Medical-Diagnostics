"""
模块名称: Diagnosis Cache (诊断缓存)
功能描述:

    提供诊断结果的缓存机制，避免对相同报告的重复诊断，节省 LLM Token 和时间成本。
    使用 SQLite 存储缓存数据，支持 TTL (Time-To-Live) 过期机制。

设计理念:

    1.  **内容寻址**: 使用输入报告的 MD5 哈希作为缓存 Key，确保内容变更自动失效。
    2.  **持久化存储**: 相比内存缓存，SQLite 重启不丢失，适合长文本诊断场景。
    3.  **自动过期**: 每次读取检查时间戳，自动过滤过期数据。

线程安全性:

    - SQLite 默认支持多线程并发读取，写入时需注意锁竞争 (WAL 模式可优化)。

依赖关系:

    - `sqlite3`: 嵌入式数据库。
    - `src.core.settings`: 获取缓存数据库路径。
"""

import sqlite3
import hashlib
import json
import time
from typing import Optional, Dict, Any
from pathlib import Path
from src.services.logging import log_info, log_warn

# [定义类] ##############################################################################################################
# [缓存管理器] ==========================================================================================================
class DiagnosisCache:
    """
    诊断结果缓存管理器。
    使用 SQLite 持久化存储相似病例的诊断结果，以提高响应速度。
    """
    
    # [初始化] ============================================================================================================
    def __init__(self, db_path: str = "data/medical_diagnostics.db"):
        """
        初始化缓存管理器。
        :param db_path: 数据库文件路径
        """
        # [step1] 保存路径
        self.db_path = db_path
        # [step2] 自动初始化表结构
        self._init_cache_table()
    
    # [内部-初始化表] =====================================================================================================
    def _init_cache_table(self):
        """初始化缓存数据库表结构"""
        try:
            # [step1] 连接数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # [step2] 创建缓存表（如果不存在）
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS diagnosis_cache (
                    report_hash TEXT PRIMARY KEY,
                    diagnosis_result TEXT NOT NULL,
                    confidence REAL,
                    created_at INTEGER NOT NULL,
                    accessed_at INTEGER NOT NULL,
                    hit_count INTEGER DEFAULT 0
                )
            """)
            
            # [step3] 创建时间索引（优化过期清理）
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_created 
                ON diagnosis_cache(created_at)
            """)
            
            # [step4] 提交并关闭
            conn.commit()
            conn.close()
            log_info("[Cache] 诊断缓存表初始化成功")
        except Exception as e:
            log_warn(f"[Cache] 缓存表初始化失败: {e}")
    
    # [工具-计算哈希] =====================================================================================================
    @staticmethod
    def compute_hash(report: str) -> str:
        """
        计算报告内容的 MD5 哈希值。
        用于生成唯一的缓存键。
        :param report: 医疗报告文本
        :return: 32位十六进制哈希字符串
        """
        # [step1] 文本标准化（去除空白、统一小写）
        normalized = " ".join(report.split())
        normalized = normalized.lower()
        
        # [step2] 计算 MD5
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    # [操作-读取缓存] =====================================================================================================
    def get(self, report_hash: str, ttl: int = 3600) -> Optional[Dict[str, Any]]:
        """
        根据哈希获取缓存的诊断结果。
        会自动检查 TTL 并更新访问统计。
        :param report_hash: 报告哈希值
        :param ttl: 缓存有效期（秒）
        :return: 缓存结果字典或 None
        """
        try:
            # [step1] 查询数据库
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT diagnosis_result, confidence, created_at, hit_count
                FROM diagnosis_cache
                WHERE report_hash = ?
            """, (report_hash,))
            
            result = cursor.fetchone()
            
            # [step2] 命中处理
            if result:
                diagnosis_result, confidence, created_at, hit_count = result
                current_time = int(time.time())
                
                # [step3] TTL 检查
                if current_time - created_at < ttl:
                    # 更新统计
                    cursor.execute("""
                        UPDATE diagnosis_cache
                        SET accessed_at = ?, hit_count = ?
                        WHERE report_hash = ?
                    """, (current_time, hit_count + 1, report_hash))
                    
                    conn.commit()
                    conn.close()
                    
                    log_info(f"[Cache] 缓存命中: {report_hash[:8]}... (命中次数: {hit_count + 1})")
                    
                    return {
                        "diagnosis": diagnosis_result,
                        "confidence": confidence,
                        "cached": True,
                        "hit_count": hit_count + 1
                    }
                else:
                    # [step4] 过期清理
                    cursor.execute("""
                        DELETE FROM diagnosis_cache
                        WHERE report_hash = ?
                    """, (report_hash,))
                    
                    conn.commit()
                    log_info(f"[Cache] 缓存已过期并删除: {report_hash[:8]}...")
            
            conn.close()
            return None
            
        except Exception as e:
            log_warn(f"[Cache] 读取缓存失败: {e}")
            return None
    
    # [操作-写入缓存] =====================================================================================================
    def set(self, report_hash: str, diagnosis: str, confidence: float = 0.0):
        """
        写入或更新缓存记录。
        :param report_hash: 报告哈希值
        :param diagnosis: 诊断结果内容
        :param confidence: 置信度分数
        """
        try:
            # [step1] 准备数据
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            current_time = int(time.time())
            
            # [step2] 插入或替换记录
            cursor.execute("""
                INSERT OR REPLACE INTO diagnosis_cache
                (report_hash, diagnosis_result, confidence, created_at, accessed_at, hit_count)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (report_hash, diagnosis, confidence, current_time, current_time))
            
            # [step3] 提交事务
            conn.commit()
            conn.close()
            
            log_info(f"[Cache] 缓存保存成功: {report_hash[:8]}...")
            
        except Exception as e:
            log_warn(f"[Cache] 保存缓存失败: {e}")
    
    # [维护-清理过期] =====================================================================================================
    def clear_expired(self, ttl: int = 3600):
        """
        批量清理所有过期缓存。
        :param ttl: 缓存有效期（秒）
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # [step1] 计算过期时间阈值
            expired_time = int(time.time()) - ttl
            
            # [step2] 删除过期记录
            cursor.execute("""
                DELETE FROM diagnosis_cache
                WHERE created_at < ?
            """, (expired_time,))
            
            deleted_count = cursor.rowcount
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                log_info(f"[Cache] 清理了 {deleted_count} 条过期缓存")
                
        except Exception as e:
            log_warn(f"[Cache] 清理过期缓存失败: {e}")
    
    # [维护-清空所有] =====================================================================================================
    def clear_all(self) -> int:
        """
        强制清空所有缓存记录。
        :return: 删除的记录数
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # [step1] 删除全表数据
            cursor.execute("DELETE FROM diagnosis_cache")
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            log_info(f"[Cache] 已清除所有缓存，共 {deleted_count} 条")
            return deleted_count
            
        except Exception as e:
            log_warn(f"[Cache] 清除缓存失败: {e}")
            return 0
    
    # [统计-获取信息] =====================================================================================================
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存系统的统计指标。
        :return: 包含总数、命中数、平均命中的字典
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # [step1] 聚合查询
            cursor.execute("SELECT COUNT(*) FROM diagnosis_cache")
            total_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT SUM(hit_count) FROM diagnosis_cache")
            total_hits = cursor.fetchone()[0] or 0
            
            cursor.execute("SELECT AVG(hit_count) FROM diagnosis_cache")
            avg_hits = cursor.fetchone()[0] or 0
            
            conn.close()
            
            return {
                "total_cached": total_count,
                "total_hits": total_hits,
                "average_hits": round(avg_hits, 2)
            }
            
        except Exception as e:
            log_warn(f"[Cache] 获取统计信息失败: {e}")
            return {
                "total_cached": 0,
                "total_hits": 0,
                "average_hits": 0
            }

# [定义函数] ##############################################################################################################
# [全局单例-获取缓存] =====================================================================================================
_cache = None

def get_cache() -> DiagnosisCache:
    """
    获取 DiagnosisCache 的全局单例。
    :return: 缓存管理器实例
    """
    global _cache
    if _cache is None:
        _cache = DiagnosisCache()
    return _cache
