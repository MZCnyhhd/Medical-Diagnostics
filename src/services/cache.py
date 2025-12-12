"""
诊断缓存服务
============
用于缓存相似病例的诊断结果，提高响应速度
"""

import hashlib
import time
from typing import Optional, Dict, Any
import sqlite3
from pathlib import Path

from src.services.logging import log_info, log_warn


class DiagnosisCache:
    """诊断结果缓存管理器"""
    
    def __init__(self, db_path: str = "data/medical_diagnostics.db"):
        """
        初始化缓存管理器
        
        Args:
            db_path: 数据库文件路径
        """
        self.db_path = db_path
        self._init_cache_table()
    
    def _init_cache_table(self):
        """初始化缓存表"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 创建缓存表
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
            
            # 创建索引
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_created 
                ON diagnosis_cache(created_at)
            """)
            
            conn.commit()
            conn.close()
            log_info("[Cache] 诊断缓存表初始化成功")
        except Exception as e:
            log_warn(f"[Cache] 缓存表初始化失败: {e}")
    
    @staticmethod
    def compute_hash(report: str) -> str:
        """
        计算报告的哈希值（用于缓存键）
        
        Args:
            report: 医疗报告文本
            
        Returns:
            报告的 MD5 哈希值
        """
        # 标准化文本（去除多余空白、统一换行符）
        normalized = " ".join(report.split())
        normalized = normalized.lower()  # 转小写
        
        # 计算哈希
        return hashlib.md5(normalized.encode('utf-8')).hexdigest()
    
    def get(self, report_hash: str, ttl: int = 3600) -> Optional[Dict[str, Any]]:
        """
        获取缓存的诊断结果
        
        Args:
            report_hash: 报告哈希值
            ttl: 缓存有效期（秒）
            
        Returns:
            缓存的诊断结果，如果不存在或已过期则返回 None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 查询缓存
            cursor.execute("""
                SELECT diagnosis_result, confidence, created_at, hit_count
                FROM diagnosis_cache
                WHERE report_hash = ?
            """, (report_hash,))
            
            result = cursor.fetchone()
            
            if result:
                diagnosis_result, confidence, created_at, hit_count = result
                current_time = int(time.time())
                
                # 检查是否过期
                if current_time - created_at < ttl:
                    # 更新访问时间和命中次数
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
                    # 缓存过期，删除
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
    
    def set(self, report_hash: str, diagnosis: str, confidence: float = 0.0):
        """
        保存诊断结果到缓存
        
        Args:
            report_hash: 报告哈希值
            diagnosis: 诊断结果
            confidence: 置信度 (0-1)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            current_time = int(time.time())
            
            # 插入或更新缓存
            cursor.execute("""
                INSERT OR REPLACE INTO diagnosis_cache
                (report_hash, diagnosis_result, confidence, created_at, accessed_at, hit_count)
                VALUES (?, ?, ?, ?, ?, 0)
            """, (report_hash, diagnosis, confidence, current_time, current_time))
            
            conn.commit()
            conn.close()
            
            log_info(f"[Cache] 缓存保存成功: {report_hash[:8]}...")
            
        except Exception as e:
            log_warn(f"[Cache] 保存缓存失败: {e}")
    
    def clear_expired(self, ttl: int = 3600):
        """
        清理过期的缓存
        
        Args:
            ttl: 缓存有效期（秒）
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            expired_time = int(time.time()) - ttl
            
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
    
    def clear_all(self) -> int:
        """
        清除所有缓存
        
        Returns:
            删除的缓存条数
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("DELETE FROM diagnosis_cache")
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            log_info(f"[Cache] 已清除所有缓存，共 {deleted_count} 条")
            return deleted_count
            
        except Exception as e:
            log_warn(f"[Cache] 清除缓存失败: {e}")
            return 0
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计数据
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # 总缓存数
            cursor.execute("SELECT COUNT(*) FROM diagnosis_cache")
            total_count = cursor.fetchone()[0]
            
            # 总命中次数
            cursor.execute("SELECT SUM(hit_count) FROM diagnosis_cache")
            total_hits = cursor.fetchone()[0] or 0
            
            # 平均命中率
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


# 全局缓存实例
_cache = None

def get_cache() -> DiagnosisCache:
    """获取全局缓存实例"""
    global _cache
    if _cache is None:
        _cache = DiagnosisCache()
    return _cache
