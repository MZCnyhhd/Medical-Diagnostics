"""
模块名称: Unified Logging (统一日志服务)
功能描述:

    提供全系统统一的日志记录功能。
    支持控制台输出和文件持久化，自动根据配置调整日志级别 (DEBUG/INFO)。

设计理念:

    1.  **单例配置**: 全局只配置一次 Logger，避免重复输出。
    2.  **结构化**: 虽然目前输出文本，但保留了扩展为 JSON 结构化日志的能力。
    3.  **简单易用**: 封装 `log_info`, `log_error` 等快捷函数，减少样板代码。

线程安全性:

    - Python 标准库 `logging` 模块是线程安全的。

依赖关系:

    - `logging`: 标准库。
    - `colorlog`: 控制台彩色输出 (增强可读性)。
"""

import logging
import sys
import os
from pathlib import Path
from colorlog import ColoredFormatter

# [初始化配置] ===========================================================================================================
# [step1] 获取日志级别（默认 INFO）
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

# [step2] 创建并配置日志器
_logger = logging.getLogger("medical_diagnostics")
_logger.setLevel(_level_map.get(_log_level, logging.INFO))

# [step3] 配置 Handler (避免重复添加)
if not _logger.handlers:
    # 创建控制台 handler
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setLevel(logging.DEBUG)
    
    # 设置格式：时间 [级别] 消息
    # 使用 colorlog 进行彩色输出
    try:
        from colorlog import ColoredFormatter
        _formatter = ColoredFormatter(
            "%(log_color)s%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            }
        )
    except ImportError:
        # 降级处理
        _formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

    _console_handler.setFormatter(_formatter)
    _logger.addHandler(_console_handler)
    # 强制不向上传播，防止 Streamlit root logger 重复打印
    _logger.propagate = False

# [定义函数] ############################################################################################################
# [日志-Info] ===========================================================================================================
def log_info(*args, **kwargs):
    """
    记录 INFO 级别日志。
    支持多个参数拼接。
    """
    message = " ".join(str(arg) for arg in args)
    _logger.info(message)

# [日志-Warn] ===========================================================================================================
def log_warn(*args, **kwargs):
    """
    记录 WARNING 级别日志。
    """
    message = " ".join(str(arg) for arg in args)
    _logger.warning(message)

# [日志-Error] ==========================================================================================================
def log_error(*args, **kwargs):
    """
    记录 ERROR 级别日志。
    """
    message = " ".join(str(arg) for arg in args)
    _logger.error(message)

# [日志-Debug] ==========================================================================================================
def log_debug(*args, **kwargs):
    """
    记录 DEBUG 级别日志。
    """
    message = " ".join(str(arg) for arg in args)
    _logger.debug(message)
