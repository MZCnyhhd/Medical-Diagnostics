"""
日志服务模块：统一的日志管理

使用 Python 标准 logging 模块，支持：
- 日志级别控制（通过环境变量 LOG_LEVEL）
- 格式化输出（时间戳 + 级别 + 消息）
- 控制台彩色输出
"""

import logging
import os
import sys

# 获取日志级别（默认 INFO）
_log_level = os.getenv("LOG_LEVEL", "INFO").upper()
_level_map = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARN": logging.WARNING,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}

# 创建日志器
_logger = logging.getLogger("medical_diagnostics")
_logger.setLevel(_level_map.get(_log_level, logging.INFO))

# 避免重复添加 handler
if not _logger.handlers:
    # 创建控制台 handler
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setLevel(logging.DEBUG)
    
    # 设置格式
    _formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    _console_handler.setFormatter(_formatter)
    _logger.addHandler(_console_handler)


def log_info(*args, **kwargs):
    """记录 INFO 级别日志"""
    message = " ".join(str(arg) for arg in args)
    _logger.info(message)


def log_warn(*args, **kwargs):
    """记录 WARNING 级别日志"""
    message = " ".join(str(arg) for arg in args)
    _logger.warning(message)


def log_error(*args, **kwargs):
    """记录 ERROR 级别日志"""
    message = " ".join(str(arg) for arg in args)
    _logger.error(message)


def log_debug(*args, **kwargs):
    """记录 DEBUG 级别日志"""
    message = " ".join(str(arg) for arg in args)
    _logger.debug(message)
