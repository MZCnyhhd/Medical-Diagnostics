"""
模块名称: Export Tools (导出工具)
功能描述:

    提供将诊断结果导出为文件的功能。
    目前支持生成 Markdown 文件流，供用户下载。

设计理念:

    1.  **流式处理**: 使用 `io.BytesIO` 在内存中生成文件流，避免产生临时文件。
    2.  **格式通用**: Markdown 格式通用性强，易于阅读和转换。

线程安全性:

    - 无状态函数，线程安全。

依赖关系:

    - 标准库 `io`.
"""

import io


def generate_markdown(content: str) -> io.BytesIO:
    """生成 Markdown 文件流"""
    buffer = io.BytesIO()
    buffer.write(content.encode('utf-8'))
    buffer.seek(0)
    return buffer
