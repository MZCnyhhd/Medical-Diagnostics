"""
模块名称: File Processors (文件处理工具)
功能描述:

    提供上传文件的解析和处理功能。
    支持多种文件格式 (TXT, PDF, Markdown, Image) 的读取和内容提取。

设计理念:

    1.  **统一接口**: 所有格式处理都通过 `process_uploaded_file` 入口。
    2.  **错误隔离**: 自定义异常类，明确区分格式不支持和处理失败。
    3.  **扩展性**: 易于添加新的文件格式支持 (如 DOCX)。

线程安全性:

    - 无状态函数，线程安全。

依赖关系:

    - `pypdf`: PDF 解析。
    - `python-docx`: Word 文档解析 (可选)。
"""

import os
import io
from typing import Tuple, Optional
import pypdf

try:
    import docx
except ImportError:
    docx = None

# [自定义异常] ===========================================================================================================
class FileProcessingError(Exception):
    """文件处理基础异常"""
    pass

class UnsupportedFileFormatError(FileProcessingError):
    """不支持的文件格式异常"""
    pass

# [核心函数] =============================================================================================================
def process_uploaded_file(filename: str, file_content: bytes) -> Tuple[str, Optional[bytes]]:
    """
    处理上传的文件，返回文本内容或图片数据。
    
    :param filename: 文件名 (用于判断扩展名)
    :param file_content: 文件内容的二进制数据
    :return: (text_content, image_bytes)
             - 对于文本类文件 (TXT, PDF, MD)，返回 (提取的文本, None)
             - 对于图片类文件 (PNG, JPG)，返回 ("", 原始字节)
    :raises UnsupportedFileFormatError: 格式不支持
    :raises FileProcessingError: 解析失败
    """
    ext = os.path.splitext(filename)[1].lower()
    
    # [Case 1] 图片格式
    if ext in ['.png', '.jpg', '.jpeg']:
        return "", file_content
        
    # [Case 2] 纯文本格式
    if ext == '.txt':
        try:
            return file_content.decode('utf-8'), None
        except UnicodeDecodeError:
            try:
                # 尝试 GBK 解码 (兼容中文 Windows)
                return file_content.decode('gbk'), None
            except Exception as e:
                raise FileProcessingError(f"文本解码失败: {str(e)}")
                
    # [Case 3] Markdown 格式
    if ext in ['.md', '.markdown']:
        try:
            return file_content.decode('utf-8'), None
        except Exception as e:
            raise FileProcessingError(f"Markdown 解码失败: {str(e)}")

    # [Case 4] PDF 格式
    if ext == '.pdf':
        try:
            text = ""
            pdf_file = io.BytesIO(file_content)
            reader = pypdf.PdfReader(pdf_file)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            return text, None
        except Exception as e:
            raise FileProcessingError(f"PDF 解析失败: {str(e)}")
            
    # [Case 5] Word 格式 (如果有 python-docx)
    if ext == '.docx':
        if docx is None:
            raise FileProcessingError("服务器缺少 python-docx 库，无法解析 .docx 文件")
        try:
            doc = docx.Document(io.BytesIO(file_content))
            text = "\n".join([para.text for para in doc.paragraphs])
            return text, None
        except Exception as e:
             raise FileProcessingError(f"Word 文档解析失败: {str(e)}")
            
    # [Case 6] 不支持的格式
    raise UnsupportedFileFormatError(f"不支持的文件格式: {ext}")
