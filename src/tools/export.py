import io


def generate_markdown(content: str) -> io.BytesIO:
    """生成 Markdown 文件流"""
    buffer = io.BytesIO()
    buffer.write(content.encode('utf-8'))
    buffer.seek(0)
    return buffer
