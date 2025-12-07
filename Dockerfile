# 运行多学科医疗诊断 Agent 的基础镜像
FROM python:3.11-slim

# 避免 Python 生成 .pyc，并让日志直接输出到控制台
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 预先安装依赖
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 默认命令：运行主诊断脚本
CMD ["python", "app.py"]
