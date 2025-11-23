import pytest
import os
import sys

# 将项目根目录加入 python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def mock_medical_report():
    return """
    患者：张三
    年龄：45
    主诉：持续性头痛一周，伴有恶心。
    既往史：高血压病史5年。
    """
