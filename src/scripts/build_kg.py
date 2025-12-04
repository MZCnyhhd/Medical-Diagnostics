"""
知识图谱构建脚本：从医学知识库 Markdown 文档中抽取结构化知识并构建 Neo4j 知识图谱

使用方法：
1. 确保 Neo4j 数据库已启动并配置好连接信息（环境变量或 config/apikey.env）
2. 运行脚本：python src/scripts/build_kg.py

脚本会：
- 读取 data/knowledge_base/ 下的所有 .md 文件
- 使用 LLM 抽取结构化知识（疾病、症状、检查、治疗、科室）
- 将抽取的知识写入 Neo4j 知识图谱
"""

import os
import sys
from pathlib import Path
import re
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from src.services.kg import get_kg
from src.services.llm import get_chat_model
from src.services.logging import log_info, log_warn, log_error
from src.core.config import APIKEY_ENV_PATH

# 加载环境变量
load_dotenv(dotenv_path=APIKEY_ENV_PATH, override=True)


def extract_disease_name_from_file(file_path: Path) -> str:
    """
    从文件名提取疾病名称（去掉 .md 后缀）
    
    Args:
        file_path: 文件路径
    
    Returns:
        疾病名称
    """
    return file_path.stem


def extract_structured_knowledge(content: str, disease_name: str) -> Dict:
    """
    使用 LLM 从医学文档中抽取结构化知识
    
    Args:
        content: 医学文档内容
        disease_name: 疾病名称
    
    Returns:
        结构化知识字典，包含症状、检查、治疗、科室等信息
    """
    llm = get_chat_model()
    
    prompt = f"""
你是一位医学知识抽取专家。请从以下医学文档中提取结构化知识。

疾病名称：{disease_name}

文档内容：
{content[:3000]}  # 限制长度避免 token 过多

请提取以下信息，并以 JSON 格式返回：
{{
    "symptoms": ["症状1", "症状2", ...],  # 该疾病的常见症状
    "examinations": ["检查1", "检查2", ...],  # 该疾病需要的检查项目
    "treatments": ["治疗1", "治疗2", ...],  # 该疾病的治疗方法
    "departments": ["科室1", "科室2", ...],  # 该疾病所属的科室
    "description": "疾病的简要描述"
}}

只返回 JSON，不要返回其他文字。
"""
    
    try:
        response = llm.invoke(prompt)
        text = getattr(response, "content", str(response))
        
        # 清理文本，提取 JSON
        text = text.strip()
        # 移除可能的 markdown 代码块标记
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # 尝试找到 JSON 对象
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            return json.loads(json_str)
        else:
            log_warn(f"[KG] 无法从 LLM 响应中提取 JSON: {text[:200]}")
            return {}
    except Exception as e:
        log_error(f"[KG] 知识抽取失败 ({disease_name}): {e}")
        return {}


def map_department_name(department: str) -> str:
    """
    将科室名称映射到标准格式
    
    Args:
        department: 原始科室名称
    
    Returns:
        标准化的科室名称
    """
    # 科室名称映射表
    mapping = {
        "心脏科": "心脏科医生",
        "心内科": "心脏科医生",
        "心血管科": "心脏科医生",
        "消化科": "消化科医生",
        "消化内科": "消化科医生",
        "心理科": "心理医生",
        "精神科": "精神科医生",
        "神经科": "神经科医生",
        "神经内科": "神经科医生",
        "内分泌科": "内分泌科医生",
        "免疫科": "免疫科医生",
        "皮肤科": "皮肤科医生",
        "肿瘤科": "肿瘤科医生",
        "血液科": "血液科医生",
        "肾脏科": "肾脏科医生",
        "肾内科": "肾脏科医生",
        "风湿科": "风湿科医生",
        "肺科": "肺科医生",
        "呼吸科": "肺科医生",
    }
    
    # 尝试精确匹配
    if department in mapping:
        return mapping[department]
    
    # 尝试部分匹配
    for key, value in mapping.items():
        if key in department or department in key:
            return value
    
    # 如果都不匹配，返回原名称（去掉"医生"后缀）
    return department.replace("医生", "").replace("科", "科医生")


def build_knowledge_graph(knowledge_base_dir: Path = None):
    """
    构建知识图谱
    
    Args:
        knowledge_base_dir: 知识库目录路径，默认 data/knowledge_base
    """
    if knowledge_base_dir is None:
        knowledge_base_dir = project_root / "data" / "knowledge_base"
    
    if not knowledge_base_dir.exists():
        log_error(f"[KG] 知识库目录不存在: {knowledge_base_dir}")
        return
    
    kg = get_kg()
    if not kg.driver:
        log_error("[KG] Neo4j 连接失败，无法构建知识图谱")
        return
    
    log_info(f"[KG] 开始构建知识图谱，知识库目录: {knowledge_base_dir}")
    
    # 获取所有 Markdown 文件
    md_files = list(knowledge_base_dir.glob("*.md"))
    log_info(f"[KG] 找到 {len(md_files)} 个医学文档")
    
    success_count = 0
    error_count = 0
    
    for i, file_path in enumerate(md_files, 1):
        disease_name = extract_disease_name_from_file(file_path)
        log_info(f"[KG] [{i}/{len(md_files)}] 处理: {disease_name}")
        
        try:
            # 读取文件内容
            content = file_path.read_text(encoding="utf-8")
            
            # 使用 LLM 抽取结构化知识
            knowledge = extract_structured_knowledge(content, disease_name)
            
            if not knowledge:
                log_warn(f"[KG] 未能从 {disease_name} 中抽取到知识")
                error_count += 1
                continue
            
            # 创建疾病实体
            description = knowledge.get("description", "")
            kg.create_disease(disease_name, description)
            
            # 创建症状实体并建立关系
            symptoms = knowledge.get("symptoms", [])
            for symptom in symptoms:
                if symptom and symptom.strip():
                    kg.create_symptom(symptom.strip())
                    kg.link_disease_symptom(disease_name, symptom.strip())
            
            # 创建检查实体并建立关系
            examinations = knowledge.get("examinations", [])
            for exam in examinations:
                if exam and exam.strip():
                    kg.create_examination(exam.strip())
                    kg.link_disease_examination(disease_name, exam.strip())
            
            # 创建治疗实体并建立关系
            treatments = knowledge.get("treatments", [])
            for treatment in treatments:
                if treatment and treatment.strip():
                    kg.create_treatment(treatment.strip())
                    kg.link_disease_treatment(disease_name, treatment.strip())
            
            # 创建科室实体并建立关系
            departments = knowledge.get("departments", [])
            for dept in departments:
                if dept and dept.strip():
                    dept_standard = map_department_name(dept.strip())
                    kg.create_department(dept_standard)
                    kg.link_disease_department(disease_name, dept_standard)
            
            success_count += 1
            log_info(f"[KG] ✓ {disease_name} 知识已导入")
            
        except Exception as e:
            log_error(f"[KG] 处理 {disease_name} 时出错: {e}")
            error_count += 1
    
    # 输出统计信息
    log_info(f"[KG] 知识图谱构建完成！成功: {success_count}, 失败: {error_count}")
    
    # 显示图谱统计
    stats = kg.get_statistics()
    if stats:
        log_info(f"[KG] 图谱统计:")
        log_info(f"  - 疾病: {stats.get('disease_count', 0)}")
        log_info(f"  - 症状: {stats.get('symptom_count', 0)}")
        log_info(f"  - 检查: {stats.get('exam_count', 0)}")
        log_info(f"  - 治疗: {stats.get('treatment_count', 0)}")
        log_info(f"  - 科室: {stats.get('dept_count', 0)}")
        log_info(f"  - 关系: {stats.get('relation_count', 0)}")
    
    kg.close()


if __name__ == "__main__":
    build_knowledge_graph()

