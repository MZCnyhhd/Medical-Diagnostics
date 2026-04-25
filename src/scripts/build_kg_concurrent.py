"""
模块名称: Knowledge Graph Builder - 并发版 (图谱构建脚本)
功能描述:

    离线脚本，用于从原始医疗文本 (Markdown) 构建 Neo4j 知识图谱。
    支持并发 LLM 提取 + 顺序 Neo4j 导入，性能提升 5-7 倍。

设计理念:

    1.  **并发提取**: 使用 ThreadPoolExecutor 并发调用 LLM，加速知识提取。
    2.  **顺序导入**: 提取完成后顺序导入到 Neo4j，避免写冲突。
    3.  **批量操作**: 使用 Cypher 事务合并单个疾病的所有操作。

"""

import os
import sys
from pathlib import Path
import re
import json
import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# [环境设置 | Environment Setup] ========================================================================================
# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# [第三方库 | Third-party Libraries] ====================================================================================
from dotenv import load_dotenv

# [内部模块 | Internal Modules] =========================================================================================
from src.services.kg import get_kg
from src.services.llm import get_chat_model
from src.services.logging import log_info, log_warn, log_error
from src.core.settings import APIKEY_ENV_PATH

# 加载环境变量
try:
    load_dotenv(dotenv_path=APIKEY_ENV_PATH, override=True, encoding="utf-8")
except UnicodeDecodeError:
    load_dotenv(dotenv_path=APIKEY_ENV_PATH, override=True, encoding="gbk")


# [定义函数] ############################################################################################################
# [脚本-提取疾病名称] ======================================================================================================
def extract_disease_name_from_file(file_path: Path) -> str:
    """
    从文件名提取疾病名称（去掉 .md 后缀）
    
    Args:
        file_path: 文件路径
    
    Returns:
        疾病名称
    """
    return file_path.stem


# [脚本-抽取结构化知识] ====================================================================================================
def extract_structured_knowledge(content: str, disease_name: str) -> Dict:
    """
    使用 LLM 从医学文档中抽取结构化知识
    
    Args:
        content: 医学文档内容
        disease_name: 疾病名称
    
    Returns:
        结构化知识字典，包含症状、检查、治疗、科室等信息
    """
    # [step1] 获取 LLM 实例
    llm = get_chat_model()
    
    # [step2] 构建提示词
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
    
    # [step3] 调用 LLM
    try:
        response = llm.invoke(prompt)
        
        # 如果是 LangChain 的 AIMessage 对象，需要取 content 属性
        if hasattr(response, 'content'):
            response = response.content
        
        # [step4] 解析 JSON 响应
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            knowledge = json.loads(json_match.group())
            return knowledge
    except Exception as e:
        log_error(f"[KG] LLM 调用失败: {e}")
    
    return None


# [脚本-科室映射] =========================================================================================================
def map_department_name(department: str) -> str:
    """
    将不规范的科室名称映射到标准名称
    
    Args:
        department: 科室名称
    
    Returns:
        标准科室名称
    """
    # [step1] 定义科室映射表
    mapping = {
        "内科": "内科医生",
        "外科": "外科医生",
        "妇产科": "妇产科医生",
        "儿科": "儿科医生",
        "眼科": "眼科医生",
        "耳鼻喉科": "耳鼻喉科医生",
        "皮肤科": "皮肤科医生",
        "精神心理科": "精神心理科医生",
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
    
    # [step2] 尝试精确匹配
    if department in mapping:
        return mapping[department]
    
    # [step3] 尝试部分匹配
    for key, value in mapping.items():
        if key in department or department in key:
            return value
    
    # [step4] 默认处理：如果都不匹配，返回原名称（去掉"医生"后缀并尝试规范化）
    return department.replace("医生", "").replace("科", "科医生")


# [脚本-单个疾病的知识提取] ==============================================================================================
def extract_disease_knowledge(file_path: Path) -> Tuple[str, Dict]:
    """
    提取单个疾病的知识（用于并发执行）
    返回 (disease_name, knowledge_dict)
    """
    disease_name = extract_disease_name_from_file(file_path)
    try:
        # 读取文件内容
        content = file_path.read_text(encoding="utf-8")
        
        # 使用 LLM 抽取结构化知识
        knowledge = extract_structured_knowledge(content, disease_name)
        
        if not knowledge:
            log_warn(f"[KG] 未能从 {disease_name} 中抽取到知识")
            return disease_name, None
        
        return disease_name, knowledge
    except Exception as e:
        log_error(f"[KG] 提取 {disease_name} 知识失败: {e}")
        return disease_name, None


# [脚本-构建知识图谱（并发版）] ===========================================================================================
def build_knowledge_graph_concurrent(knowledge_base_dir: Path = None, max_workers: int = 5):
    """
    构建知识图谱（并发版本）
    
    Args:
        knowledge_base_dir: 知识库目录路径，默认 data/knowledge_base
        max_workers: 并发线程数（默认 5），用于 LLM 提取
    """
    # [step1] 确定知识库目录
    if knowledge_base_dir is None:
        knowledge_base_dir = project_root / "data" / "knowledge_base"
    
    if not knowledge_base_dir.exists():
        log_error(f"[KG] 知识库目录不存在: {knowledge_base_dir}")
        return
    
    # [step2] 初始化图谱
    kg = get_kg()
    if not kg.driver:
        log_error("[KG] Neo4j 连接失败，无法构建知识图谱")
        return
    
    log_info(f"[KG] 开始构建知识图谱，知识库目录: {knowledge_base_dir}")
    
    # [step3] 获取所有 Markdown 文件
    md_files = list(knowledge_base_dir.glob("*.md"))
    log_info(f"[KG] 找到 {len(md_files)} 个医学文档，开启 {max_workers} 个并发线程用于 LLM 提取")
    
    success_count = 0
    error_count = 0
    total_start = time.time()
    
    # [step4] 第一阶段：并发提取所有疾病的知识
    log_info(f"[KG] ========== 第一阶段：并发 LLM 提取 ==========")
    knowledge_data = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(extract_disease_knowledge, f): f for f in md_files}
        
        for i, future in enumerate(as_completed(future_to_file), 1):
            try:
                disease_name, knowledge = future.result()
                if knowledge:
                    knowledge_data[disease_name] = knowledge
                    progress = f"[{i}/{len(md_files)}]"
                    log_info(f"[KG] {progress} ✓ {disease_name} 知识提取完成")
                else:
                    error_count += 1
            except Exception as e:
                log_error(f"[KG] 并发任务异常: {e}")
                error_count += 1
    
    extract_time = time.time() - total_start
    log_info(f"[KG] 第一阶段完成，共 {len(knowledge_data)} 个疾病成功提取，耗时 {extract_time:.1f}s")
    
    # [step5] 第二阶段：顺序导入到 Neo4j（避免并发写入冲突）
    log_info(f"[KG] ========== 第二阶段：顺序导入 Neo4j ==========")
    log_info(f"[KG] 开始导入到 Neo4j （共 {len(knowledge_data)} 个疾病）")
    
    import_start = time.time()
    for i, (disease_name, knowledge) in enumerate(knowledge_data.items(), 1):
        try:
            # 批量导入所有实体和关系
            kg.import_disease_batch(
                disease_name=disease_name,
                description=knowledge.get("description", ""),
                symptoms=knowledge.get("symptoms", []),
                examinations=knowledge.get("examinations", []),
                treatments=knowledge.get("treatments", []),
                departments=knowledge.get("departments", [])
            )
            
            success_count += 1
            progress = f"[{i}/{len(knowledge_data)}]"
            log_info(f"[KG] {progress} ✓ {disease_name} 已导入到 Neo4j")
            
        except Exception as e:
            log_error(f"[KG] [{i}/{len(knowledge_data)}] ✗ 导入 {disease_name} 时出错: {e}")
            error_count += 1
    
    import_time = time.time() - import_start
    log_info(f"[KG] 第二阶段完成，共 {success_count} 个疾病成功导入，耗时 {import_time:.1f}s")
    
    # [step6] 输出统计信息
    total_time = time.time() - total_start
    avg_time = total_time / len(md_files) if md_files else 0
    log_info(f"[KG] ========== 构建完成 ==========")
    log_info(f"[KG] 总耗时: {total_time:.1f}s ({total_time/60:.2f} 分钟)")
    log_info(f"[KG] - 提取阶段: {extract_time:.1f}s")
    log_info(f"[KG] - 导入阶段: {import_time:.1f}s")
    log_info(f"[KG] - 平均每个疾病: {avg_time:.2f}s")
    log_info(f"[KG] 成功: {success_count}, 失败: {error_count}")
    
    # [step7] 显示图谱统计
    stats = kg.get_statistics()
    if stats:
        log_info(f"[KG] ========== 图谱统计 ==========")
        log_info(f"  - 疾病: {stats.get('disease_count', 0)}")
        log_info(f"  - 症状: {stats.get('symptom_count', 0)}")
        log_info(f"  - 检查: {stats.get('exam_count', 0)}")
        log_info(f"  - 治疗: {stats.get('treatment_count', 0)}")
        log_info(f"  - 科室: {stats.get('dept_count', 0)}")
        log_info(f"  - 关系: {stats.get('relation_count', 0)}")
    
    kg.close()


if __name__ == "__main__":
    # 可自定义并发数，默认 5
    build_knowledge_graph_concurrent(max_workers=5)
