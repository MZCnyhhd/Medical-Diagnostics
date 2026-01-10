"""
Graph RAG 混合检索模块：结合向量检索和知识图谱的增强检索
================================================================

Graph RAG（Graph-based Retrieval-Augmented Generation）是一种高级检索技术，
通过结合传统向量检索和知识图谱的结构化查询，提供更全面、更准确的知识检索。

核心优势：
1. 向量检索：捕获语义相似性，适合模糊查询
2. 图谱检索：捕获结构化关系，适合精确推理
3. 混合融合：取两者之长，提供更完整的上下文

工作流程：
==========

第一步：实体提取
- 使用 LLM 从查询文本中提取医学实体（症状、疾病、检查项目等）
- 识别实体类型，为后续图谱查询做准备

第二步：双通道检索
- 向量检索：使用传统 RAG 进行语义相似度搜索
- 图谱检索：基于提取的实体，在知识图谱中进行结构化查询

第三步：结果融合
- 合并两个通道的检索结果
- 去重、排序，构建增强的上下文

第四步：格式化输出
- 将检索结果格式化为 LLM 可以理解的文本
- 区分来源（向量/图谱），便于溯源

环境变量配置：
- ENABLE_GRAPH_RAG：是否启用 Graph RAG（默认 true）
- GRAPH_RAG_VECTOR_K：向量检索返回的文档数量（默认 3）
- GRAPH_RAG_GRAPH_K：图谱检索返回的实体数量（默认 5）

依赖模块：
- src.services.rag：向量检索功能
- src.services.kg：知识图谱服务
- src.services.llm：LLM 调用
"""

# ==================== 标准库导入 ====================
import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# ==================== 项目内部模块导入 ====================
from src.services.logging import log_info, log_warn, log_error, log_debug
from src.services.rag import retrieve_knowledge_snippets, _is_rag_enabled
from src.services.kg import get_kg, KnowledgeGraph
from src.services.llm import get_chat_model


# ==================== 数据结构定义 ====================

@dataclass
class ExtractedEntity:
    """
    从查询中提取的医学实体
    
    Attributes:
        name: 实体名称
        entity_type: 实体类型（symptom/disease/examination/treatment/department）
        confidence: 置信度（0-1）
    """
    name: str
    entity_type: str
    confidence: float = 1.0


@dataclass
class RetrievalResult:
    """
    检索结果项
    
    Attributes:
        content: 检索到的内容
        source: 来源（vector/graph）
        score: 相关性得分
        metadata: 额外的元数据
    """
    content: str
    source: str  # "vector" 或 "graph"
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphRAGResult:
    """
    Graph RAG 混合检索的完整结果
    
    Attributes:
        entities: 提取的实体列表
        vector_results: 向量检索结果
        graph_results: 图谱检索结果
        merged_context: 合并后的上下文文本
    """
    entities: List[ExtractedEntity]
    vector_results: List[RetrievalResult]
    graph_results: List[RetrievalResult]
    merged_context: str


# ==================== 配置函数 ====================

def _is_graph_rag_enabled() -> bool:
    """
    检查是否启用 Graph RAG 功能
    
    通过环境变量 ENABLE_GRAPH_RAG 控制：
    - "true"、"1"、"yes"、"on" 或不设置：启用
    - "false"、"0"、"no"、"off"：禁用
    
    Returns:
        bool: True 表示启用，False 表示禁用
    """
    flag = os.getenv("ENABLE_GRAPH_RAG", "true").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _get_vector_k() -> int:
    """获取向量检索返回的文档数量"""
    try:
        return int(os.getenv("GRAPH_RAG_VECTOR_K", "3"))
    except ValueError:
        return 3


def _get_graph_k() -> int:
    """获取图谱检索返回的实体数量"""
    try:
        return int(os.getenv("GRAPH_RAG_GRAPH_K", "5"))
    except ValueError:
        return 5


# ==================== 实体提取 ====================

def extract_medical_entities(query: str) -> List[ExtractedEntity]:
    """
    使用 LLM 从查询文本中提取医学实体
    
    这是 Graph RAG 的第一步：识别查询中的关键医学概念，
    为后续的图谱查询提供输入。
    
    支持的实体类型：
    - symptom：症状（如"头痛"、"发热"、"咳嗽"）
    - disease：疾病（如"糖尿病"、"高血压"）
    - examination：检查项目（如"血常规"、"CT"）
    - treatment：治疗方法（如"手术"、"药物治疗"）
    - department：科室（如"内科"、"心脏科"）
    
    Args:
        query: 用户查询文本或医疗报告摘要
    
    Returns:
        List[ExtractedEntity]: 提取的实体列表
    
    示例：
        >>> entities = extract_medical_entities("患者主诉头痛、发热三天")
        >>> for e in entities:
        ...     print(f"{e.name} ({e.entity_type})")
        头痛 (symptom)
        发热 (symptom)
    """
    if not query or not query.strip():
        return []
    
    try:
        llm = get_chat_model()
        
        # 构建实体提取的 Prompt
        prompt = f"""你是一位医学实体识别专家。请从以下医疗文本中提取关键的医学实体。

文本内容：
{query[:2000]}

请提取以下类型的实体，并以 JSON 格式返回：
{{
    "entities": [
        {{"name": "实体名称", "type": "实体类型", "confidence": 置信度}},
        ...
    ]
}}

实体类型说明：
- symptom：症状（如头痛、发热、咳嗽、胸闷、呼吸困难等）
- disease：疾病（如糖尿病、高血压、冠心病等）
- examination：检查项目（如血常规、CT、MRI、心电图等）
- treatment：治疗方法（如手术、药物治疗、化疗等）
- department：科室（如内科、外科、心脏科等）

置信度范围 0-1，表示你对这个实体识别的把握程度。

注意事项：
1. 只提取明确出现在文本中的实体
2. 症状是最重要的实体类型，请仔细识别
3. 如果没有找到任何实体，返回空数组
4. 只返回 JSON，不要返回其他文字"""

        # 调用 LLM
        response = llm.invoke(prompt)
        text = getattr(response, "content", str(response))
        
        # 解析 JSON 响应
        text = text.strip()
        # 移除可能的 markdown 代码块标记
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # 尝试找到 JSON 对象
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            json_str = text[start:end]
            data = json.loads(json_str)
            
            entities = []
            for item in data.get("entities", []):
                entity = ExtractedEntity(
                    name=item.get("name", "").strip(),
                    entity_type=item.get("type", "unknown"),
                    confidence=float(item.get("confidence", 1.0))
                )
                if entity.name:  # 只添加有效的实体
                    entities.append(entity)
            
            log_debug(f"[GraphRAG] 从查询中提取了 {len(entities)} 个实体")
            return entities
        else:
            log_warn("[GraphRAG] 无法从 LLM 响应中解析实体")
            return []
            
    except Exception as e:
        log_warn(f"[GraphRAG] 实体提取失败: {e}")
        return []


# ==================== 图谱检索 ====================

def retrieve_from_knowledge_graph(
    entities: List[ExtractedEntity],
    kg: KnowledgeGraph,
    limit: int = 5
) -> List[RetrievalResult]:
    """
    基于提取的实体，从知识图谱中检索相关知识
    
    检索策略：
    1. 对于症状实体：查找可能相关的疾病
    2. 对于疾病实体：查找完整的疾病信息（症状、检查、治疗等）
    3. 对于其他实体：进行关键词搜索
    
    Args:
        entities: 提取的实体列表
        kg: 知识图谱实例
        limit: 返回结果数量限制
    
    Returns:
        List[RetrievalResult]: 图谱检索结果列表
    """
    if not entities or not kg.driver:
        return []
    
    results = []
    
    # 收集所有症状实体
    symptoms = [e.name for e in entities if e.entity_type == "symptom"]
    
    # 收集所有疾病实体
    diseases = [e.name for e in entities if e.entity_type == "disease"]
    
    # ========== 策略1：根据症状查找相关疾病 ==========
    if symptoms:
        try:
            disease_matches = kg.find_diseases_by_symptoms(symptoms, limit=limit)
            for match in disease_matches:
                disease_name = match.get("disease_name", "")
                description = match.get("description", "")
                match_count = match.get("match_count", 0)
                matched_symptoms = match.get("matched_symptoms", [])
                
                if disease_name:
                    content = f"【疾病】{disease_name}\n"
                    if description:
                        content += f"描述：{description}\n"
                    content += f"匹配症状：{', '.join(matched_symptoms)}（共 {match_count} 个匹配）"
                    
                    results.append(RetrievalResult(
                        content=content,
                        source="graph",
                        score=match_count / len(symptoms) if symptoms else 1.0,
                        metadata={
                            "type": "disease_by_symptom",
                            "disease_name": disease_name,
                            "matched_symptoms": matched_symptoms
                        }
                    ))
        except Exception as e:
            log_warn(f"[GraphRAG] 症状-疾病查询失败: {e}")
    
    # ========== 策略2：获取疾病的完整信息 ==========
    for disease_name in diseases[:limit]:
        try:
            disease_info = kg.get_disease_info(disease_name)
            if disease_info:
                content = f"【疾病详情】{disease_info.get('name', disease_name)}\n"
                
                if disease_info.get("description"):
                    content += f"描述：{disease_info['description']}\n"
                
                if disease_info.get("symptoms"):
                    content += f"常见症状：{', '.join(disease_info['symptoms'])}\n"
                
                if disease_info.get("examinations"):
                    content += f"建议检查：{', '.join(disease_info['examinations'])}\n"
                
                if disease_info.get("treatments"):
                    content += f"治疗方法：{', '.join(disease_info['treatments'])}\n"
                
                if disease_info.get("departments"):
                    content += f"就诊科室：{', '.join(disease_info['departments'])}"
                
                results.append(RetrievalResult(
                    content=content,
                    source="graph",
                    score=1.0,
                    metadata={
                        "type": "disease_info",
                        "disease_name": disease_name,
                        "full_info": disease_info
                    }
                ))
                
                # 查找相关疾病（鉴别诊断）
                related = kg.get_related_diseases(disease_name, limit=3)
                if related:
                    related_names = [r.get("disease_name") for r in related if r.get("disease_name")]
                    if related_names:
                        results.append(RetrievalResult(
                            content=f"【鉴别诊断】与 {disease_name} 相关的疾病：{', '.join(related_names)}",
                            source="graph",
                            score=0.8,
                            metadata={
                                "type": "related_diseases",
                                "base_disease": disease_name,
                                "related": related_names
                            }
                        ))
        except Exception as e:
            log_warn(f"[GraphRAG] 疾病信息查询失败 ({disease_name}): {e}")
    
    # ========== 策略3：其他实体的关键词搜索 ==========
    other_entities = [e for e in entities if e.entity_type not in ("symptom", "disease")]
    for entity in other_entities[:3]:
        try:
            search_results = kg.search_entities(entity.name)
            for sr in search_results[:2]:
                entity_type = sr.get("type", "未知")
                entity_name = sr.get("name", "")
                description = sr.get("description", "")
                
                if entity_name:
                    content = f"【{entity_type}】{entity_name}"
                    if description:
                        content += f"\n{description}"
                    
                    results.append(RetrievalResult(
                        content=content,
                        source="graph",
                        score=0.7,
                        metadata={
                            "type": "entity_search",
                            "entity_type": entity_type,
                            "entity_name": entity_name
                        }
                    ))
        except Exception as e:
            log_warn(f"[GraphRAG] 实体搜索失败 ({entity.name}): {e}")
    
    log_debug(f"[GraphRAG] 图谱检索返回 {len(results)} 条结果")
    return results


# ==================== 向量检索封装 ====================

def retrieve_from_vector_store(query: str, k: int = 3) -> List[RetrievalResult]:
    """
    从向量数据库检索相关文档（封装原有 RAG 功能）
    
    Args:
        query: 查询文本
        k: 返回文档数量
    
    Returns:
        List[RetrievalResult]: 向量检索结果列表
    """
    # 调用原有的 RAG 检索函数
    raw_result = retrieve_knowledge_snippets(query, k=k)
    
    if not raw_result:
        return []
    
    results = []
    # 解析原有格式：[参考1] 内容...
    for line in raw_result.split('\n'):
        if line.strip():
            # 移除 [参考X] 前缀
            content = re.sub(r'^\[参考\d+\]\s*', '', line.strip())
            if content:
                results.append(RetrievalResult(
                    content=content,
                    source="vector",
                    score=1.0,
                    metadata={"type": "vector_search"}
                ))
    
    log_debug(f"[GraphRAG] 向量检索返回 {len(results)} 条结果")
    return results


# ==================== 结果融合 ====================

def merge_retrieval_results(
    vector_results: List[RetrievalResult],
    graph_results: List[RetrievalResult],
    max_results: int = 10
) -> List[RetrievalResult]:
    """
    融合向量检索和图谱检索的结果
    
    融合策略：
    1. 交替排列：先图谱后向量，确保结构化知识优先
    2. 去重：基于内容相似性去除重复项
    3. 截断：限制总结果数量
    
    Args:
        vector_results: 向量检索结果
        graph_results: 图谱检索结果
        max_results: 最大结果数量
    
    Returns:
        List[RetrievalResult]: 融合后的结果列表
    """
    merged = []
    seen_contents = set()
    
    # 交替添加结果，图谱优先
    max_len = max(len(graph_results), len(vector_results))
    for i in range(max_len):
        # 先添加图谱结果
        if i < len(graph_results):
            result = graph_results[i]
            content_key = result.content[:100]  # 用前100字符作为去重键
            if content_key not in seen_contents:
                merged.append(result)
                seen_contents.add(content_key)
        
        # 再添加向量结果
        if i < len(vector_results):
            result = vector_results[i]
            content_key = result.content[:100]
            if content_key not in seen_contents:
                merged.append(result)
                seen_contents.add(content_key)
        
        # 达到最大数量时停止
        if len(merged) >= max_results:
            break
    
    return merged[:max_results]


def format_retrieval_results(results: List[RetrievalResult]) -> str:
    """
    将检索结果格式化为 LLM 可以理解的文本
    
    Args:
        results: 检索结果列表
    
    Returns:
        str: 格式化的文本
    """
    if not results:
        return ""
    
    formatted_parts = []
    
    # 分类整理结果
    graph_results = [r for r in results if r.source == "graph"]
    vector_results = [r for r in results if r.source == "vector"]
    
    # 格式化图谱知识
    if graph_results:
        formatted_parts.append("=== 知识图谱检索结果 ===")
        for i, result in enumerate(graph_results, 1):
            formatted_parts.append(f"[图谱{i}] {result.content}")
    
    # 格式化向量知识
    if vector_results:
        formatted_parts.append("\n=== 向量检索结果 ===")
        for i, result in enumerate(vector_results, 1):
            formatted_parts.append(f"[文档{i}] {result.content}")
    
    return "\n".join(formatted_parts)


# ==================== 主入口函数 ====================

def retrieve_hybrid_knowledge(query: str) -> GraphRAGResult:
    """
    Graph RAG 混合检索的主入口函数
    
    执行完整的混合检索流程：
    1. 从查询中提取医学实体
    2. 并行执行向量检索和图谱检索
    3. 融合检索结果
    4. 格式化输出
    
    Args:
        query: 查询文本（通常是患者的医疗报告或症状描述）
    
    Returns:
        GraphRAGResult: 完整的检索结果，包含：
            - entities: 提取的实体
            - vector_results: 向量检索结果
            - graph_results: 图谱检索结果
            - merged_context: 合并后的上下文文本
    
    使用示例：
        >>> result = retrieve_hybrid_knowledge("患者主诉多饮多尿，体重下降")
        >>> print(result.merged_context)
        === 知识图谱检索结果 ===
        [图谱1] 【疾病】糖尿病...
        === 向量检索结果 ===
        [文档1] 糖尿病是一种以高血糖为特征的代谢性疾病...
    """
    log_info("[GraphRAG] 开始混合检索...")
    
    # 初始化结果
    entities = []
    vector_results = []
    graph_results = []
    
    # ========== 第一步：实体提取 ==========
    if _is_graph_rag_enabled():
        entities = extract_medical_entities(query)
        log_info(f"[GraphRAG] 提取到 {len(entities)} 个医学实体")
        for e in entities:
            log_debug(f"  - {e.name} ({e.entity_type}, 置信度: {e.confidence:.2f})")
    
    # ========== 第二步：向量检索 ==========
    if _is_rag_enabled():
        vector_k = _get_vector_k()
        vector_results = retrieve_from_vector_store(query, k=vector_k)
        log_info(f"[GraphRAG] 向量检索返回 {len(vector_results)} 条结果")
    
    # ========== 第三步：图谱检索 ==========
    if _is_graph_rag_enabled() and entities:
        kg = get_kg()
        if kg.driver:
            graph_k = _get_graph_k()
            graph_results = retrieve_from_knowledge_graph(entities, kg, limit=graph_k)
            log_info(f"[GraphRAG] 图谱检索返回 {len(graph_results)} 条结果")
        else:
            log_warn("[GraphRAG] 知识图谱不可用，跳过图谱检索")
    
    # ========== 第四步：结果融合 ==========
    merged_results = merge_retrieval_results(vector_results, graph_results)
    merged_context = format_retrieval_results(merged_results)
    
    log_info(f"[GraphRAG] 混合检索完成，共 {len(merged_results)} 条结果")
    
    return GraphRAGResult(
        entities=entities,
        vector_results=vector_results,
        graph_results=graph_results,
        merged_context=merged_context
    )


def retrieve_hybrid_knowledge_snippets(query: str, k: int = 3) -> str:
    """
    Graph RAG 混合检索的简化接口（兼容原有 RAG 接口）
    
    这是一个便捷函数，可以直接替换原有的 retrieve_knowledge_snippets 函数。
    返回格式与原函数兼容，便于无缝升级。
    
    Args:
        query: 查询文本
        k: 用于控制返回结果的数量参考（实际数量可能因融合策略而变化）
    
    Returns:
        str: 格式化的知识片段文本
            - 包含图谱检索和向量检索的融合结果
            - 如果 Graph RAG 不可用，自动降级为纯向量检索
    
    使用示例：
        >>> knowledge = retrieve_hybrid_knowledge_snippets("患者主诉多饮多尿")
        >>> if knowledge:
        ...     prompt = f"参考知识：\\n{knowledge}\\n\\n请分析以下病例..."
    """
    # 检查 Graph RAG 是否启用
    if not _is_graph_rag_enabled():
        # Graph RAG 禁用，降级为纯向量检索
        log_debug("[GraphRAG] Graph RAG 已禁用，使用纯向量检索")
        return retrieve_knowledge_snippets(query, k=k)
    
    try:
        # 执行混合检索
        result = retrieve_hybrid_knowledge(query)
        return result.merged_context
    except Exception as e:
        # 混合检索失败，降级为纯向量检索
        log_warn(f"[GraphRAG] 混合检索失败，降级为向量检索: {e}")
        return retrieve_knowledge_snippets(query, k=k)

