# [导入模块] ############################################################################################################
# [标准库 | Standard Libraries] =========================================================================================
import os                                                                  # 操作系统接口：环境变量
import json                                                                # JSON 解析
import re                                                                  # 正则表达式：文本清洗
from typing import List, Dict, Any                                         # 类型提示
from dataclasses import dataclass, field                                   # 数据类：结构化数据定义
# [内部模块 | Internal Modules] =========================================================================================
from src.services.logging import log_info, log_warn, log_debug             # 统一日志服务
from src.services.rag import retrieve_knowledge_snippets, _is_rag_enabled  # 向量检索
from src.services.kg import get_kg, KnowledgeGraph                         # 知识图谱服务
from src.services.llm import get_chat_model                                # 模型工厂
# [创建类] ##############################################################################################################
# [装饰器-内部-提取的实体] =================================================================================================
@dataclass
class ExtractedEntity:
    """提取的医学实体"""
    name: str
    entity_type: str
    confidence: float = 1.0
# [装饰器-内部-检索结果] ==================================================================================================
@dataclass
class RetrievalResult:
    """检索结果"""
    content: str
    source: str
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
# [装饰器-内部-GraphRAG结果] ==============================================================================================
@dataclass
class GraphRAGResult:
    """Graph RAG 混合检索结果"""
    entities: List[ExtractedEntity]
    vector_results: List[RetrievalResult]
    graph_results: List[RetrievalResult]
    merged_context: str
# [定义函数] ############################################################################################################
# [内部- GraphRAG 是否启用] ==============================================================================================
def _is_graph_rag_enabled() -> bool:
    """检测 Graph RAG 是否启用"""
    flag = os.getenv("ENABLE_GRAPH_RAG", "true").strip().lower()
    return flag not in {"0", "false", "no", "off"}
# [内部-获取向量 k ] =====================================================================================================
def _get_vector_k() -> int:
    """获取向量检索数量配置"""
    try:
        return int(os.getenv("GRAPH_RAG_VECTOR_K", "3"))
    except ValueError:
        return 3
# [内部-获取 Graph k ] ==================================================================================================
def _get_graph_k() -> int:
    """获取图谱检索数量配置"""
    try:
        return int(os.getenv("GRAPH_RAG_GRAPH_K", "5"))
    except ValueError:
        return 5
# [内部-解析实体 JSON ] ==================================================================================================
def _parse_entity_json(text: str) -> list[dict]:
    """
    从 LLM 响应中解析实体 JSON。
    :param text: LLM 原始响应
    :return: 实体字典列表
    """
    # [step1] 清洗文本
    text = text.strip()
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    # [step2] 提取 JSON 部分
    start = text.find('{')
    end = text.rfind('}') + 1
    if start < 0 or end <= start:
        return []
    # [step3] 解析 JSON
    try:
        data = json.loads(text[start:end])
        return data.get("entities", [])
    except:
        return []
# [外部-提取医疗实体] =====================================================================================================
def extract_medical_entities(query: str) -> List[ExtractedEntity]:
    """
    使用 LLM 从医疗文本中提取关键实体。
    :param query: 医疗报告文本
    :return: 提取的实体列表
    """
    # [step1] 卫语句：空查询
    if not query or not query.strip():
        return []
    # [step2] 构建提示词
    prompt = f"""你是一位医学实体识别专家。请从以下医疗文本中提取关键的医学实体。
文本内容：
{query[:2000]}
请提取以下类型的实体，并以 JSON 格式返回：
{{"entities": [{{"name": "实体名称", "type": "实体类型", "confidence": 置信度}}, ...]}}
实体类型：symptom(症状)、disease(疾病)、examination(检查)、treatment(治疗)、department(科室)
置信度范围 0-1。只返回 JSON，不要返回其他文字。"""
    # [step3] 调用 LLM
    try:
        llm = get_chat_model()
        response = llm.invoke(prompt)
        text = getattr(response, "content", str(response))
    except Exception as e:
        log_warn(f"[GraphRAG] 实体提取失败: {e}")
        return []
    # [step4] 解析结果
    raw_entities = _parse_entity_json(text)
    entities = []
    for item in raw_entities:
        name = item.get("name", "").strip()
        if name:
            entities.append(ExtractedEntity(
                name=name,
                entity_type=item.get("type", "unknown"),
                confidence=float(item.get("confidence", 1.0))
            ))
    log_debug(f"[GraphRAG] 从查询中提取了 {len(entities)} 个实体")
    return entities
# [内部-症状查疾病] =======================================================================================================
def _query_diseases_by_symptoms(kg: KnowledgeGraph, symptoms: list[str], limit: int) -> List[RetrievalResult]:
    """根据症状查询可能的疾病"""
    results = []
    # [step1] 调用知识图谱查询症状关联的疾病
    try:
        disease_matches = kg.find_diseases_by_symptoms(symptoms, limit=limit)
    except Exception as e:
        log_warn(f"[GraphRAG] 症状-疾病查询失败: {e}")
        return results
    # [step2] 遍历匹配结果，构建 RetrievalResult
    for match in disease_matches:
        disease_name = match.get("disease_name", "")
        if not disease_name:
            continue
        # [step3] 提取字段并格式化内容
        description = match.get("description", "")
        match_count = match.get("match_count", 0)
        matched_symptoms = match.get("matched_symptoms", [])
        content = f"【疾病】{disease_name}\n"
        if description:
            content += f"描述：{description}\n"
        content += f"匹配症状：{', '.join(matched_symptoms)}（共 {match_count} 个匹配）"
        # [step4] 计算匹配得分并添加到结果
        results.append(RetrievalResult(
            content=content, source="graph",
            score=match_count / len(symptoms) if symptoms else 1.0,
            metadata={"type": "disease_by_symptom", "disease_name": disease_name, "matched_symptoms": matched_symptoms}
        ))
    return results
# [内部-疾病详情查询] =====================================================================================================
def _query_disease_info(kg: KnowledgeGraph, disease_name: str) -> List[RetrievalResult]:
    """查询单个疾病的详细信息"""
    results = []
    # [step1] 调用知识图谱获取疾病详情
    try:
        disease_info = kg.get_disease_info(disease_name)
    except Exception as e:
        log_warn(f"[GraphRAG] 疾病信息查询失败 ({disease_name}): {e}")
        return results
    # [step2] 卫语句：无结果直接返回
    if not disease_info:
        return results
    # [step3] 构建疾病详情内容（按字段拼接）
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
    # [step4] 添加疾病详情到结果
    results.append(RetrievalResult(
        content=content, source="graph", score=1.0,
        metadata={"type": "disease_info", "disease_name": disease_name, "full_info": disease_info}
    ))
    # [step5] 查询并添加相关疾病（鉴别诊断）
    try:
        related = kg.get_related_diseases(disease_name, limit=3)
        related_names = [r.get("disease_name") for r in related if r.get("disease_name")]
        if related_names:
            results.append(RetrievalResult(
                content=f"【鉴别诊断】与 {disease_name} 相关的疾病：{', '.join(related_names)}",
                source="graph", score=0.8,
                metadata={"type": "related_diseases", "base_disease": disease_name, "related": related_names}
            ))
    except Exception as e:
        log_warn(f"[GraphRAG] 相关疾病查询失败: {e}")
    return results
# [内部-搜索单个实体] =====================================================================================================
def _search_single_entity(kg: KnowledgeGraph, entity: ExtractedEntity) -> List[RetrievalResult]:
    """搜索单个非症状/疾病实体"""
    results = []
    try:
        search_results = kg.search_entities(entity.name)
    except Exception as e:
        log_warn(f"[GraphRAG] 实体搜索失败 ({entity.name}): {e}")
        return results
    for sr in search_results[:2]:
        entity_name = sr.get("name", "")
        if not entity_name:
            continue
        entity_type = sr.get("type", "未知")
        description = sr.get("description", "")
        content = f"【{entity_type}】{entity_name}"
        if description:
            content += f"\n{description}"
        results.append(RetrievalResult(
            content=content, source="graph", score=0.7,
            metadata={"type": "entity_search", "entity_type": entity_type, "entity_name": entity_name}
        ))
    return results
# [外部-知识图谱检索] =====================================================================================================
def retrieve_from_knowledge_graph(
        entities: List[ExtractedEntity],
        kg: KnowledgeGraph,
        limit: int = 5) -> List[RetrievalResult]:
    """
    从知识图谱检索相关知识。
    :param entities: 提取的实体列表
    :param kg: 知识图谱实例
    :param limit: 每类查询的结果限制
    :return: 检索结果列表
    """
    # [step1] 卫语句：无实体或图谱不可用
    if not entities or not kg.driver:
        return []
    results = []
    # [step2] 按类型分组实体
    symptoms = [e.name for e in entities if e.entity_type == "symptom"]
    diseases = [e.name for e in entities if e.entity_type == "disease"]
    other_entities = [e for e in entities if e.entity_type not in ("symptom", "disease")]
    # [step3] 症状查疾病
    if symptoms:
        results.extend(_query_diseases_by_symptoms(kg, symptoms, limit))
    # [step4] 疾病详情查询
    for disease_name in diseases[:limit]:
        results.extend(_query_disease_info(kg, disease_name))
    # [step5] 其他实体搜索（委托辅助函数）
    for entity in other_entities[:3]:
        results.extend(_search_single_entity(kg, entity))
    log_debug(f"[GraphRAG] 图谱检索返回 {len(results)} 条结果")
    return results
# [外部-向量检索] ========================================================================================================
def retrieve_from_vector_store(query: str, k: int = 3) -> List[RetrievalResult]:
    """
    从向量数据库检索相关知识。
    :param query: 查询文本
    :param k: 返回结果数量
    :return: 检索结果列表
    """
    # [step1] 调用底层向量检索接口
    raw_result = retrieve_knowledge_snippets(query, k=k)
    # [step2] 卫语句：空结果直接返回
    if not raw_result:
        return []
    # [step3] 解析原始结果，转换为 RetrievalResult 列表
    results = []
    for line in raw_result.split('\n'):
        if not line.strip():
            continue
        # [step4] 移除 [参考N] 前缀，提取纯内容
        content = re.sub(r'^\[参考\d+\]\s*', '', line.strip())
        if content:
            results.append(RetrievalResult(content=content, source="vector", score=1.0, metadata={"type": "vector_search"}))
    log_debug(f"[GraphRAG] 向量检索返回 {len(results)} 条结果")
    return results
# [外部-结果合并] ========================================================================================================
def merge_retrieval_results(
        vector_results: List[RetrievalResult],
        graph_results: List[RetrievalResult],
        max_results: int = 10) -> List[RetrievalResult]:
    """
    交替合并向量和图谱检索结果（去重）。
    :param vector_results: 向量检索结果
    :param graph_results: 图谱检索结果
    :param max_results: 最大结果数
    :return: 合并后的结果列表
    """
    # [step1] 初始化结果容器和去重集合
    merged = []
    seen_contents = set()
    # [step2] 计算最大遍历长度（取两个列表的较大值）
    max_len = max(len(graph_results), len(vector_results))
    # [step3] 交替遍历两个列表
    for i in range(max_len):
        if len(merged) >= max_results:
            break
        if i < len(graph_results):
            result = graph_results[i]
            content_key = result.content[:100]
            if content_key not in seen_contents:
                merged.append(result)
                seen_contents.add(content_key)
        if i < len(vector_results) and len(merged) < max_results:
            result = vector_results[i]
            content_key = result.content[:100]
            if content_key not in seen_contents:
                merged.append(result)
                seen_contents.add(content_key)
    # [step4] 返回合并结果（确保不超过最大数量）
    return merged[:max_results]
# [外部-格式化检索结果] ====================================================================================================
def format_retrieval_results(results: List[RetrievalResult]) -> str:
    """
    将检索结果格式化为可读文本。
    :param results: 检索结果列表
    :return: 格式化的文本字符串
    """
    # [step1] 卫语句：空列表返回空字符串
    if not results:
        return ""
    formatted_parts = []
    # [step2] 按来源分组结果
    graph_results = [r for r in results if r.source == "graph"]
    vector_results = [r for r in results if r.source == "vector"]
    # [step3] 格式化知识图谱结果
    if graph_results:
        formatted_parts.append("=== 知识图谱检索结果 ===")
        for i, result in enumerate(graph_results, 1):
            formatted_parts.append(f"[图谱{i}] {result.content}")
    # [step4] 格式化向量检索结果
    if vector_results:
        formatted_parts.append("\n=== 向量检索结果 ===")
        for i, result in enumerate(vector_results, 1):
            formatted_parts.append(f"[文档{i}] {result.content}")
    # [step5] 拼接并返回
    return "\n".join(formatted_parts)
# [外部-检索混合知识] =====================================================================================================
def retrieve_hybrid_knowledge(query: str) -> GraphRAGResult:
    """
    执行混合检索（向量 + 知识图谱）。
    :param query: 查询文本
    :return: GraphRAGResult 结果对象
    """
    # [step1] 初始化：记录日志并初始化结果容器
    log_info("[GraphRAG] 开始混合检索...")
    entities, vector_results, graph_results = [], [], []
    # [step2] 实体提取：使用 LLM 从查询中提取医学实体
    if _is_graph_rag_enabled():
        entities = extract_medical_entities(query)
        log_info(f"[GraphRAG] 提取到 {len(entities)} 个医学实体")
        for e in entities:
            log_debug(f"  - {e.name} ({e.entity_type}, 置信度: {e.confidence:.2f})")
    # [step3] 向量检索：从向量数据库检索相似文档
    if _is_rag_enabled():
        vector_k = _get_vector_k()
        vector_results = retrieve_from_vector_store(query, k=vector_k)
        log_info(f"[GraphRAG] 向量检索返回 {len(vector_results)} 条结果")
    # [step4] 图谱检索：基于实体查询知识图谱
    if _is_graph_rag_enabled() and entities:
        kg = get_kg()
        if kg.driver:
            graph_k = _get_graph_k()
            graph_results = retrieve_from_knowledge_graph(entities, kg, limit=graph_k)
            log_info(f"[GraphRAG] 图谱检索返回 {len(graph_results)} 条结果")
        else:
            log_warn("[GraphRAG] 知识图谱不可用，跳过图谱检索")
    # [step5] 结果合并：交替合并向量和图谱结果并去重
    merged_results = merge_retrieval_results(vector_results, graph_results)
    # [step6] 格式化输出：转换为可读文本
    merged_context = format_retrieval_results(merged_results)
    log_info(f"[GraphRAG] 混合检索完成，共 {len(merged_results)} 条结果")
    # [step7] 返回结构化结果对象
    return GraphRAGResult(entities=entities, vector_results=vector_results, graph_results=graph_results, merged_context=merged_context)
# [外部-检索混合知识片段] ==================================================================================================
def retrieve_hybrid_knowledge_snippets(query: str, k: int = 3) -> str:
    """
    混合检索的简化接口，直接返回格式化文本。
    :param query: 查询文本
    :param k: 结果数量（未使用，保持接口兼容）
    :return: 格式化的知识文本
    """
    # [step1] 卫语句：Graph RAG 禁用时降级
    if not _is_graph_rag_enabled():
        log_debug("[GraphRAG] Graph RAG 已禁用，使用纯向量检索")
        return retrieve_knowledge_snippets(query, k=k)
    # [step2] 执行混合检索
    try:
        result = retrieve_hybrid_knowledge(query)
        return result.merged_context
    except Exception as e:
        log_warn(f"[GraphRAG] 混合检索失败，降级为向量检索: {e}")
        return retrieve_knowledge_snippets(query, k=k)