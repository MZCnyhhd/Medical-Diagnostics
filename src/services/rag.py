"""RAG 工具模块：封装向量检索逻辑。

当前实现：
- 使用 PineconeEmbeddings 作为文本向量化模型（依赖 PINECONE_API_KEY）。
- 使用 Pinecone 向量数据库（依赖 PINECONE_API_KEY、PINECONE_INDEX_NAME）。

默认使用与索引相同的 embedding 模型（例如 llama-text-embed-v2），
以保证向量维度与索引配置一致。

如果环境变量未配置好或服务不可用，会自动降级为返回空上下文，
以保证主流程仍能正常运行。
"""

import os
from typing import List

from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from src.services.logging import log_warn, log_info


def _is_rag_enabled() -> bool:
    """
    检查是否启用 RAG 功能
    
    通过环境变量 ENABLE_RAG 控制，默认启用。
    如果设置为 "false"、"0"、"no"、"off"，则禁用 RAG。
    
    Returns:
        bool: True 表示启用 RAG，False 表示禁用
    """
    flag = os.getenv("ENABLE_RAG", "true").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _get_embedding_model() -> PineconeEmbeddings:
    """
    构造文本向量化模型实例（Embedding Model）
    
    将文本转换为高维向量，用于向量相似度搜索。
    默认使用 "llama-text-embed-v2" 模型，可通过环境变量覆盖。
    
    重要：必须确保与 Pinecone 控制台中索引使用的模型维度一致，
    否则向量检索会失败。
    
    Returns:
        PineconeEmbeddings: Embedding 模型实例
    """
    model_name = os.getenv("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2")
    return PineconeEmbeddings(model=model_name)


def _get_vectorstore() -> PineconeVectorStore | None:
    """
    获取向量数据库连接对象
    
    支持两种模式：
    1. 云端模式：使用 Pinecone 云端向量数据库（需要 API Key）
    2. 本地模式：使用本地 FAISS 向量数据库（无需 API Key，适合离线部署）
    
    需要环境变量：
    - PINECONE_API_KEY：Pinecone API Key（云端模式必需）
    - PINECONE_INDEX_NAME：Pinecone 索引名称（默认 "medical-knowledge"）
    - USE_LOCAL_RAG：设置为 "true" 启用本地模式
    
    Returns:
        PineconeVectorStore | FAISS | None: 向量数据库对象，失败返回 None
    """
    if not _is_rag_enabled():
        return None

    # ========== 本地 RAG 模式（FAISS） ==========
    # 检查是否启用本地 RAG（使用 FAISS 而非 Pinecone）
    use_local_rag = os.getenv("USE_LOCAL_RAG", "false").lower() == "true"

    if use_local_rag:
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            
            model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            
            save_path = "local_vector_index"
            if not os.path.exists(save_path):
                log_warn(f"[RAG] 本地索引目录 {save_path} 不存在，请先运行 ingest_knowledge.py")
                return None
                
            vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
            return vectorstore
        except Exception as e:
            log_warn(f"[RAG] 加载本地 FAISS 索引失败: {e}")
            return None

    # ========== 云端 RAG 模式（Pinecone） ==========
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-knowledge")

    if not api_key:
        # 未配置 Pinecone API Key，直接关闭 RAG 功能
        # 这不会影响主诊断流程，只是没有知识库增强
        return None

    # 获取 Embedding 模型（用于将查询文本向量化）
    embedding = _get_embedding_model()

    try:
        # 连接到已存在的 Pinecone 索引
        # 注意：索引需要预先在 Pinecone 控制台创建，或通过 ingest_knowledge.py 脚本创建
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embedding,
        )
    except Exception as e:  # noqa: BLE001
        # 初始化失败时降级为 None，主业务逻辑继续运行
        # 这确保即使 RAG 不可用，诊断系统仍能正常工作
        log_warn("[RAG] 初始化 Pinecone 向量索引失败：", type(e).__name__)
        return None

    return vectorstore


def retrieve_knowledge_snippets(query: str, k: int = 3) -> str:
    """
    从向量数据库中检索相关医学知识片段（RAG 核心函数）
    
    工作流程：
    1. 将查询文本（通常是医疗报告）向量化
    2. 在向量数据库中搜索最相似的 k 条知识文档
    3. 将检索结果格式化为文本，供 LLM 参考
    
    这个函数在 Agent 执行前被调用，检索到的知识会被注入到 Prompt 中，
    增强 LLM 的医学知识，提高诊断准确性。
    
    Args:
        query (str): 查询文本，通常是患者的医疗报告或其摘要
        k (int): 返回最相似的文档数量，默认 3 条
    
    Returns:
        str: 格式化的知识片段文本，例如：
            "[参考1] 糖尿病是一种...\n[参考2] 血糖控制需要..."
        如果 RAG 未启用或检索失败，返回空字符串（不影响主流程）
    
    Example:
        >>> knowledge = retrieve_knowledge_snippets("患者主诉多饮多尿", k=3)
        >>> print(knowledge)
        [参考1] 糖尿病典型症状包括多饮、多尿、多食...
        [参考2] 血糖监测是糖尿病管理的重要环节...
    """
    # 检查 RAG 是否启用
    if not _is_rag_enabled():
        return ""

    # 获取向量数据库连接
    vectorstore = _get_vectorstore()
    if vectorstore is None:
        # 向量数据库不可用，返回空字符串（降级处理）
        return ""

    try:
        # 执行向量相似度搜索
        # similarity_search 会：
        # 1. 将 query 文本向量化
        # 2. 计算与知识库中所有向量的相似度（余弦相似度）
        # 3. 返回 Top-K 最相似的文档
        docs = vectorstore.similarity_search(query, k=k)
    except Exception as e:  # noqa: BLE001
        # 检索失败时记录警告但不抛出异常，确保主流程继续
        log_warn("[RAG] 向量检索失败，已跳过向量知识库，这不会影响主诊断流程。错误类型：", type(e).__name__)
        return ""

    # 格式化检索结果
    snippets: List[str] = []
    for i, doc in enumerate(docs, start=1):
        text = (doc.page_content or "").strip()
        if not text:
            continue
        # 添加编号，便于 LLM 引用
        snippets.append(f"[参考{i}] {text}")

    # 返回拼接后的知识片段文本
    return "\n".join(snippets)


def retrieve_knowledge_with_kg(query: str, k: int = 3, use_kg: bool = True) -> str:
    """
    结合向量检索和知识图谱的混合检索
    
    工作流程：
    1. 从向量数据库检索相关文档（RAG）
    2. 从知识图谱查询结构化关系（KG）
    3. 合并两种结果，提供更全面的知识支持
    
    Args:
        query (str): 查询文本（医疗报告）
        k (int): 向量检索返回的文档数量
        use_kg (bool): 是否启用知识图谱查询
    
    Returns:
        str: 合并后的知识文本
    """
    # 第一步：向量检索（RAG）
    vector_knowledge = retrieve_knowledge_snippets(query, k=k)
    
    # 第二步：知识图谱查询（如果启用）
    kg_knowledge = ""
    if use_kg:
        try:
            from src.services.kg import get_kg
            
            kg = get_kg()
            if kg.driver:
                # 尝试从查询文本中提取症状关键词
                # 这里简化处理，实际可以用 NLP 提取
                symptoms_keywords = _extract_symptoms_from_query(query)
                
                if symptoms_keywords:
                    # 根据症状查找相关疾病
                    diseases = kg.find_diseases_by_symptoms(symptoms_keywords, limit=3)
                    
                    if diseases:
                        kg_parts = ["[知识图谱]"]
                        for disease in diseases:
                            disease_name = disease.get("disease_name", "")
                            matched_symptoms = disease.get("matched_symptoms", [])
                            match_count = disease.get("match_count", 0)
                            
                            kg_parts.append(
                                f"疾病：{disease_name}（匹配症状数：{match_count}，"
                                f"匹配症状：{', '.join(matched_symptoms[:3])}）"
                            )
                        
                        kg_knowledge = "\n".join(kg_parts)
                        log_info(f"[RAG+KG] 从知识图谱找到 {len(diseases)} 个相关疾病")
        except Exception as e:
            log_warn(f"[RAG+KG] 知识图谱查询失败: {e}")
    
    # 合并两种知识源
    if vector_knowledge and kg_knowledge:
        return f"{vector_knowledge}\n\n{kg_knowledge}"
    elif vector_knowledge:
        return vector_knowledge
    elif kg_knowledge:
        return kg_knowledge
    else:
        return ""


def _extract_symptoms_from_query(query: str) -> List[str]:
    """
    从查询文本中提取可能的症状关键词（简化版）
    
    实际应用中可以使用更复杂的 NLP 方法（如 NER、关键词匹配等）
    
    Args:
        query: 查询文本
    
    Returns:
        症状关键词列表
    """
    # 常见症状关键词列表（简化示例）
    common_symptoms = [
        "多饮", "多尿", "多食", "体重下降", "口渴",
        "腹痛", "腹泻", "便秘", "恶心", "呕吐",
        "头痛", "头晕", "心悸", "胸闷", "气短",
        "发热", "咳嗽", "咳痰", "呼吸困难",
        "皮疹", "瘙痒", "关节痛", "乏力", "失眠"
    ]
    
    found_symptoms = []
    for symptom in common_symptoms:
        if symptom in query:
            found_symptoms.append(symptom)
    
    return found_symptoms[:5]  # 最多返回 5 个症状
