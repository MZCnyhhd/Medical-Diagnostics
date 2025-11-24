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
from src.services.logging import log_warn


def _is_rag_enabled() -> bool:
    flag = os.getenv("ENABLE_RAG", "true").strip().lower()
    return flag not in {"0", "false", "no", "off"}


def _get_embedding_model() -> PineconeEmbeddings:
    """构造 Pinecone 向量模型实例。

    默认使用 "llama-text-embed-v2" 模型，可通过 PINECONE_EMBEDDING_MODEL 覆盖。
    需要确保与 Pinecone 控制台中索引使用的模型维度一致。
    """

    model_name = os.getenv("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2")
    return PineconeEmbeddings(model=model_name)


def _get_vectorstore() -> PineconeVectorStore | None:
    """基于已有的 Pinecone 索引构造一个向量检索对象。

    需要：
    - PINECONE_API_KEY：Pinecone API Key（由 langchain-pinecone / pinecone 客户端读取）。
    - PINECONE_INDEX_NAME：已经在 Pinecone 控制台或入库脚本中创建好的索引名称。
    """

    if not _is_rag_enabled():
        return None

    # 检查是否启用本地 RAG
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

    # --- 以下是原有 Pinecone 逻辑 ---
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-knowledge")

    if not api_key:
        # 未配置 Pinecone，直接关闭 RAG 功能
        return None

    embedding = _get_embedding_model()

    try:
        # 假设索引已经存在；实际上可以通过 Pinecone 控制台或入库脚本创建
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embedding,
        )
    except Exception as e:  # noqa: BLE001
        # 初始化失败时降级为 None，主业务逻辑继续运行
        log_warn("[RAG] 初始化 Pinecone 向量索引失败：", type(e).__name__)
        return None

    return vectorstore


def retrieve_knowledge_snippets(query: str, k: int = 3) -> str:
    """根据查询语句从向量库中检索 k 条相关医学知识，拼接为短文本。

    - query 通常可以是患者的原始医疗报告或其摘要。
    - 若向量库不可用或检索失败，返回空字符串。
    """

    if not _is_rag_enabled():
        return ""

    vectorstore = _get_vectorstore()
    if vectorstore is None:
        return ""

    try:
        docs = vectorstore.similarity_search(query, k=k)
    except Exception as e:  # noqa: BLE001
        log_warn("[RAG] 向量检索失败，已跳过向量知识库，这不会影响主诊断流程。错误类型：", type(e).__name__)
        return ""

    snippets: List[str] = []
    for i, doc in enumerate(docs, start=1):
        text = (doc.page_content or "").strip()
        if not text:
            continue
        snippets.append(f"[参考{i}] {text}")

    return "\n".join(snippets)
