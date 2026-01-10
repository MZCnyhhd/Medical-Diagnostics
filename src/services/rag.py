# [导入模块] ############################################################################################################
# [标准库 | Standard Libraries] =========================================================================================
import os                                                              # 操作系统接口：环境变量与路径操作
from typing import List, Any, Optional                                 # 类型提示：类型注解支持
# [第三方库 | Third-party Libraries] ====================================================================================
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings # Pinecone：云端向量存储
# 尝试导入 FAISS 以用于类型提示，如果不可用则忽略（实际使用在函数内部再次导入或处理）
try:
    from langchain_community.vectorstores import FAISS                 # FAISS：本地向量存储
except ImportError:
    FAISS = Any
# [内部模块 | Internal Modules] =========================================================================================
from src.services.logging import log_warn                              # 统一日志服务：警告日志
# [定义函数] ############################################################################################################
# [内部- RAG 是否已启用] ==================================================================================================
def _is_rag_enabled() -> bool:
    """
    检测 RAG 功能是否启用。
    通过环境变量 ENABLE_RAG 控制，默认启用。
    :return: True 表示启用，False 表示禁用
    """
    # [step1] 读取环境变量并标准化
    flag = os.getenv("ENABLE_RAG", "true").strip().lower()
    # [step2] 判断是否为禁用标志
    return flag not in {"0", "false", "no", "off"}
# [内部-获取嵌入模型] =====================================================================================================
def _get_embedding_model() -> PineconeEmbeddings:
    """
    获取 Pinecone 嵌入模型实例。
    :return: PineconeEmbeddings 对象
    """
    # [step1] 从环境变量获取模型名称，默认使用 llama-text-embed-v2
    model_name = os.getenv("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2")
    # [step2] 返回嵌入模型实例
    return PineconeEmbeddings(model=model_name)
# [内部-加载本地FAISS索引] ================================================================================================
def _load_local_faiss() -> Optional["FAISS"]:
    """
    加载本地 FAISS 向量索引。
    :return: FAISS 向量存储实例，加载失败返回 None
    """
    # [step1] 延迟导入本地依赖
    try:
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
    except ImportError as e:
        log_warn(f"[RAG] 本地模式依赖缺失: {e}")
        return None
    # [step2] 获取本地嵌入模型
    model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    # [step3] 检查索引目录是否存在
    save_path = "local_vector_index"
    if not os.path.exists(save_path):
        log_warn(f"[RAG] 本地索引目录 {save_path} 不存在，请先运行 ingest_knowledge.py")
        return None
    # [step4] 加载 FAISS 索引
    try:
        return FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        log_warn(f"[RAG] 加载本地 FAISS 索引失败: {e}")
        return None
# [内部-加载Pinecone云端索引] =============================================================================================
def _load_pinecone_index() -> Optional[PineconeVectorStore]:
    """
    加载 Pinecone 云端向量索引。
    :return: PineconeVectorStore 实例，加载失败返回 None
    """
    # [step1] 检查 API Key 是否配置
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        return None
    # [step2] 获取索引名称和嵌入模型
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-knowledge")
    embedding = _get_embedding_model()
    # [step3] 连接 Pinecone 索引
    try:
        return PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embedding)
    except Exception as e:
        log_warn("[RAG] 初始化 Pinecone 向量索引失败：", type(e).__name__)
        return None
# [内部-获取向量存储] =====================================================================================================
def _get_vectorstore() -> Any:
    """
    获取向量存储实例（支持本地 FAISS 或云端 Pinecone）。
    优先级：本地模式 > 云端模式。
    :return: 向量存储实例，不可用时返回 None
    """
    # [step1] 卫语句：RAG 未启用直接返回
    if not _is_rag_enabled():
        return None
    # [step2] 检查是否使用本地模式
    use_local_rag = os.getenv("USE_LOCAL_RAG", "false").lower() == "true"
    if use_local_rag:
        return _load_local_faiss()
    # [step3] 默认使用云端 Pinecone
    return _load_pinecone_index()
# [外部-知识检索] ========================================================================================================
def retrieve_knowledge_snippets(query: str, k: int = 3) -> str:
    """
    从向量数据库检索与查询相关的知识片段。
    :param query: 查询文本
    :param k: 返回结果数量
    :return: 格式化的知识片段字符串，失败返回空字符串
    """
    # [step1] 卫语句：RAG 未启用或向量存储不可用
    if not _is_rag_enabled():
        return ""
    vectorstore = _get_vectorstore()
    if vectorstore is None:
        return ""
    # [step2] 执行相似度搜索
    try:
        docs = vectorstore.similarity_search(query, k=k)
    except Exception as e:
        log_warn("[RAG] 向量检索失败，已跳过向量知识库。错误类型：", type(e).__name__)
        return ""
    # [step3] 格式化检索结果
    snippets: List[str] = []
    for i, doc in enumerate(docs, start=1):
        text = (doc.page_content or "").strip()
        if text:
            snippets.append(f"[参考{i}] {text}")
    # [step4] 返回拼接结果
    return "\n".join(snippets)
