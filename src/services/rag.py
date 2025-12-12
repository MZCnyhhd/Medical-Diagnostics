"""
RAG 工具模块：封装向量检索逻辑
================================

RAG（Retrieval-Augmented Generation，检索增强生成）是一种结合了
信息检索和文本生成的技术，通过从知识库中检索相关文档来增强
LLM 的回答能力。

RAG 的工作原理：
1. 将知识库中的文档切分成小段落
2. 使用 Embedding 模型将每个段落转换为高维向量
3. 将向量存储在向量数据库中
4. 查询时，将用户问题也转换为向量
5. 在向量数据库中搜索最相似的段落
6. 将检索到的段落注入 LLM 的 Prompt 中

在本项目中的应用：
- 存储医学知识库（疾病描述、症状、治疗方案等）
- 诊断时检索与患者症状相关的医学知识
- 将知识注入专科医生的 Prompt，提高诊断准确性

支持的向量数据库：
1. Pinecone（云端）：
   - 托管服务，无需自建基础设施
   - 支持大规模向量存储和检索
   - 需要 PINECONE_API_KEY
   
2. FAISS（本地）：
   - Facebook 开源的向量检索库
   - 完全本地运行，数据不出本地
   - 适合离线部署和隐私敏感场景

核心函数：
- retrieve_knowledge_snippets：从向量数据库检索相关知识片段
- _get_vectorstore：获取向量数据库连接
- _get_embedding_model：获取 Embedding 模型
- _is_rag_enabled：检查 RAG 功能是否启用

环境变量配置：
- ENABLE_RAG：是否启用 RAG（默认 true）
- USE_LOCAL_RAG：是否使用本地 FAISS（默认 false）
- PINECONE_API_KEY：Pinecone API Key（云端模式必需）
- PINECONE_INDEX_NAME：Pinecone 索引名称（默认 "medical-knowledge"）
- PINECONE_EMBEDDING_MODEL：Embedding 模型名称
- LOCAL_EMBEDDING_MODEL：本地 Embedding 模型名称
"""

# ==================== 标准库导入 ====================
# os：用于读取环境变量
import os
# typing：类型注解
from typing import List

# ==================== LangChain 和 Pinecone 导入 ====================
# PineconeVectorStore：LangChain 对 Pinecone 向量数据库的封装
# PineconeEmbeddings：Pinecone 提供的 Embedding 模型
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings

# ==================== 项目内部模块导入 ====================
# 日志工具
from src.services.logging import log_warn


def _is_rag_enabled() -> bool:
    """
    检查是否启用 RAG 功能
    
    通过环境变量 ENABLE_RAG 控制，提供开关以便在不同场景下灵活配置：
    - 开发调试时可能想禁用 RAG 以加快启动速度
    - 某些部署环境可能不需要 RAG 功能
    - API Key 未配置时可以手动禁用避免错误
    
    环境变量值：
    - "true"、"1"、"yes"、"on" 或不设置：启用 RAG
    - "false"、"0"、"no"、"off"：禁用 RAG
    
    Returns:
        bool: True 表示启用 RAG，False 表示禁用
    
    使用示例：
    ```python
    if _is_rag_enabled():
        # 执行 RAG 相关逻辑
        pass
    else:
        # 跳过 RAG，使用默认行为
        pass
    ```
    """
    # 从环境变量读取配置，默认值为 "true"（启用）
    flag = os.getenv("ENABLE_RAG", "true").strip().lower()
    # 检查是否为禁用值
    return flag not in {"0", "false", "no", "off"}


def _get_embedding_model() -> PineconeEmbeddings:
    """
    构造文本向量化模型实例（Embedding Model）
    
    Embedding 模型的作用：
    将文本转换为高维向量（通常是 768 或 1536 维）。
    语义相似的文本会被映射到向量空间中相近的位置，
    这样就可以通过计算向量距离来找到相关文档。
    
    模型选择：
    默认使用 Pinecone 提供的 "llama-text-embed-v2" 模型，
    这是一个高质量的文本 Embedding 模型。
    可以通过环境变量 PINECONE_EMBEDDING_MODEL 更换模型。
    
    重要注意事项：
    必须确保这里使用的模型与 Pinecone 控制台中索引配置的模型一致，
    否则向量维度不匹配会导致检索失败。
    
    Returns:
        PineconeEmbeddings: Embedding 模型实例
            - 可以调用 embed_query() 将查询文本向量化
            - 可以调用 embed_documents() 将多个文档向量化
    
    使用示例：
    ```python
    embedding = _get_embedding_model()
    # 将查询文本转换为向量
    query_vector = embedding.embed_query("患者有发热症状")
    ```
    """
    # 从环境变量读取模型名称，默认使用 llama-text-embed-v2
    model_name = os.getenv("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2")
    # 创建 PineconeEmbeddings 实例
    # 会自动使用 PINECONE_API_KEY 进行认证
    return PineconeEmbeddings(model=model_name)


def _get_vectorstore() -> PineconeVectorStore | None:
    """
    获取向量数据库连接对象
    
    支持两种模式，根据环境变量自动选择：
    
    1. 云端模式（Pinecone）：
       - 使用 Pinecone 云端向量数据库
       - 需要 PINECONE_API_KEY
       - 数据存储在 Pinecone 云端服务器
       - 适合生产环境、需要大规模存储的场景
    
    2. 本地模式（FAISS）：
       - 使用 Facebook 的 FAISS 向量检索库
       - 无需 API Key，完全本地运行
       - 数据存储在本地磁盘
       - 适合离线部署、数据隐私敏感的场景
    
    所需环境变量：
    - PINECONE_API_KEY：Pinecone API Key（云端模式必需）
    - PINECONE_INDEX_NAME：索引名称（默认 "medical-knowledge"）
    - USE_LOCAL_RAG：设置为 "true" 启用本地模式
    - LOCAL_EMBEDDING_MODEL：本地 Embedding 模型名称
    
    Returns:
        PineconeVectorStore | FAISS | None: 
            - 成功：返回向量数据库对象
            - 失败：返回 None（RAG 未启用、API Key 未配置、连接失败等）
    
    注意：
    返回 None 时不会抛出异常，调用方需要检查返回值
    """
    # 首先检查 RAG 是否启用
    if not _is_rag_enabled():
        # RAG 被禁用，直接返回 None
        return None

    # ==================== 本地 RAG 模式（FAISS）====================
    # 检查是否启用本地 RAG
    use_local_rag = os.getenv("USE_LOCAL_RAG", "false").lower() == "true"

    if use_local_rag:
        try:
            # 导入本地 RAG 所需的库
            # FAISS：Facebook 开源的向量检索库
            from langchain_community.vectorstores import FAISS
            # HuggingFaceEmbeddings：使用 HuggingFace 模型生成 Embedding
            from langchain_huggingface import HuggingFaceEmbeddings
            
            # 获取本地 Embedding 模型名称
            # 默认使用 all-MiniLM-L6-v2，这是一个轻量级但效果不错的模型
            model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            # 创建 Embedding 模型实例
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            
            # 检查本地索引目录是否存在
            save_path = "local_vector_index"
            if not os.path.exists(save_path):
                # 索引不存在，提示用户运行入库脚本
                log_warn(f"[RAG] 本地索引目录 {save_path} 不存在，请先运行 ingest_knowledge.py")
                return None
            
            # 加载本地 FAISS 索引
            # allow_dangerous_deserialization=True 允许反序列化
            # （FAISS 索引使用 pickle 序列化，需要此选项）
            vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
            return vectorstore
            
        except Exception as e:
            # 本地 RAG 加载失败
            log_warn(f"[RAG] 加载本地 FAISS 索引失败: {e}")
            return None

    # ==================== 云端 RAG 模式（Pinecone）====================
    # 获取 Pinecone 配置
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-knowledge")

    # 检查 API Key 是否配置
    if not api_key:
        # 未配置 Pinecone API Key，静默返回 None
        # 这不会影响主诊断流程，只是没有知识库增强
        return None

    # 获取 Embedding 模型
    # 用于将查询文本向量化
    embedding = _get_embedding_model()

    try:
        # 连接到已存在的 Pinecone 索引
        # 注意：索引需要预先在 Pinecone 控制台创建
        # 或通过 ingest_knowledge.py 脚本创建
        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=index_name,
            embedding=embedding,
        )
    except Exception as e:  # noqa: BLE001
        # 初始化失败时降级为 None
        # 主业务逻辑会继续运行，只是没有 RAG 增强
        log_warn("[RAG] 初始化 Pinecone 向量索引失败：", type(e).__name__)
        return None

    return vectorstore


def retrieve_knowledge_snippets(query: str, k: int = 3) -> str:
    """
    从向量数据库中检索相关医学知识片段（RAG 核心函数）
    
    这是 RAG 的核心检索函数，在专科医生智能体执行前被调用。
    检索到的知识会被注入到 Prompt 中，增强 LLM 的医学知识。
    
    工作流程：
    ==========
    
    第一步：检查 RAG 是否可用
    - 检查 ENABLE_RAG 环境变量
    - 检查向量数据库连接是否成功
    
    第二步：执行向量相似度搜索
    - 将查询文本（通常是医疗报告）向量化
    - 在向量数据库中计算余弦相似度
    - 返回 Top-K 最相似的文档
    
    第三步：格式化检索结果
    - 为每个检索到的文档添加编号
    - 拼接成便于 LLM 阅读的格式
    
    Args:
        query (str): 查询文本
            - 通常是患者的医疗报告或其摘要
            - 也可以是特定的医学问题
            - 例如："患者主诉多饮多尿，体重下降"
        
        k (int): 返回最相似的文档数量，默认 3 条
            - 较小的 k 值：更精确但可能遗漏相关信息
            - 较大的 k 值：更全面但可能引入噪音
            - 建议根据 Prompt 长度限制调整
    
    Returns:
        str: 格式化的知识片段文本
            
            成功时返回：
            ```
            [参考1] 糖尿病是一种以高血糖为特征的代谢性疾病...
            [参考2] 血糖监测是糖尿病管理的重要环节，包括...
            [参考3] 糖尿病的治疗方法包括饮食控制、运动...
            ```
            
            失败时返回空字符串（不影响主流程）：
            - RAG 未启用
            - 向量数据库连接失败
            - 检索过程发生错误
    
    使用示例：
    ```python
    # 检索与症状相关的医学知识
    knowledge = retrieve_knowledge_snippets("患者主诉多饮多尿", k=3)
    
    if knowledge:
        # 将知识注入 Prompt
        prompt = f"参考知识：\\n{knowledge}\\n\\n请分析以下病例..."
    else:
        # 没有检索到知识，使用原始 Prompt
        prompt = "请分析以下病例..."
    ```
    
    性能说明：
    - 向量检索通常在毫秒级完成
    - 主要延迟在网络通信（云端模式）
    - 本地模式几乎无延迟
    """
    # ========== 第一步：检查 RAG 是否可用 ==========
    # 检查环境变量是否启用 RAG
    if not _is_rag_enabled():
        # RAG 被禁用，返回空字符串
        return ""

    # 获取向量数据库连接
    vectorstore = _get_vectorstore()
    if vectorstore is None:
        # 向量数据库不可用
        # 可能的原因：API Key 未配置、连接失败、本地索引不存在
        # 返回空字符串，不影响主流程（降级处理）
        return ""

    # ========== 第二步：执行向量相似度搜索 ==========
    try:
        # 调用向量数据库的相似度搜索方法
        # similarity_search 会：
        # 1. 将 query 文本使用 Embedding 模型向量化
        # 2. 在向量数据库中计算与所有文档向量的余弦相似度
        # 3. 返回相似度最高的 k 个文档
        docs = vectorstore.similarity_search(query, k=k)
    except Exception as e:  # noqa: BLE001
        # 检索失败时记录警告但不抛出异常
        # 确保主流程可以继续执行
        log_warn("[RAG] 向量检索失败，已跳过向量知识库，这不会影响主诊断流程。错误类型：", type(e).__name__)
        return ""

    # ========== 第三步：格式化检索结果 ==========
    # 用于存储格式化后的知识片段
    snippets: List[str] = []
    
    # 遍历检索到的文档
    for i, doc in enumerate(docs, start=1):
        # 提取文档内容并清理空白字符
        text = (doc.page_content or "").strip()
        # 跳过空文档
        if not text:
            continue
        # 添加编号，便于 LLM 在回答时引用
        # 格式：[参考1] 文档内容...
        snippets.append(f"[参考{i}] {text}")

    # 将所有知识片段用换行符连接
    return "\n".join(snippets)
