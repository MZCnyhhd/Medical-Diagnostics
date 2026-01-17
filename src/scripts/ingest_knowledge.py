"""
模块名称: Knowledge Ingestion (知识库摄入脚本)
功能描述:

    离线脚本，用于将医疗文档 (PDF, TXT, MD) 摄入到向量数据库。
    执行文档加载、文本切片 (Chunking)、向量化 (Embedding) 和存储流程。

设计理念:

    1.  **管道化**: Load -> Split -> Embed -> Store 清晰的流水线。
    2.  **增量更新**: (TODO) 设计上应支持只处理新增文档。
    3.  **多格式支持**: 利用 LangChain Loader 支持多种输入格式。

线程安全性:

    - 脚本通常单线程运行。

依赖关系:

    - `src.services.rag`: 向量库操作。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings
from src.services.logging import log_info, log_error, log_warn


def load_text_documents(root: Path) -> list[Document]:
    docs: list[Document] = []

    for path in list(root.rglob("*.txt")) + list(root.rglob("*.md")):
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:  # noqa: BLE001
            print(f"[RAG] 读取文件失败: {path}: {e}")
            continue

        docs.append(Document(page_content=content, metadata={"source": str(path)}))

    return docs


def ingest_docs() -> str:
    """执行知识入库流程并返回状态信息。"""
    try:
        load_dotenv("config/apikey.env", encoding="utf-8")
    except UnicodeDecodeError:
        load_dotenv("config/apikey.env", encoding="gbk")

    kb_dir = Path("data/knowledge_base")
    if not kb_dir.exists():
        return "[RAG] data/knowledge_base/ 目录不存在，请先创建并放入 .txt 医学知识文档。"

    print(f"[RAG] 正在从 {kb_dir} 加载文档...")
    raw_docs = load_text_documents(kb_dir)
    if not raw_docs:
        return "[RAG] 未在 data/knowledge_base/ 中找到任何 .txt 文档。"
    print(f"[RAG] 成功加载 {len(raw_docs)} 个文档。")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    split_docs = splitter.split_documents(raw_docs)
    print(f"[RAG] 文档已切分为 {len(split_docs)} 个片段。")

    # 检查是否启用本地 RAG
    use_local_rag = os.getenv("USE_LOCAL_RAG", "false").lower() == "true"

    if use_local_rag:
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            
            # 使用本地 HuggingFace 模型生成 Embedding
            model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            log_info(f"[RAG] 正在初始化本地 Embedding 模型: {model_name} (首次运行可能需要下载)...")
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            log_info(f"[RAG] 本地 Embedding 模型加载成功")
            
            # 创建 FAISS 索引
            log_info("[RAG] 正在生成本地 FAISS 索引 (这可能需要几分钟)...")
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            
            # 保存到本地
            save_path = "local_vector_index"
            vectorstore.save_local(save_path)
            
            return f"[RAG] 成功将 {len(split_docs)} 个文本切片写入本地 FAISS 索引 ({save_path})。"
        except Exception as e:
            log_error(f"[RAG] 本地 FAISS 索引创建失败: {e}")
            return f"[RAG] 本地 FAISS 索引创建失败: {e}"

    # --- 以下是原有 Pinecone 逻辑 ---
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "medical-knowledge")

    if not pinecone_api_key:
        return "[RAG] 未检测到 PINECONE_API_KEY，无法写入向量库。"

    embedding_model = PineconeEmbeddings(
        model=os.getenv("PINECONE_EMBEDDING_MODEL", "llama-text-embed-v2"),
    )

    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from tqdm import tqdm
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

    try:
        # 减小批次大小以平滑 Token 消耗 (TPM Limit: 250k)
        # 每个切片约 200-300 Tokens，Batch=32 -> ~10k Tokens/Batch
        # 允许一定程度的并发，配合 Tenacity 自动重试处理 429
        batch_size = 32
        total_docs = len(split_docs)
        print(f"[RAG] 开始写入向量库，共 {total_docs} 个切片，批次大小 {batch_size}，使用带重试机制的并发...")

        batches = [split_docs[i : i + batch_size] for i in range(0, total_docs, batch_size)]
        
        # 定义重试策略: 遇到异常指数退避重试，最多 5 次，最大等待 60 秒
        @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
        def process_batch_with_retry(batch_docs):
            PineconeVectorStore.from_documents(
                documents=batch_docs,
                embedding=embedding_model,
                index_name=index_name,
            )
            return True

        # 降低并发数以避免瞬间触发限流
        max_workers = 3
        failed_batches = 0
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交任务
            futures = {executor.submit(process_batch_with_retry, batch): i for i, batch in enumerate(batches)}
            
            for future in tqdm(as_completed(futures), total=len(batches), desc="Ingesting to Pinecone (Smart Parallel)"):
                try:
                    future.result()
                except Exception as e:
                    failed_batches += 1
                    print(f"\n[Error] 批次 {futures[future]} 最终失败: {e}")

        if failed_batches > 0:
            return f"[RAG] 部分写入失败: {failed_batches} 个批次未完成。"
            
    except Exception as e:  # noqa: BLE001
        return f"[RAG] 写入 Pinecone 向量库失败: {e}"
            
    except Exception as e:  # noqa: BLE001
        return f"[RAG] 写入 Pinecone 向量库失败，请检查索引是否已创建：{e}"

    return f"[RAG] 成功将 {len(split_docs)} 个文本切片写入 Pinecone 索引 '{index_name}'。"


def main() -> None:
    print(ingest_docs())


if __name__ == "__main__":
    main()
