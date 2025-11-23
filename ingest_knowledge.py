"""知识入库脚本：将本地医学知识文件写入 Pinecone 向量库。

使用方式（示例）：
1. 在项目根目录下创建 `KnowledgeBase/` 文件夹，并放入若干 `.txt` 或 `.md` 医学知识文档。
2. 在 `apikey.env` 中配置：
   - DASHSCOPE_API_KEY=你的_DashScope_Key
   - PINECONE_API_KEY=你的_Pinecone_Key
   - PINECONE_INDEX_NAME=medical-knowledge  (可自定义)
3. 安装依赖：
   - pip install -r requirements.txt
4. 运行本脚本：
   - python ingest_knowledge.py

脚本会：
- 读取 KnowledgeBase/ 下的所有 .txt 文件；
- 使用 DashScopeEmbeddings 将文本切分并向量化；
- 写入指定的 Pinecone 索引中。

注意：
- 若索引不存在，需先在 Pinecone 控制台创建，或根据实际需要扩展本脚本以自动创建索引。
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore, PineconeEmbeddings


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
    load_dotenv("apikey.env")

    kb_dir = Path("KnowledgeBase")
    if not kb_dir.exists():
        return "[RAG] KnowledgeBase/ 目录不存在，请先创建并放入 .txt 医学知识文档。"

    raw_docs = load_text_documents(kb_dir)
    if not raw_docs:
        return "[RAG] 未在 KnowledgeBase/ 中找到任何 .txt 文档。"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    split_docs = splitter.split_documents(raw_docs)

    # 检查是否启用本地 RAG
    use_local_rag = os.getenv("USE_LOCAL_RAG", "false").lower() == "true"

    if use_local_rag:
        try:
            from langchain_community.vectorstores import FAISS
            from langchain_huggingface import HuggingFaceEmbeddings
            
            # 使用本地 HuggingFace 模型生成 Embedding
            model_name = os.getenv("LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
            
            # 创建 FAISS 索引
            vectorstore = FAISS.from_documents(split_docs, embeddings)
            
            # 保存到本地
            save_path = "local_vector_index"
            vectorstore.save_local(save_path)
            
            return f"[RAG] 成功将 {len(split_docs)} 个文本切片写入本地 FAISS 索引 ({save_path})。"
        except Exception as e:
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

    try:
        # 分批处理以避免 Rate Limit Error
        batch_size = 10
        total_docs = len(split_docs)
        
        for i in range(0, total_docs, batch_size):
            batch = split_docs[i : i + batch_size]
            PineconeVectorStore.from_documents(
                documents=batch,
                embedding=embedding_model,
                index_name=index_name,
            )
            # 每批次处理完暂停 2 秒
            time.sleep(2)
            
    except Exception as e:  # noqa: BLE001
        return f"[RAG] 写入 Pinecone 向量库失败，请检查索引是否已创建：{e}"

    return f"[RAG] 成功将 {len(split_docs)} 个文本切片写入 Pinecone 索引 '{index_name}'。"


def main() -> None:
    print(ingest_docs())


if __name__ == "__main__":
    main()
