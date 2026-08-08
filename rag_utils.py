# rag_utils.py
# 无langchain版本，仅使用chromadb、ollama原生接口
import os
import chromadb
from chromadb.utils import embedding_functions

# 路径配置
CHROMA_DB_PATH = "./chroma_db"
DOC_FILE_PATH = "./docs/fund.txt"
EMBED_MODEL = "nomic-embed-text"
COLLECTION_NAME = "knowledge_base"

# 初始化嵌入模型
def get_embedding_fn():
    return embedding_functions.OllamaEmbeddingFunction(
        model_name=EMBED_MODEL
    )

# 读取本地文档文本
def load_document_text():
    with open(DOC_FILE_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    # 简单文本分片
    chunks = []
    chunk_size = 200
    overlap = 20
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        chunks.append(chunk)
    return chunks

# 初始化向量库
def init_vector_db():
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    embed_fn = get_embedding_fn()
    try:
        coll = client.get_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
    except:
        # 集合不存在，新建并写入文档
        coll = client.create_collection(name=COLLECTION_NAME, embedding_function=embed_fn)
        texts = load_document_text()
        ids = [f"doc_{i}" for i in range(len(texts))]
        coll.add(documents=texts, ids=ids)
    return coll

# 强制知识库检索
def search_knowledge(query: str) -> str:
    coll = init_vector_db()
    res = coll.query(query_texts=[query], n_results=10)
    docs = res["documents"][0]
    print("\n=====RAG检索到的知识库内容=====")
    for item in docs:
        print(item)
    print("================================\n")
    return "\n".join(docs)