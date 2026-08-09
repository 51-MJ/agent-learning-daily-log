# rag_utils.py
import chromadb
from chromadb.utils import embedding_functions

# 初始化本地持久化向量库
client = chromadb.PersistentClient(path="./vector_db")
embed_fn = embedding_functions.OllamaEmbeddingFunction(
    model_name="nomic-embed-text",
    url="http://localhost:11434"
)
collection = client.get_or_create_collection(
    name="local_knowledge",
    embedding_function=embed_fn
)

# 文本分片函数，带重叠窗口
def split_text(text: str, chunk_size=200, overlap=30) -> list:
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# 批量写入知识库
def add_knowledge(text_list: list[str], meta_ids: list[str] = None):
    all_chunks = []
    all_ids = []
    for idx, txt in enumerate(text_list):
        chunks = split_text(txt)
        for c_idx, chunk in enumerate(chunks):
            chunk_id = f"doc_{idx}_chunk_{c_idx}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
    collection.add(documents=all_chunks, ids=all_ids)

# 检索：返回topN高相似度片段
def search_knowledge(query: str, top_n=3) -> str:
    res = collection.query(query_texts=[query], n_results=top_n)
    docs = res["documents"][0]
    if not docs:
        return "暂无相关知识库内容"
    return "\n".join([f"参考片段：{d}" for d in docs])

# 导出对外接口
__all__ = ["add_knowledge", "search_knowledge"]