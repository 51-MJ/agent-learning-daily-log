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

# 批量写入知识库（支持source元数据，增量追加）
def add_knowledge(text_list: list[str], source: str = "default"):
    """
    批量写入知识库
    :param text_list: 文本列表
    :param source: 来源标识（通常是文件名），用于溯源和增量更新
    """
    all_chunks = []
    all_ids = []
    all_metas = []
    for idx, txt in enumerate(text_list):
        chunks = split_text(txt)
        for c_idx, chunk in enumerate(chunks):
            # ID包含source，避免不同文件的片段ID冲突
            chunk_id = f"{source}_doc_{idx}_chunk_{c_idx}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            # 元数据记录来源文件名
            all_metas.append({"source": source})
    collection.add(
        documents=all_chunks,
        ids=all_ids,
        metadatas=all_metas
    )

# 按source删除某个知识库文件的所有片段（用于增量更新前清理旧版本）
def delete_knowledge_by_source(source: str):
    """
    删除指定来源的所有知识库片段
    :param source: 来源文件名
    """
    try:
        # 先查出该source下所有ID
        res = collection.get(where={"source": source}, include=[])
        ids_to_delete = res["ids"]
        if ids_to_delete:
            collection.delete(ids=ids_to_delete)
            print(f"已删除来源 [{source}] 的 {len(ids_to_delete)} 个片段")
        else:
            print(f"来源 [{source}] 不存在，无需删除")
    except Exception as e:
        print(f"删除来源 [{source}] 时出错：{e}")

# 查看向量库中所有已有的来源文件
def list_knowledge_sources() -> list:
    """
    返回向量库中所有已存在的source文件名列表
    """
    all_data = collection.get(include=["metadatas"])
    sources = set()
    for meta in all_data["metadatas"]:
        if meta and "source" in meta:
            sources.add(meta["source"])
    return sorted(list(sources))

# 检索：返回topN高相似度片段，带来源信息
def search_knowledge(query: str, top_n=3, source_filter: str = None) -> str:
    """
    检索知识库
    :param query: 用户提问
    :param top_n: 返回片段数量
    :param source_filter: 可选，只检索指定来源文件
    :return: 带来源信息的检索结果文本
    """
    if source_filter:
        res = collection.query(
            query_texts=[query],
            n_results=top_n,
            where={"source": source_filter}
        )
    else:
        res = collection.query(query_texts=[query], n_results=top_n)

    docs = res["documents"][0]
    metas = res["metadatas"][0]

    if not docs:
        return "暂无相关知识库内容"

    # 拼接时带上来源文件名
    result_lines = []
    for i, (doc, meta) in enumerate(zip(docs, metas), 1):
        src = meta.get("source", "未知") if meta else "未知"
        result_lines.append(f"[来源：{src}] {doc}")
    return "\n".join(result_lines)

# 导出对外接口
__all__ = [
    "add_knowledge",
    "search_knowledge",
    "delete_knowledge_by_source",
    "list_knowledge_sources"
]
