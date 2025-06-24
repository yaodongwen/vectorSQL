import sqlite3
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import os
import json
import pickle
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import time
import matplotlib.pyplot as plt
import sqlite_vec 

# 加载嵌入模型
def load_model(model_path="/Users/yaodongwen/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"):
    try:
        # 首先尝试加载本地模型
        model_path = model_path
        model = SentenceTransformer(model_path)
        print(f"成功加载本地模型: {model_path}")
    except Exception as e:
        # 如果本地模型加载失败，使用Hugging Face上的小型模型
        print(f"本地模型加载失败: {e}")
        print("尝试从Hugging Face加载多语言模型...")
        model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
        print("成功加载Hugging Face模型")

    # 获取模型的输出维度
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"模型输出维度: {embedding_dim}")
    return model

# 创建一个SQLite数据库连接
def conn_sql(path="vec_no_vec1.db"):
    no_vec0_db_path = path
    conn = sqlite3.connect(no_vec0_db_path)
    cursor = conn.cursor()

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)  # 创建使用vec0的虚拟表
    # 测试扩展是否成功加载
    cursor.execute("SELECT vec_version()")
    version = cursor.fetchone()[0]
    print(f"成功加载sqlite-vec扩展，版本: {version}")
    return conn


# 定义插入函数
def insert_documents(conn, cursor, documents):
    # 清空表
    cursor.execute("DELETE FROM vec_documents")
    for doc in documents:
        # 虚拟表插入（如果可用）
        embedding_json = json.dumps(doc["embedding"].tolist())
        try:
            cursor.execute(
                "INSERT INTO vec_documents(document_id, content_embedding, category, original_content) VALUES (?, vec_f32(?), ?, ?)",
                (doc["id"], embedding_json, doc["category"], doc["content"])
            )
        except Exception as e:
            print(f"插入虚拟表失败: {e}")
    conn.commit()

def knn_search(cursor, query_embedding, k=3, category=None):
    """执行KNN搜索"""
    # 使用vec0虚拟表
    query_json = json.dumps(query_embedding.tolist())
    if category:
        cursor.execute("""
        SELECT document_id, original_content, category, distance
        FROM vec_documents 
        WHERE content_embedding MATCH ? AND k = ? AND category = ?
        """, (query_json, k, category))
    else:
        cursor.execute("""
        SELECT document_id, original_content, category, distance
        FROM vec_documents 
        WHERE content_embedding MATCH ? AND k = ?
        """, (query_json, k))
    
    return [(row[0], row[1], row[2], row[3]) for row in cursor.fetchall()]

def main():
    model = load_model()
    # 获取模型的输出维度
    embedding_dim = model.get_sentence_embedding_dimension()
    conn = conn_sql()
    cursor = conn.cursor()
    # 准备一些示例文本
    documents = [
        {"id": 1, "content": "机器学习是人工智能的一个子领域", "category": "技术"},
        {"id": 2, "content": "深度学习是机器学习的一种方法", "category": "技术"},
        {"id": 3, "content": "向量数据库可以高效存储和检索向量数据", "category": "数据库"},
        {"id": 4, "content": "SQLite是一个轻量级的关系型数据库", "category": "数据库"},
        {"id": 5, "content": "Python是一种流行的编程语言", "category": "编程"},
        {"id": 6, "content": "自然语言处理是处理人类语言的技术", "category": "技术"},
        {"id": 7, "content": "向量相似度搜索在推荐系统中很常用", "category": "技术"},
        {"id": 8, "content": "大数据分析需要高效的数据存储和处理", "category": "数据"}
    ]

    # 为每个文档生成embedding向量
    for doc in documents:
        embedding = model.encode(doc["content"])
        doc["embedding"] = embedding

    # 展示部分数据
    for doc in documents[:2]:
        print(f"ID: {doc['id']}, 内容: {doc['content']}")
        print(f"向量维度: {len(doc['embedding'])}, 向量前几个元素: {doc['embedding'][:5]}...\n")

    # 使用sqlite-vec插入
    insert_documents(conn, cursor, documents)

    # 生成查询向量
    query_text = "数据库技术与应用"
    query_embedding = model.encode(query_text)

    # 使用sqlite-vec执行KNN查询
    k = 3
    results = knn_search(cursor, query_embedding, k)

    print(f"查询：'{query_text}'的最近{k}个结果 (使用sqlite-vec):")
    for doc_id, content, category, distance in results:
        print(f"ID: {doc_id}, 距离: {distance:.4f}, 类别: {category}, 内容: {content}")

if __name__=="__main__":
    main()