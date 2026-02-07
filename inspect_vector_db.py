import chromadb

# 连接数据库
client = chromadb.PersistentClient(path="./dice_vector_db")
collection = client.get_collection("hotpotqa_pool")

# 1. 查看基本信息
print("=" * 60)
print("向量数据库信息")
print("=" * 60)
print(f"Collection名称: {collection.name}")
print(f"总记录数: {collection.count()}")
print()

# 2. 获取所有数据（小数据集可以这样做）
results = collection.get(
    include=['embeddings', 'documents', 'metadatas']
)

print("=" * 60)
print("数据结构")
print("=" * 60)
print(f"IDs数量: {len(results['ids'])}")
print(f"Embeddings形状: {len(results['embeddings'])}条 x {len(results['embeddings'][0])}维")
print(f"Documents数量: {len(results['documents'])}")
print(f"Metadatas数量: {len(results['metadatas'])}")
print()

# 3. 查看前3条数据
print("=" * 60)
print("前3条数据示例")
print("=" * 60)
for i in range(min(3, len(results['ids']))):
    print(f"\n【记录 {i+1}】")
    print(f"ID: {results['ids'][i]}")
    print(f"Embedding维度: {len(results['embeddings'][i])}")
    print(f"Embedding前5维: {results['embeddings'][i][:5]}")
    print(f"\nTransferable Knowledge (TK):")
    print(results['documents'][i][:200] + "..." if len(results['documents'][i]) > 200 else results['documents'][i])
    print(f"\nRaw Trajectory (前200字符):")
    traj = results['metadatas'][i]['raw_trajectory']
    print(traj[:200] + "..." if len(traj) > 200 else traj)
    print("-" * 60)

# 4. 测试相似度查询
print("\n" + "=" * 60)
print("测试：查询与第1条最相似的3条记录")
print("=" * 60)
query_results = collection.query(
    query_embeddings=[results['embeddings'][0]],
    n_results=3,
    include=['documents', 'distances']
)

for i, (doc, dist) in enumerate(zip(query_results['documents'][0], query_results['distances'][0])):
    print(f"\n相似度排名 {i+1} (距离: {dist:.4f}):")
    print(doc[:150] + "..." if len(doc) > 150 else doc)