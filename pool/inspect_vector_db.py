from pathlib import Path

import chromadb

ROOT_DIR = Path(__file__).resolve().parents[1]

def load_config():
    config_path = ROOT_DIR / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found at: {config_path}")
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Missing dependency: PyYAML. Install with `pip install pyyaml`.") from exc
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def get_required(cfg, *keys):
    cur = cfg
    path = []
    for key in keys:
        path.append(str(key))
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing config: {'.'.join(path)}")
        cur = cur[key]
    return cur

CONFIG = load_config()
INSPECT_CFG = get_required(CONFIG, "inspect_vector_db")
DB_PATH = str(ROOT_DIR / get_required(CONFIG, "db_path"))
COLLECTION_NAME = get_required(CONFIG, "collection_name")
EXAMPLE_INDEX = int(get_required(INSPECT_CFG, "example_index"))
NUM_EXAMPLES = int(get_required(INSPECT_CFG, "num_examples"))
QUERY_TOP_K = int(get_required(INSPECT_CFG, "query_top_k"))

# 连接数据库
client = chromadb.PersistentClient(path=DB_PATH)
collection = client.get_collection(COLLECTION_NAME)

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

# 3. 连续查看 num_examples 条示例及其 TK、轨迹
total = len(results['ids'])
start = max(0, min(EXAMPLE_INDEX, total - 1))
end = min(start + NUM_EXAMPLES, total)
print("=" * 60)
print(f"示例与 TK（共 {end - start} 条，索引 {start} ~ {end - 1}）")
print("=" * 60)
for idx in range(start, end):
    i = idx
    print(f"\n【记录 {i + 1} / {total}】 ID: {results['ids'][i]}")
    print(f"\nTransferable Knowledge (TK):")
    tk = results['documents'][i]
    print(tk if len(tk) <= 500 else tk[:500] + "\n... (截断)")
    print(f"\nRaw Trajectory:")
    traj = results['metadatas'][i]['raw_trajectory']
    print(traj if len(traj) <= 600 else traj[:600] + "\n... (截断)")
    print("-" * 60)

# 4. 测试相似度查询
print("\n" + "=" * 60)
print("测试：查询与第1条最相似的3条记录")
print("=" * 60)
query_results = collection.query(
    query_embeddings=[results['embeddings'][0]],
    n_results=QUERY_TOP_K,
    include=['documents', 'distances']
)

for i, (doc, dist) in enumerate(zip(query_results['documents'][0], query_results['distances'][0])):
    print(f"\n相似度排名 {i+1} (距离: {dist:.4f}):")
    print(doc[:150] + "..." if len(doc) > 150 else doc)
