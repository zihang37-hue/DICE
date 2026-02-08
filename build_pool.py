import json
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import ollama
from tqdm import tqdm
from dice_agent import DICEAgent, exact_match

TARGET_SIZE = 500  # 本次新增的目标条数
START_INDEX = 0    # 从筛选后的数据集索引开始（可设置为2000等）
DB_PATH = "./dice_vector_db"
COLLECTION_NAME = "hotpotqa_pool"
MAX_STEPS = 6
LEVELS = ["hard", "medium"]  # 可改为 None 表示不过滤难度

EXTRACTION_PROMPT = """Extract the abstract reasoning strategy from the trajectory below.

Rules:
- Do NOT mention any specific names, dates, places, or entities.
- Use generic terms: Entity A, Entity B, Subject, Attribute.
- Output exactly three lines: Type, Steps, Strategy.

Example 1:
Trajectory:
Task: Which of Entity A or Entity B was created first?
Thought: I need to find when Entity A was created.
Action: Search[Entity A]
Observation: [Title: Entity A] Entity A was created in 1990...
Thought: Now I need to find when Entity B was created.
Action: Search[Entity B]
Observation: [Title: Entity B] Entity B was created in 1975...
Thought: Entity B (1975) is earlier than Entity A (1990).
Action: Finish[Entity B]

Output:
Type: comparison
Steps: 1) Search Entity A for Attribute. 2) Search Entity B for same Attribute. 3) Compare and finish.
Strategy: Search two entities separately, compare a shared attribute, return the one that fits.

Example 2:
Trajectory:
Task: What is the occupation of the founder of Entity A?
Thought: I need to find who founded Entity A.
Action: Search[Entity A]
Observation: [Title: Entity A] Entity A was founded by Subject B...
Thought: Now I need to find Subject B's occupation.
Action: Search[Subject B]
Observation: [Title: Subject B] Subject B is a scientist...
Thought: Subject B is a scientist.
Action: Finish[scientist]

Output:
Type: multi-hop
Steps: 1) Search Entity A to find intermediate Subject. 2) Search Subject for target Attribute. 3) Finish.
Strategy: Chain through an intermediate entity discovered in the first search to reach the final answer.

Example 3:
Trajectory:
Task: What is the Attribute of Entity A?
Thought: I should search for Entity A.
Action: Search[Entity A]
Observation: [Title: Entity A] Entity A is a French novelist...
Thought: The answer is French.
Action: Finish[French]

Output:
Type: lookup
Steps: 1) Search Entity A. 2) Extract Attribute from results. 3) Finish.
Strategy: Direct single-entity search and attribute extraction.

Now extract from this trajectory:
{trajectory}

Output:"""

def history_to_trajectory(question, history, final_answer):
    """将baseline运行时的history转为轨迹字符串"""
    traj = f"Task: {question}\n"
    for record in history:
        traj += f"Thought: {record['thought']}\n"
        traj += f"Action: {record['action']}\n"
        traj += f"Observation: {record['observation']}\n"
    traj += "Thought: I have sufficient information to answer.\n"
    traj += f"Action: Finish[{final_answer}]\n"
    return traj


def build():
    print(f"初始化Embedding模型")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"初始化向量数据库(ChromaDB)")
    # 将数据存储到DB_PATH这个文件中（追加写入，不清空旧库）
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    print(f"加载HotpotQA数据集")
    dataset = load_dataset("hotpot_qa", "distractor", split=f"train", trust_remote_code=True)
    # 难度筛选
    if LEVELS:
        filtered_dataset = dataset.filter(lambda x: x['level'] in LEVELS)
        print(f"筛选难度 {LEVELS} 后: {len(filtered_dataset)}")
    else:
        filtered_dataset = dataset
    filtered_dataset = filtered_dataset.shuffle(seed=42)
    if START_INDEX > 0:
        filtered_dataset = filtered_dataset.select(range(START_INDEX, len(filtered_dataset)))

    # baseline agent（用于跑题收集成功轨迹）
    agent = DICEAgent()

    print(f"开始构建知识库（baseline刷题 + Gemma提取TK）")

    # 已有ID，避免重复写入
    existing = set(collection.get(include=[]).get("ids", []))
    ids = []
    embeddings = []
    metadatas = []
    documents = [] # 存储TK

    for i, item in enumerate(tqdm(filtered_dataset, desc="Processing")):
        if len(ids) >= TARGET_SIZE:
            break

        # 用 baseline agent 跑题，只保留答对的轨迹
        question = item["question"]
        gold_answer = item["answer"]
        pred = agent.run_task(question, max_steps=MAX_STEPS, use_tk=False)
        if not pred or not exact_match(pred, gold_answer):
            continue
        raw_traj = history_to_trajectory(question, agent.history, pred)
        # 提取知识（通过本地Gemma）
        try:
            res = ollama.generate(
                model="gemma:2b",
                prompt=EXTRACTION_PROMPT.format(trajectory=raw_traj),
                options={'temperature': 0}
            )
            tk = res['response'].strip()
        except Exception as e:
            print(f"Error extracting knowledge for item {i}: {e}")
            continue
        
        # 计算向量
        vec = encoder.encode(tk).tolist()
        item_id = str(item['id'])
        if item_id in existing:
            continue
        ids.append(item_id)
        embeddings.append(vec)
        metadatas.append({"raw_trajectory": raw_traj})
        documents.append(tk)

        print(f"已成功生成 {len(ids)} 条（尝试了 {i} 题）")
    
    # 检查是否有成功处理的数据
    if not ids:
        print("⚠️ 警告：没有成功处理任何样本！请检查Ollama服务是否正常运行。")
        return
    
    print(f"✅ 成功处理 {len(ids)}/{TARGET_SIZE} 个样本")
    print(f"📦 添加到向量数据库...")
    
    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas
    )
    print(f"🎉 知识库构建完成！共 {len(ids)} 条记录。")

if __name__ == "__main__":
    build()
    