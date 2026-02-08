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

EXTRACTION_PROMPT = """Please extract the transferable knowledge from the following trajectory of an agent. The trajectory contains a sequence of Action and Observations, describing how the agent solve a question by thinking, acting and observing. Here are some examples.

Trajectory:

Question: Which magazine was started first Arthur's Magazine or First for Women?

Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.

Action 1: Search[Arthur's Magazine]

Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia in the 19th century.

Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.

Action 2: Search[First for Women]

Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA.[1] The magazine was started in 1989.

Thought 3: First for Women was started in 1989. 1844 (Arthur's Magazine) < 1989 (First for Women), so Arthur's Magazine was started first.

Action 3: Finish[Arthur's Magazine]

Transferable Knowledge: We need to compare the feature "start year" of the two magazines, so we need to search the two magazines respectively, find out the start year of each of them and compare this feature to figure out which one is earlier.

Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?

Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.

Action 1: Search[Pavel Urysohn]

Observation 1: Pavel Samuilovich Urysohn (February 3, 1898 - August 17, 1924) was a Soviet mathematician who is best known for his contributions in dimension theory.

Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next and find its type of work.

Action 2: Search[Leonid Levin]

Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.

Thought 3: Leonid Levin is a mathematician and computer scientist. So Pavel Urysohn and Leonid Levin have the same type of work.

Action 3: Finish[yes]

Transferable Knowledge: We need to compare the feature "work" of the two persons to determine whether they were known for the same type of work, so we need to search the two person respectively, find out the job they did, and compare if they are the same.

Trajectory:
{trajectory}

Transferable Knowledge:"""

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
    