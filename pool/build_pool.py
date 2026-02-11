import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import ollama
from tqdm import tqdm
from dice import DICEAgent, exact_match

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

# 功能点：从 config 读取建库与模型相关配置
CONFIG = load_config()
MODELS = get_required(CONFIG, "models")
BUILD_CFG = get_required(CONFIG, "build_pool")

DB_PATH = str(ROOT_DIR / get_required(CONFIG, "db_path"))
COLLECTION_NAME = get_required(CONFIG, "collection_name")
EMBEDDING_MODEL = get_required(MODELS, "embedding_model")
RETRIEVER_LLM = get_required(MODELS, "retriever_llm")

TARGET_SIZE = int(get_required(BUILD_CFG, "target_size"))
START_INDEX = int(get_required(BUILD_CFG, "start_index"))
MAX_STEPS = int(get_required(BUILD_CFG, "max_steps"))
LEVELS = get_required(BUILD_CFG, "levels")
DATASET_NAME = get_required(BUILD_CFG, "dataset_name")
DATASET_CONFIG = get_required(BUILD_CFG, "dataset_config")
DATASET_SPLIT = get_required(BUILD_CFG, "split")

EXTRACTION_PROMPT = """Please extract the transferable knowledge from the following trajectory of an agent. The trajectory contains a sequence of Action and Observations, describing how the agent solve a question by thinking, acting and observing. Here are some examples.

=== EXAMPLE 1 ===
Trajectory:
Question: Which magazine was started first Arthur's Magazine or First for Women?
Thought 1: I need to search Arthur's Magazine and First for Women, and find which was started first.
Action 1: Search[Arthur's Magazine]
Observation 1: Arthur's Magazine (1844-1846) was an American literary periodical published in Philadelphia.
Thought 2: Arthur's Magazine was started in 1844. I need to search First for Women next.
Action 2: Search[First for Women]
Observation 2: First for Women is a woman's magazine published by Bauer Media Group in the USA. The magazine was started in 1989.
Thought 3: 1844 < 1989, so Arthur's Magazine was started first.
Action 3: Finish[Arthur's Magazine]

Transferable Knowledge: Search two entities separately to find a shared attribute, then compare those attributes to determine which one satisfies the condition.

=== EXAMPLE 2 ===
Trajectory:
Question: Were Pavel Urysohn and Leonid Levin known for the same type of work?
Thought 1: I need to search Pavel Urysohn and Leonid Levin, find their types of work, then find if they are the same.
Action 1: Search[Pavel Urysohn]
Observation 1: Pavel Samuilovich Urysohn was a Soviet mathematician who is best known for his contributions in dimension theory.
Thought 2: Pavel Urysohn is a mathematician. I need to search Leonid Levin next.
Action 2: Search[Leonid Levin]
Observation 2: Leonid Anatolievich Levin is a Soviet-American mathematician and computer scientist.
Thought 3: Both are mathematicians, so they have the same type of work.
Action 3: Finish[yes]

Transferable Knowledge: Search two entities separately to find a shared attribute, then determine if those attributes are the same.

=== EXAMPLE 3 ===
Trajectory:
Question: What government position was held by the woman who used the pen name combiz?
Thought 1: I need to find who used the pen name combiz.
Action 1: Search[combiz pen name]
Observation 1: Faezeh Hashemi is an Iranian politician and women's rights advocate who used the pen name combiz.
Thought 2: Faezeh Hashemi used the pen name combiz. Now I need to find her government position.
Action 2: Search[Faezeh Hashemi]
Observation 2: Faezeh Hashemi served as a member of the Iranian parliament from 1996 to 2000.
Thought 3: She served as a member of the Iranian parliament.
Action 3: Finish[member of the Iranian parliament]

Transferable Knowledge: First search for the intermediate entity described by a given attribute, then search that entity to find the target attribute asked in the question.

=== END OF EXAMPLES ===

Now extract the transferable knowledge from the following trajectory. Output exactly ONE sentence about the abstract reasoning strategy. Do NOT repeat the examples above.

Trajectory:
{trajectory}

Transferable Knowledge:"""

def history_to_trajectory(question, history, final_answer):
    """将 agent 的 history 转为带步骤号的轨迹字符串，与 EXTRACTION_PROMPT 中示例格式一致"""
    traj = f"Question: {question}\n"
    for i, record in enumerate(history, 1):
        traj += f"Thought {i}: {record['thought']}\n"
        traj += f"Action {i}: {record['action']}\n"
        traj += f"Observation {i}: {record['observation']}\n"
    step_n = len(history) + 1
    traj += f"Thought {step_n}: I have sufficient information to answer.\n"
    traj += f"Action {step_n}: Finish[{final_answer}]\n"
    return traj


def build():
    # 功能点：初始化嵌入模型，用于将提取出的 TK 文本向量化
    print(f"初始化Embedding模型")
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    # 功能点：连接向量库，若不存在则创建；追加写入，不清空已有数据
    print(f"初始化向量数据库(ChromaDB)")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    # 功能点：加载 HotpotQA 指定 split，按 LEVELS 筛选难度，固定 seed 打乱，再从 START_INDEX 截断
    print(f"加载HotpotQA数据集")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, trust_remote_code=True)
    if LEVELS:
        filtered_dataset = dataset.filter(lambda x: x['level'] in LEVELS)
        print(f"筛选难度 {LEVELS} 后: {len(filtered_dataset)}")
    else:
        filtered_dataset = dataset
    filtered_dataset = filtered_dataset.shuffle(seed=42)
    if START_INDEX > 0:
        filtered_dataset = filtered_dataset.select(range(START_INDEX, len(filtered_dataset)))

    # 功能点：建库时用 agent 跑题且 0 示例（use_tk=False, use_demos=False），只收集答对的轨迹
    agent = DICEAgent()

    print(f"开始构建知识库（0 示例刷题 + 检索模型提取 TK）")

    # 功能点：读取库中已有 id 集合，避免同一题重复写入
    existing = set(collection.get(include=[]).get("ids", []))
    added_count = 0

    for i, item in enumerate(tqdm(filtered_dataset, desc="Processing")):
        if added_count >= TARGET_SIZE:
            break

        # 功能点：用 0 示例模式跑题，仅当答对且与标准答案 EM 一致才进入后续流程
        question = item["question"]
        gold_answer = item["answer"]
        pred = agent.run_task(question, max_steps=MAX_STEPS, use_tk=False, use_demos=False)
        if not pred or not exact_match(pred, gold_answer):
            continue
        # 功能点：将本条的 history + 最终答案 转成轨迹字符串，供 TK 提取 prompt 使用
        raw_traj = history_to_trajectory(question, agent.history, pred)
        # 功能点：用检索模型（如 Gemma）根据 EXTRACTION_PROMPT 从轨迹中提取一条 TK 文本
        try:
            res = ollama.generate(
                model=RETRIEVER_LLM,
                prompt=EXTRACTION_PROMPT.format(trajectory=raw_traj),
                options={'temperature': 0}
            )
            tk = res['response'].strip()
        except Exception as e:
            print(f"Error extracting knowledge for item {i}: {e}")
            continue

        # 功能点：将 TK 文本向量化，得到用于检索的 embedding
        vec = encoder.encode(tk).tolist()
        item_id = str(item['id'])
        if item_id in existing:
            continue

        # 功能点：单条写入库（documents=TK 文本, embeddings=向量, metadatas=原始轨迹），便于中断不丢数据
        collection.add(
            documents=[tk],
            embeddings=[vec],
            ids=[item_id],
            metadatas=[{"raw_trajectory": raw_traj}]
        )
        existing.add(item_id)
        added_count += 1
        print(f"已成功生成 {added_count} 条（尝试了 {i} 题）")

    if added_count == 0:
        print("⚠️ 警告：没有成功处理任何样本！请检查Ollama服务是否正常运行。")
        return

    print(f"🎉 知识库构建完成！共 {added_count} 条记录。")

if __name__ == "__main__":
    build()
