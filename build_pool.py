import json
import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import ollama
from tqdm import tqdm

TARGET_SIZE = 2000
DB_PATH = "./dice_vector_db"
COLLECTION_NAME = "hotpotqa_pool"

EXTRACTION_PROMPT = """Extract the abstract reasoning strategy from the trajectory below.

Rules:
- Do NOT mention any specific names, dates, places, or entities from the trajectory.
- Use generic terms: Entity A, Entity B, Subject, Attribute, Target.
- Follow the output format exactly: Type, Steps, Strategy.

Example 1 (comparison):
Trajectory:
Task: Which magazine was published first, X or Y?
Thought: I need to search for information about specific entities to compare or connect them.
Action: Search[X, Y]
Observation: I found the following information:
X: X is a magazine first published in 1844.
Y: Y is a magazine first published in 1989.
Thought: I have sufficient information to answer.
Action: Finish[X]

Output:
Type: comparison
Steps: 1) Search both Entity A and Entity B. 2) Extract the same Attribute from each. 3) Compare values and finish.
Strategy: Retrieve a shared attribute from two entities and compare them.

Example 2 (multi-hop):
Trajectory:
Task: Who is the mother of the founder of Company Z?
Thought: I need to search for information about specific entities to compare or connect them.
Action: Search[Company Z, Person A]
Observation: I found the following information:
Company Z: Company Z was founded by Person A.
Person A: Person A was born in 1964. His mother is Person B.
Thought: I have sufficient information to answer.
Action: Finish[Person B]

Output:
Type: multi-hop
Steps: 1) Search the main Entity to identify an intermediate Subject. 2) Find the Target attribute of that Subject. 3) Finish with the Target.
Strategy: Chain through an intermediate entity to reach the requested relationship.

Example 3 (bridge):
Trajectory:
Task: What city is the football club from that plays in League X?
Thought: I need to search for information about specific entities to compare or connect them.
Action: Search[League X, Club A]
Observation: I found the following information:
Club A: Club A is a football club based in City M that plays in League X.
League X: League X is a professional football league.
Thought: I have sufficient information to answer.
Action: Finish[City M]

Output:
Type: bridge
Steps: 1) Search Entity A to find its link to Entity B. 2) Extract the bridging Attribute from the connection. 3) Finish.
Strategy: Find a connecting relationship between two entities through shared context.

Example 4 (lookup):
Trajectory:
Task: What nationality is Person P?
Thought: I need to search for information about specific entities to compare or connect them.
Action: Search[Person P]
Observation: I found the following information:
Person P: Person P (born 1950) is a French novelist and playwright.
Thought: I have sufficient information to answer.
Action: Finish[French]

Output:
Type: lookup
Steps: 1) Search the Entity directly. 2) Extract the requested Attribute from results. 3) Finish.
Strategy: Direct search and single-attribute extraction.

Now extract from this trajectory:
{trajectory}

Output:"""

def format_trajectory(sample):
    """
    把原始数据(Q+A+Facts)转换成ReAct的轨迹
    """
    question = sample["question"]
    answer = sample["answer"]
    # 提取支撑事实的标题，支撑事实：证明agent找到的信息是正确的证据
    target_titles = []
    if 'supporting_facts' in sample:
        facts_data = sample['supporting_facts']
        if isinstance(facts_data, dict) and 'title' in facts_data:
            target_titles = list(set(facts_data['title']))
        elif isinstance(facts_data, list):
            target_titles = list(set([item[0] for item in facts_data]))
    
    # 构造轨迹字符串（注意Action:后面有空格，保持格式统一）
    traj = f"Task: {question}\n"
    traj += f"Thought: I need to search for information about specific entities to compare or connect them.\n"
    if target_titles:
        traj += f"Action: Search[{', '.join(target_titles)}]\n"

        real_context = ""
        # 修复：从sample中获取context，而不是从facts_data
        context_data = sample.get('context', {})

        if context_data:
            iterator = []
            if isinstance(context_data, dict):
                titles = context_data.get('title', [])
                # 修复：字段名是'sentences'（复数）
                sentences = context_data.get('sentences', [])
                iterator = zip(titles, sentences)
            elif isinstance(context_data, list):
                iterator = context_data
            
            # item的数据结构[title, sentences]
            for item in iterator:
                title = item[0]
                sentences = item[1]
                # 修复：使用正确的变量名target_titles
                if title in target_titles:
                    # 把前两句话拼接起来，用空格连接更自然
                    snippet = " ".join(sentences[:2])
                    real_context += f"{title}: {snippet}\n"
        if real_context:
            traj += f"Observation: I found the following information:\n{real_context}\n"
        else:
            real_context = "No relevant information found."
            traj += f"Observation: {real_context}\n"
    traj += f"Thought: I have sufficient information to answer.\n"
    traj += f"Action: Finish[{answer}]\n"

    return traj


def build():
    print(f"初始化Embedding模型")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"初始化向量数据库(ChromaDB)")
    # 将数据存储到DB_PATH这个文件中
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION_NAME) # 清空旧数据
    except:
        pass
    collection = client.create_collection(COLLECTION_NAME)
    print(f"加载HotpotQA数据集")
    dataset = load_dataset("hotpot_qa", "distractor", split=f"train", trust_remote_code=True)
    # 筛选出中等和难题
    filtered_dataset = dataset.filter(lambda x: x['level'] in ['hard', 'medium'])
    if len(filtered_dataset) > TARGET_SIZE:
        sampled_dataset = filtered_dataset.shuffle(seed=42).select(range(TARGET_SIZE))
    else:
        sampled_dataset = filtered_dataset
    print(f"Gemma开始构建知识库")

    ids = []
    embeddings = []
    metadatas = []
    documents = [] # 存储TK

    for i, item in enumerate(tqdm(sampled_dataset, desc="Processing")):
        # 造轨迹
        raw_traj = format_trajectory(item)
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
        ids.append(str(item['id']))
        embeddings.append(vec)
        metadatas.append({"raw_trajectory": raw_traj})
        documents.append(tk)
    
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
    