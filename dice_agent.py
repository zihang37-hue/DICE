import re
import ollama
import chromadb
from sentence_transformers import SentenceTransformer
import wikipedia

# 主推理模型（Ollama 本地），可改为 qwen2.5:7b等
MAIN_LLM = "qwen2.5:7b"
# 检索器（TK 预测）与 build_pool 中 TK 提取一致
RETRIEVER_LLM = "gemma:2b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DB_PATH = "./dice_vector_db"
COLLECTION_NAME = "hotpotqa_pool"


def call_llm(model, prompt, temperature=0.0, stop=None, max_tokens=500):
    """统一通过 Ollama 调用本地模型"""
    try:
        opts = {"temperature": temperature, "num_predict": max_tokens}
        if stop:
            opts["stop"] = stop
        res = ollama.generate(model=model, prompt=prompt, options=opts)
        return (res.get("response") or "").strip()
    except Exception as e:
        print(f"⚠️ Ollama 调用失败 ({model}): {e}")
        return ""

# [在线阶段 Prompt] 让 Gemma 预测下一步需要什么逻辑
PREDICTION_PROMPT = """[Instruction]
You are an expert in strategic reasoning.
Based on the current task and history, predict the **abstract reasoning logic** needed for the NEXT step.

[Constraints]
1. Do NOT try to answer the question.
2. Do NOT mention specific entity names (like "Titanic", "Obama"). Use terms like "Subject", "Entity", "Attribute".
3. The output must be a strategy description that matches the Transferable Knowledge in our database.

[Examples]

Input:
Task: Which film was released earlier, Titanic or Avatar?
History: []
Required Reasoning Strategy:
To solve this comparison problem, the agent should first identify the two entities involved. Then, it needs to initiate a search for the first entity to retrieve its specific attribute (release date).

Input:
Task: Who is the mother of the founder of Amazon?
History: 
Action: Search[Amazon founder]
Observation: Jeff Bezos is the founder of Amazon.
Required Reasoning Strategy:
Since the key subject (founder) has been identified from the search results, the agent should now perform a secondary search specifically targeting that subject to find the requested relationship (mother).

--------------------------------------------------

[Current Context]
Task: {task}
History:
{history}

[Output]
Required Reasoning Strategy:"""

# 主LLM的ReAct规则与格式
REACT_SYSTEM_PROMPT = """You are a smart agent answering questions using Wikipedia search.
You MUST follow the Thought-Action format strictly. Every response MUST contain exactly ONE "Action:" line.

RULES:
1. Search for ONE entity at a time (e.g., Search[France], NOT Search[France,capital]).
2. After each Observation, check if it contains the answer. If yes, immediately use Action: Finish[answer].
3. If a search fails or returns irrelevant results, try a DIFFERENT and SIMPLER keyword.
4. After 2-3 searches, you likely have enough information. Combine what you learned and use Action: Finish[answer].
5. The answer should be SHORT and DIRECT (e.g., "Paris", "George Washington", "Pacific Ocean").
6. NEVER output Finish[Unknown] if you have seen ANY useful information in previous Observations.
7. ALWAYS output an Action line. Your response MUST end with: Action: Search[entity] or Action: Finish[answer]

FORMAT (you MUST follow this exactly):
Thought: [your reasoning, referencing information from Observations]
Action: Search[entity] or Finish[answer]

Examples:
{demonstrations}

Now solve this task step by step.
Task: {task}
"""

# Wikipedia搜索环境封装
class RobustWikipediaEnv:
    def __init__(self):
        # 尝试设置代理（如果你有VPN，可以在这里设置，或者在终端export）
        # import os
        # os.environ["https_proxy"] = "http://127.0.0.1:7890" 
        wikipedia.set_lang("en")  # 设定英文维基
    
    def search(self, query):
        print(f"🌍 正在搜索 Wiki: {query}")  # 搜索入口
        try:
            # auto_suggest=True 能解决一部分 "Paris" 搜不到的问题
            results = wikipedia.search(query)
            if not results:
                return f"Observation: No Wikipedia page found for '{query}'. Try a different keyword."  # 无结果
            
            # 取第一个结果
            page_title = results[0]
            try:
                page = wikipedia.page(page_title, auto_suggest=False)
                # 只取 Summary 的前 3000 字符
                content = page.summary[:3000].replace("\n", " ")
                return f"Observation: [Title: {page_title}] {content}..."  # 返回摘要
            except wikipedia.exceptions.DisambiguationError as e:
                return f"Observation: Ambiguous term '{query}'. Options: {', '.join(e.options[:5])}"  # 歧义页
            except wikipedia.exceptions.PageError:
                return f"Observation: Page '{page_title}' does not exist."  # 页面不存在
                
        except Exception as e:
            return f"Observation: Wikipedia Connection Error: {str(e)}. (Check your network!)"  # 网络/请求错误

# DICE Agent主体逻辑
class DICEAgent:
    def __init__(self):
        print(f"正在初始化agent")  # 初始化提示

        # 连接数据库
        self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)

        # 加载embedding模型
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)

        # 初始化历史记录
        self.history = []
    
    def retrieve_demos(self, task, current_history_str, top_k=3):
        # 用检索模型预测“需要的推理策略”
        input_text = PREDICTION_PROMPT.format(task=task, history=current_history_str)
        predict_tk = call_llm(RETRIEVER_LLM, input_text, temperature=0.0, max_tokens=200)
        if not predict_tk:
            predict_tk = "General reasoning strategy"
        
        # 将预测的可迁移知识转换为向量
        query_vec = self.encoder.encode(predict_tk).tolist()

        # 在向量数据库中搜索最相似的向量
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # 提取原始轨迹
        retrieved_trajectories = []
        if results['metadatas']:
            for meta in results['metadatas'][0]:
                retrieved_trajectories.append(meta['raw_trajectory'])
        
        return retrieved_trajectories  # 返回示例轨迹

    def step(self, task, step_n, retry=False, use_tk=True):
        # 拼接历史轨迹作为上下文
        history_str = ""
        for record in self.history:
            history_str += f"Thought: {record['thought']}\n"
            history_str += f"Action: {record['action']}\n"
            history_str += f"Observation: {record['observation']}\n"
        
        # 检索示例轨迹并拼接到提示词（use_tk=False 时跳过检索）
        if use_tk:
            demos = self.retrieve_demos(task, history_str)
            demo_text = "\n\n".join(demos) if demos else ""
        else:
            demo_text = ""

        # 组装主LLM提示词
        full_prompt = REACT_SYSTEM_PROMPT.format(task=task, demonstrations=demo_text)
        full_prompt += history_str

        if retry:
            # 重试时，强制要求模型输出 Action
            full_prompt += "Thought: Based on all the information I have gathered from previous Observations, I can now provide the answer.\nAction:"
        else:
            full_prompt += "Thought:"

        # 调用主LLM生成下一步
        content = call_llm(MAIN_LLM, full_prompt, temperature=0.0, stop=["Observation:"], max_tokens=300)

        print(f"\n🔄 [Step {step_n}]{' (retry)' if retry else ''}")
        print(f"💭 Thought: {content}")
        
        # === 提取 Action ===
        action = None
        if "Action:" in content:
            action = content.split("Action:")[-1].strip().split("\n")[0]
        elif "Finish[" in content:
            # 兼容模型有时候忘了写 Action: 前缀
            match = re.search(r'Finish\[.*?\]', content)
            if match: action = match.group(0)
        elif "Search[" in content:
            # 兼容模型忘了写 Action: 但写了 Search[...]
            match = re.search(r'Search\[.*?\]', content)
            if match: action = match.group(0)
        
        # 如果仍然没有提取到 Action
        if action is None:
            if not retry:
                # 第一次没提取到 Action，给模型一次重试机会
                print("⚠️ 模型未输出有效Action，尝试重试...")
                return self.step(task, step_n, retry=True, use_tk=use_tk)
            else:
                # 重试后仍然没有 Action，尝试从历史 Observation 中提取答案
                print("⚠️ 重试后仍无有效Action，尝试从历史Observation中提取答案")
                action = self._fallback_answer(task)

        # 如果上一步也是这个动作，说明模型卡住了
        if len(self.history) > 0:
            last_action = self.history[-1]['action']
            if action == last_action and "Search" in action:
                print(f"⚠️ [警告] 检测到重复搜索: {action}")
                return content, "REPEAT_ERROR"

        return content, action  # 返回本步Thought与Action

    def _fallback_answer(self, task):
        """当模型无法输出有效Action时，尝试用一次简短调用从已有Observation中总结答案"""
        # 收集所有成功的 Observation
        useful_obs = []
        for record in self.history:
            obs = record.get('observation', '')
            if obs and 'Error' not in obs and 'No Wikipedia' not in obs and '⚠️' not in obs:
                useful_obs.append(obs)
        
        if useful_obs:
            # 使用主LLM从历史Observation中抽取答案
            summary_prompt = f"""Based on the following search results, directly answer the question in a few words.

Question: {task}

Search Results:
{chr(10).join(useful_obs)}

Answer (just the answer, nothing else):"""
            answer = call_llm(MAIN_LLM, summary_prompt, temperature=0.0, max_tokens=50)
            if answer:
                answer = answer.split('\n')[0].strip()
                print(f"💡 从历史Observation中提取到答案: {answer}")
                return f"Finish[{answer}]"
        
        return "Finish[Unknown]"  # 兜底

    def run_task(self, task, max_steps=6, use_tk=True):
        """
        执行任务，带有完整的防重复和错误处理机制
        use_tk: True=使用DICE检索TK示例，False=零示例baseline
        """
        mode_tag = "DICE(+TK)" if use_tk else "Baseline(no TK)"
        print(f"\n{'='*60}\n📋 [{mode_tag}] 任务: {task}\n{'='*60}")
        self.history = []
        search_tool = RobustWikipediaEnv()
        
        # 防重复机制
        searched_queries = set()  # 记录所有搜过的query
        consecutive_failures = 0  # 连续失败次数

        for i in range(max_steps):
            # 生成下一步动作
            thought, action = self.step(task, i+1, use_tk=use_tk)
            
            # 处理检测到的连续重复（step方法返回的REPEAT_ERROR）
            if action == "REPEAT_ERROR":
                print("🛑 检测到连续重复动作")
                # 强制给一个提示，让模型改变策略
                observation = "⚠️ You are repeating the same search. Please try a DIFFERENT keyword or use Finish[answer] with your best knowledge."
                self.history.append({
                    "thought": "Repeating previous action",
                    "action": self.history[-1]['action'] if self.history else "Search[Unknown]",
                    "observation": observation
                })
                consecutive_failures += 1
                
                # 如果连续重复太多次，强制结束
                if consecutive_failures >= 2:
                    print("❌ 连续重复次数过多，强制结束任务")
                    return None
                continue

            print(f"🎯 Action: {action}")

            # 检查是否完成
            if "Finish" in action:
                # 提取答案
                match = re.search(r'Finish\[([^\]]*)\]', action)
                if match:
                    final_answer = match.group(1)
                else:
                    final_answer = action.split("Finish[")[-1].replace("]", "")
                
                print(f"\n{'='*60}")
                print(f"✅ 任务完成!")
                print(f"🏆 最终答案: {final_answer}")
                print(f"📊 总步数: {i+1}")
                print(f"{'='*60}")
                return final_answer
            
            # 执行Search动作
            observation = ""
            if "Search" in action:
                try:
                    # 提取搜索query
                    match = re.search(r'Search\[([^\]]*)\]', action)
                    if match:
                        query = match.group(1).strip()
                    else:
                        query = action.split("Search[")[-1].split("]")[0].strip()
                    
                    # 清理搜索词：将逗号替换为空格，保留所有实体
                    # 例如 "France,capital" -> "France capital"
                    # 例如 "President, United States" -> "President United States"
                    if ',' in query:
                        original_query = query
                        query = query.replace(',', ' ').strip()
                        # 去除多余空格
                        query = re.sub(r'\s+', ' ', query)
                        print(f"🔧 搜索词清理: '{original_query}' -> '{query}'")
                    
                    # 检查是否重复搜索（核心防御机制）
                    if query in searched_queries:
                        print(f"⚠️ 检测到重复搜索query: '{query}'")
                        observation = f"⚠️ You already searched for '{query}' before! Results were shown in previous Observations. Please try a DIFFERENT entity or use Finish[answer]."
                        consecutive_failures += 1
                    else:
                        # 执行新的搜索
                        searched_queries.add(query)
                        observation = search_tool.search(query)
                        
                        # 检查是否失败
                        if "No Wikipedia page found" in observation or "Could not find" in observation or "does not exist" in observation:
                            consecutive_failures += 1
                            print(f"⚠️ 搜索失败 ({consecutive_failures}次)")
                        elif "Connection Error" in observation:
                            # 网络错误不重置计数，但也不算内容失败
                            consecutive_failures += 1
                            print(f"⚠️ 网络错误 ({consecutive_failures}次)")
                        else:
                            consecutive_failures = 0  # 重置失败计数
                            # 搜索成功后追加提示，引导模型利用已有信息
                            observation += f"\n(If this answers the question '{task}', use Action: Finish[answer] now.)"
                    
                    # 连续失败太多次，给出强烈提示
                    if consecutive_failures >= 3:
                        print("⚠️ 连续3次搜索失败，提示模型使用已有知识")
                        observation += "\n\n💡 HINT: You have tried multiple searches without success. Please use Action: Finish[answer] with your best knowledge or information from previous Observations."
                        
                except Exception as e:
                    observation = f"Search error: {e}"
                    consecutive_failures += 1
            else:
                observation = "Invalid action format. Use Search[entity] or Finish[answer]."
            
            # 显示observation
            print(f"👁️ Observation: {observation[:150]}{'...' if len(observation) > 150 else ''}")
            
            # 保存历史
            self.history.append({
                "thought": thought,
                "action": action,
                "observation": observation
            })
        
        # 达到最大步数，尝试从历史Observation中提取答案
        fallback = self._fallback_answer(task)
        if fallback and fallback != "Finish[Unknown]":
            match = re.search(r'Finish\[([^\]]*)\]', fallback)
            if match:
                final_answer = match.group(1)
                print(f"\n{'='*60}")
                print(f"✅ 任务完成 (fallback)!")
                print(f"🏆 最终答案: {final_answer}")
                print(f"📊 总步数: {max_steps}")
                print(f"{'='*60}")
                return final_answer

        print(f"\n{'='*60}")
        print(f"❌ 任务失败: 达到最大步数 ({max_steps})")
        print(f"{'='*60}")
        return None

def normalize_answer(s):
    """标准化答案，用于 Exact Match 对比"""
    if not s:
        return ""
    s = s.lower().strip()
    # 去掉冠词
    for article in ["a ", "an ", "the "]:
        if s.startswith(article):
            s = s[len(article):]
    # 去掉标点
    s = re.sub(r'[^\w\s]', '', s)
    # 压缩空格
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def exact_match(pred, gold):
    """判断预测答案是否和标准答案匹配"""
    return normalize_answer(pred) == normalize_answer(gold)


if __name__ == "__main__":
    from datasets import load_dataset

    # ====== 配置 ======
    NUM_TEST = 50          # 测试题数量（可调整，建议 50-100）
    MAX_STEPS = 6          # 每题最大推理步数
    DIFFICULTY = None    # 筛选难度: "hard" / "medium" / 两者都要改为 None

    # 加载 HotpotQA 验证集
    print("📥 正在加载 HotpotQA 验证集...")
    val_dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)

    # 筛选指定难度
    if DIFFICULTY:
        val_dataset = val_dataset.filter(lambda x: x['level'] == DIFFICULTY)
        print(f"🔍 筛选 [{DIFFICULTY}] 难度: 共 {len(val_dataset)} 题")
    
    # 随机抽样（seed 不同于 build_pool，避免和训练集重叠无关但保证可复现）
    if len(val_dataset) > NUM_TEST:
        val_dataset = val_dataset.shuffle(seed=1).select(range(NUM_TEST))
    print(f"📝 本次测试: {len(val_dataset)} 题\n")

    # 初始化 Agent
    agent = DICEAgent()

    # 收集结果
    dice_correct = 0
    base_correct = 0
    results = []

    for i, item in enumerate(val_dataset):
        question = item['question']
        gold_answer = item['answer']
        print(f"\n{'#'*60}")
        print(f"# 第 {i+1}/{len(val_dataset)} 题  |  标准答案: {gold_answer}")
        print(f"{'#'*60}")

        # DICE 模式（有 TK）
        ans_tk = agent.run_task(question, max_steps=MAX_STEPS, use_tk=True)
        tk_match = exact_match(ans_tk, gold_answer) if ans_tk else False
        if tk_match:
            dice_correct += 1

        # Baseline 模式（无 TK）
        ans_no = agent.run_task(question, max_steps=MAX_STEPS, use_tk=False)
        no_match = exact_match(ans_no, gold_answer) if ans_no else False
        if no_match:
            base_correct += 1

        results.append({
            "question": question,
            "gold": gold_answer,
            "dice_answer": ans_tk,
            "dice_em": tk_match,
            "base_answer": ans_no,
            "base_em": no_match,
        })

        # 实时显示进度
        tested = i + 1
        print(f"\n📊 进度 [{tested}/{len(val_dataset)}]  "
              f"DICE: {dice_correct}/{tested} ({dice_correct/tested*100:.1f}%)  "
              f"Baseline: {base_correct}/{tested} ({base_correct/tested*100:.1f}%)")

    # ========== 最终汇总 ==========
    total = len(results)
    print(f"\n\n{'='*70}")
    print(f"{'实验结果汇总':^70}")
    print(f"{'='*70}")
    print(f"  模型:        {MAIN_LLM}")
    print(f"  检索器:      {RETRIEVER_LLM}")
    print(f"  测试题数:    {total} ({DIFFICULTY or '全部'} 难度)")
    print(f"  最大步数:    {MAX_STEPS}")
    print(f"{'─'*70}")
    print(f"  DICE (+TK)   EM: {dice_correct}/{total} = {dice_correct/total*100:.1f}%")
    print(f"  Baseline     EM: {base_correct}/{total} = {base_correct/total*100:.1f}%")
    print(f"  差值:        {(dice_correct - base_correct)/total*100:+.1f}%")
    print(f"{'='*70}")

    # 逐题明细
    print(f"\n{'─'*70}")
    print(f"{'逐题明细':^70}")
    print(f"{'─'*70}")
    for idx, r in enumerate(results, 1):
        tk_mark = "✅" if r['dice_em'] else "❌"
        no_mark = "✅" if r['base_em'] else "❌"
        print(f"{idx:>3}. Q: {r['question'][:55]}")
        print(f"     标准: {r['gold']}")
        print(f"     DICE: {r['dice_answer'] or '未回答':30} {tk_mark}")
        print(f"     Base: {r['base_answer'] or '未回答':30} {no_mark}")
        print()