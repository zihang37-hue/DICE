import re
import random

import chromadb
from sentence_transformers import SentenceTransformer

from dice.config import DB_PATH, COLLECTION_NAME, EMBEDDING_MODEL, MAIN_LLM, RETRIEVER_LLM, DICE_TOP_K, BASELINE_RANDOM_K, OBSERVATION_MAX_CHARS
from dice.env import RobustWikipediaEnv
from dice.llm import call_llm
from dice.prompts import PREDICTION_PROMPT, REACT_SYSTEM_PROMPT
from dice.utils import extract_thought, clean_demo_text, normalize_react_demo


class DICEAgent:
    def __init__(self):
        # 连接向量数据库，获取指定 collection
        self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)
        # 初始化嵌入模型，用于将 TK 文本向量化
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        # 当前任务的推理历史（Thought / Action / Observation）
        self.history = []
        # baseline 模式下每题固定一组示例，整题复用
        self._baseline_demo_text = ""

    def _sample_random_demos(self, k=None):
        """从向量库随机抽取示例轨迹（baseline 用）"""
        if k is None:
            k = BASELINE_RANDOM_K
        # 取出库中所有记录的 metadatas，用于随机抽样
        results = self.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []
        if not metadatas:
            return []
        k = min(k, len(metadatas))
        sampled = random.sample(metadatas, k)
        # 从每条 metadata 中取出 raw_trajectory 作为 demo 文本
        return [m.get("raw_trajectory", "") for m in sampled if m.get("raw_trajectory")]

    def retrieve_demos(self, task, current_history_str, top_k=None):
        """DICE：根据当前状态预测所需知识 → 向量化 → 检索相似轨迹"""
        if top_k is None:
            top_k = DICE_TOP_K
        # 功能点：生成「预测当前步所需知识」的提示词（题目 + 已有轨迹）
        input_text = PREDICTION_PROMPT.format(task=task, history=current_history_str)
        # 功能点：调用检索模型预测「当前步需要的 TK」文本
        predict_tk = call_llm(RETRIEVER_LLM, input_text, temperature=0.0, max_tokens=200)
        if not predict_tk:
            predict_tk = "General reasoning strategy"
        # 功能点：将预测出的 TK 文本向量化，用于相似度检索
        query_vec = self.encoder.encode(predict_tk).tolist()
        # 功能点：在向量库中检索与 query_vec 最相似的 top_k 条记录
        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        # 从检索结果的 metadatas 中取出每条记录的 raw_trajectory 作为 demo
        retrieved_trajectories = []
        if results['metadatas']:
            for meta in results['metadatas'][0]:
                retrieved_trajectories.append(meta['raw_trajectory'])
        return retrieved_trajectories

    def step(self, task, step_n, retry=False, use_tk=True):
        # 功能点：将当前已有的 Thought/Action/Observation 拼成字符串，供 prompt 或 TK 预测使用
        history_str = ""
        for i, record in enumerate(self.history, 1):
            history_str += f"Thought {i}: {record['thought']}\n"
            history_str += f"Action {i}: {record['action']}\n"
            history_str += f"Observation {i}: {record['observation']}\n"

        # 功能点：确定本步使用的示例——DICE 每步检索 / baseline 用任务开始时抽好的固定示例
        if use_tk:
            demos = self.retrieve_demos(task, history_str)
            if demos:
                demos = [normalize_react_demo(clean_demo_text(d)) for d in demos]
            demo_text = "\n\n".join(demos) if demos else ""
        else:
            demo_text = self._baseline_demo_text

        # 功能点：组装主模型 prompt（系统说明 + 示例 + 当前历史），并补上本次要生成的 "Thought:"
        full_prompt = REACT_SYSTEM_PROMPT.format(task=task, demonstrations=demo_text)
        full_prompt += history_str
        if retry:
            full_prompt += "Thought: Based on all the information I have gathered from previous Observations, I can now provide the answer.\nAction:"
        else:
            full_prompt += "Thought:"

        # 功能点：调用主模型生成本步的 Thought（遇 "Observation:" 停止，避免模型自己写 Observation）
        content = call_llm(MAIN_LLM, full_prompt, temperature=0.0, stop=["Observation:"], max_tokens=300)

        print(f"\n🔄 [Step {step_n}]{' (retry)' if retry else ''}")
        print(f"💭 Thought: {content}")

        # 功能点：从模型输出中解析出本步的 Action（Search[...] 或 Finish[...]）
        action = None
        if "Action:" in content:
            action = content.split("Action:")[-1].strip().split("\n")[0]
        elif "Finish[" in content:
            match = re.search(r'Finish\[.*?\]', content)
            if match:
                action = match.group(0)
        elif "Search[" in content:
            match = re.search(r'Search\[.*?\]', content)
            if match:
                action = match.group(0)

        # 功能点：若未解析到有效 Action，先重试一次；仍无则用 fallback 从历史 Observation 总结答案
        if action is None:
            if not retry:
                print("⚠️ 模型未输出有效Action，尝试重试...")
                return self.step(task, step_n, retry=True, use_tk=use_tk)
            print("⚠️ 重试后仍无有效Action，尝试从历史Observation中提取答案")
            action = self._fallback_answer(task)

        # 功能点：若本步是 Search 且与上一步相同，视为重复搜索，返回特殊标记由 run_task 处理
        if len(self.history) > 0:
            last_action = self.history[-1]['action']
            if action == last_action and "Search" in action:
                print(f"⚠️ [警告] 检测到重复搜索: {action}")
                return content, "REPEAT_ERROR"

        thought_text = extract_thought(content)
        return thought_text, action

    def _fallback_answer(self, task):
        """当模型无法输出有效 Action 时，用已有 Observation 做一次总结式调用得到答案"""
        # 功能点：筛出历史中「非错误、非警告」的 Observation，作为总结的输入
        useful_obs = []
        for record in self.history:
            obs = record.get('observation', '')
            if obs and 'Error' not in obs and 'No Wikipedia' not in obs and '⚠️' not in obs:
                useful_obs.append(obs)

        if useful_obs:
            # 功能点：构造「根据检索结果直接回答」的短 prompt
            summary_prompt = f"""Based on the following search results, directly answer the question in a few words.

Question: {task}

Search Results:
{chr(10).join(useful_obs)}

Answer (just the answer, nothing else):"""
            answer = call_llm(MAIN_LLM, summary_prompt, temperature=0.0, max_tokens=50)
            if answer:
                answer = answer.split('\n')[0].strip()
                # 功能点：若模型仍输出 Finish[...]，从括号内取出答案文本
                match = re.search(r'Finish\[([^\]]*)\]', answer)
                if match:
                    answer = match.group(1).strip()
                if answer.lower().startswith("action:"):
                    answer = answer.split(":", 1)[-1].strip()
                print(f"💡 从历史Observation中提取到答案: {answer}")
                return f"Finish[{answer}]"

        return "Finish[Unknown]"

    def run_task(self, task, max_steps=6, use_tk=True, use_demos=True):
        """
        执行任务，带有完整的防重复和错误处理机制
        use_tk: True=使用DICE每步检索示例，False=baseline每题固定随机6示例
        use_demos: True=使用示例(随机或DICE)，False=0示例(建库时用)
        """
        mode_tag = "DICE(+TK)" if use_tk else ("Baseline(no TK)" if use_demos else "Build(0 demos)")
        print(f"\n{'='*60}\n📋 [{mode_tag}] 任务: {task}\n{'='*60}")
        self.history = []
        self._baseline_demo_text = ""
        # 功能点：baseline 且使用示例时，在任务开始时从库中随机抽 k 条轨迹，整题复用
        if not use_tk and use_demos:
            demos = self._sample_random_demos(k=BASELINE_RANDOM_K)
            if demos:
                demos = [normalize_react_demo(clean_demo_text(d)) for d in demos]
            self._baseline_demo_text = "\n\n".join(demos) if demos else ""

        search_tool = RobustWikipediaEnv(max_chars=OBSERVATION_MAX_CHARS)
        searched_queries = set()
        consecutive_failures = 0

        for i in range(max_steps):
            # 功能点：执行一步推理（生成 Thought + Action），DICE 会在此步内做 TK 预测与检索
            thought, action = self.step(task, i+1, use_tk=use_tk)

            # 功能点：若 step 返回重复搜索标记，写入警告类 Observation 并视情况提前结束
            if action == "REPEAT_ERROR":
                print("🛑 检测到连续重复动作")
                observation = "⚠️ You are repeating the same search. Please try a DIFFERENT keyword or use Finish[...] with only the final answer inside brackets."
                self.history.append({
                    "thought": "Repeating previous action",
                    "action": self.history[-1]['action'] if self.history else "Search[Unknown]",
                    "observation": observation
                })
                consecutive_failures += 1

                if consecutive_failures >= 2:
                    print("❌ 连续重复次数过多，强制结束任务")
                    return None
                continue

            print(f"🎯 Action: {action}")

            # 功能点：若本步为 Finish，从括号内解析最终答案并返回，任务结束
            if "Finish" in action:
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

            # 功能点：本步为 Search 时，解析搜索词、去重、调 Wikipedia 环境得到 Observation
            observation = ""
            if "Search" in action:
                try:
                    match = re.search(r'Search\[([^\]]*)\]', action)
                    if match:
                        query = match.group(1).strip()
                    else:
                        query = action.split("Search[")[-1].split("]")[0].strip()
                    # 功能点：搜索词含逗号时做简单清洗（去逗号、合并空格）
                    if ',' in query:
                        original_query = query
                        query = query.replace(',', ' ').strip()
                        query = re.sub(r'\s+', ' ', query)
                        print(f"🔧 搜索词清理: '{original_query}' -> '{query}'")

                    if query in searched_queries:
                        print(f"⚠️ 检测到重复搜索query: '{query}'")
                        observation = f"⚠️ You already searched for '{query}' before! Results were shown in previous Observations. Please try a DIFFERENT entity or use Finish[...] with only the final answer inside brackets."
                        consecutive_failures += 1
                    else:
                        searched_queries.add(query)
                        observation = search_tool.search(query)
                        # 功能点：根据 Observation 内容判断是否搜索失败/网络错误，并决定是否追加提示
                        if "No Wikipedia page found" in observation or "Could not find" in observation or "does not exist" in observation:
                            consecutive_failures += 1
                            print(f"⚠️ 搜索失败 ({consecutive_failures}次)")
                        elif "Connection Error" in observation:
                            consecutive_failures += 1
                            print(f"⚠️ 网络错误 ({consecutive_failures}次)")
                        else:
                            consecutive_failures = 0
                            observation += f"\n(If this answers the question '{task}', use Action: Finish[...] with only the final answer inside brackets.)"

                    if consecutive_failures >= 3:
                        print("⚠️ 连续3次搜索失败，提示模型使用已有知识")
                        observation += "\n\n💡 HINT: You have tried multiple searches without success. Please use Action: Finish[...] with only the final answer inside brackets."

                except Exception as e:
                    observation = f"Search error: {e}"
                    consecutive_failures += 1
            else:
                observation = "Invalid action format. Use Search[entity] or Finish[...]."

            print(f"👁️ Observation: {observation[:150]}{'...' if len(observation) > 150 else ''}")

            # 功能点：将本步的 thought / action / observation 追加到 history，供下一步使用
            self.history.append({
                "thought": thought,
                "action": action,
                "observation": observation
            })

        # 功能点：达到最大步数仍未 Finish 时，用 fallback 从历史 Observation 总结答案；若仍无则返回 None
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
