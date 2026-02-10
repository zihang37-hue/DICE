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
        print(f"正在初始化agent")  # 初始化提示

        self.chroma_client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.chroma_client.get_collection(COLLECTION_NAME)

        self.encoder = SentenceTransformer(EMBEDDING_MODEL)

        self.history = []
        self._baseline_demo_text = ""  # baseline 每题固定一组示例，整题复用

    def _sample_random_demos(self, k=None):
        """从向量库随机抽取示例轨迹（baseline 用）"""
        if k is None:
            k = BASELINE_RANDOM_K
        results = self.collection.get(include=["metadatas"])
        metadatas = results.get("metadatas") or []
        if not metadatas:
            return []
        k = min(k, len(metadatas))
        sampled = random.sample(metadatas, k)
        return [m.get("raw_trajectory", "") for m in sampled if m.get("raw_trajectory")]

    def retrieve_demos(self, task, current_history_str, top_k=None):
        if top_k is None:
            top_k = DICE_TOP_K
        input_text = PREDICTION_PROMPT.format(task=task, history=current_history_str)
        predict_tk = call_llm(RETRIEVER_LLM, input_text, temperature=0.0, max_tokens=200)
        if not predict_tk:
            predict_tk = "General reasoning strategy"

        query_vec = self.encoder.encode(predict_tk).tolist()

        results = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        retrieved_trajectories = []
        if results['metadatas']:
            for meta in results['metadatas'][0]:
                retrieved_trajectories.append(meta['raw_trajectory'])

        return retrieved_trajectories

    def step(self, task, step_n, retry=False, use_tk=True):
        history_str = ""
        for i, record in enumerate(self.history, 1):
            history_str += f"Thought {i}: {record['thought']}\n"
            history_str += f"Action {i}: {record['action']}\n"
            history_str += f"Observation {i}: {record['observation']}\n"

        if use_tk:
            demos = self.retrieve_demos(task, history_str)
            if demos:
                demos = [normalize_react_demo(clean_demo_text(d)) for d in demos]
            demo_text = "\n\n".join(demos) if demos else ""
        else:
            demo_text = self._baseline_demo_text

        full_prompt = REACT_SYSTEM_PROMPT.format(task=task, demonstrations=demo_text)
        full_prompt += history_str

        if retry:
            full_prompt += "Thought: Based on all the information I have gathered from previous Observations, I can now provide the answer.\nAction:"
        else:
            full_prompt += "Thought:"

        content = call_llm(MAIN_LLM, full_prompt, temperature=0.0, stop=["Observation:"], max_tokens=300)

        print(f"\n🔄 [Step {step_n}]{' (retry)' if retry else ''}")
        print(f"💭 Thought: {content}")

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

        if action is None:
            if not retry:
                print("⚠️ 模型未输出有效Action，尝试重试...")
                return self.step(task, step_n, retry=True, use_tk=use_tk)
            print("⚠️ 重试后仍无有效Action，尝试从历史Observation中提取答案")
            action = self._fallback_answer(task)

        if len(self.history) > 0:
            last_action = self.history[-1]['action']
            if action == last_action and "Search" in action:
                print(f"⚠️ [警告] 检测到重复搜索: {action}")
                return content, "REPEAT_ERROR"

        thought_text = extract_thought(content)
        return thought_text, action

    def _fallback_answer(self, task):
        """当模型无法输出有效Action时，尝试用一次简短调用从已有Observation中总结答案"""
        useful_obs = []
        for record in self.history:
            obs = record.get('observation', '')
            if obs and 'Error' not in obs and 'No Wikipedia' not in obs and '⚠️' not in obs:
                useful_obs.append(obs)

        if useful_obs:
            summary_prompt = f"""Based on the following search results, directly answer the question in a few words.

Question: {task}

Search Results:
{chr(10).join(useful_obs)}

Answer (just the answer, nothing else):"""
            answer = call_llm(MAIN_LLM, summary_prompt, temperature=0.0, max_tokens=50)
            if answer:
                answer = answer.split('\n')[0].strip()
                match = re.search(r'Finish\[([^\]]*)\]', answer)
                if match:
                    answer = match.group(1).strip()
                if answer.lower().startswith("action:"):
                    answer = answer.split(":", 1)[-1].strip()
                print(f"💡 从历史Observation中提取到答案: {answer}")
                return f"Finish[{answer}]"

        return "Finish[Unknown]"

    def run_task(self, task, max_steps=6, use_tk=True):
        """
        执行任务，带有完整的防重复和错误处理机制
        use_tk: True=使用DICE每步检索示例，False=baseline每题固定随机6示例
        """
        mode_tag = "DICE(+TK)" if use_tk else "Baseline(no TK)"
        print(f"\n{'='*60}\n📋 [{mode_tag}] 任务: {task}\n{'='*60}")
        self.history = []
        self._baseline_demo_text = ""
        if not use_tk:
            demos = self._sample_random_demos(k=BASELINE_RANDOM_K)
            if demos:
                demos = [normalize_react_demo(clean_demo_text(d)) for d in demos]
            self._baseline_demo_text = "\n\n".join(demos) if demos else ""

        search_tool = RobustWikipediaEnv(max_chars=OBSERVATION_MAX_CHARS)
        searched_queries = set()
        consecutive_failures = 0

        for i in range(max_steps):
            thought, action = self.step(task, i+1, use_tk=use_tk)

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

            observation = ""
            if "Search" in action:
                try:
                    match = re.search(r'Search\[([^\]]*)\]', action)
                    if match:
                        query = match.group(1).strip()
                    else:
                        query = action.split("Search[")[-1].split("]")[0].strip()

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

            self.history.append({
                "thought": thought,
                "action": action,
                "observation": observation
            })

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
