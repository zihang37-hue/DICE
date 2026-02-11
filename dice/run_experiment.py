import sys
from pathlib import Path

# 功能点：保证项目根在 sys.path 中，以便 import dice / tools
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import chromadb
from datasets import load_dataset

from dice.agent import DICEAgent
from dice.config import MAIN_LLM, RETRIEVER_LLM, DB_PATH, COLLECTION_NAME, load_config, get_required
from dice.utils import exact_match
from tools.experiment_results import write_experiment_results


def main():
    # 功能点：从 config 读取实验与 agent 相关配置
    config = load_config()
    exp_cfg = get_required(config, "experiment")
    agent_cfg = get_required(config, "agent")
    build_cfg = get_required(config, "build_pool")
    models_cfg = get_required(config, "models")

    num_test = int(get_required(exp_cfg, "num_test"))
    max_steps = int(get_required(exp_cfg, "max_steps"))
    difficulty = get_required(exp_cfg, "difficulty")
    seeds = get_required(exp_cfg, "seeds")
    dataset_name = get_required(exp_cfg, "dataset_name")
    dataset_config = get_required(exp_cfg, "dataset_config")
    split = get_required(exp_cfg, "split")

    # 功能点：连接向量库获取当前条数，用于写入实验配置摘要（失败则记为 None）
    try:
        client = chromadb.PersistentClient(path=DB_PATH)
        collection = client.get_collection(COLLECTION_NAME)
        vector_db_count = collection.count()
    except Exception as e:
        vector_db_count = None

    config_summary = {
        "main_llm": get_required(models_cfg, "main_llm"),
        "retriever_llm": get_required(models_cfg, "retriever_llm"),
        "embedding_model": get_required(models_cfg, "embedding_model"),
        "vector_db_count": vector_db_count,
        "vector_db_difficulty": get_required(build_cfg, "levels"),
        "dice_top_k": int(get_required(agent_cfg, "dice_top_k")),
        "baseline_random_k": int(get_required(agent_cfg, "baseline_random_k")),
        "observation_max_chars": int(get_required(agent_cfg, "observation_max_chars")),
        "num_test": num_test,
        "max_steps": max_steps,
        "test_difficulty": difficulty,
        "seeds": seeds,
    }

    # 功能点：加载验证集，并按 difficulty 筛选（若配置了难度）
    print("📥 正在加载 HotpotQA 验证集...")
    val_dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)

    if difficulty:
        val_dataset = val_dataset.filter(lambda x: x['level'] == difficulty)
        print(f"🔍 筛选 [{difficulty}] 难度: 共 {len(val_dataset)} 题")

    seed_summaries = []
    for seed in seeds:
        # 功能点：按当前 seed 打乱验证集并取前 num_test 条作为本轮测试集
        if len(val_dataset) > num_test:
            test_dataset = val_dataset.shuffle(seed=seed).select(range(num_test))
        else:
            test_dataset = val_dataset
        print(f"\n🧪 随机种子: {seed} | 本次测试: {len(test_dataset)} 题\n")

        agent = DICEAgent()

        dice_correct = 0
        base_correct = 0
        results = []

        for i, item in enumerate(test_dataset):
            question = item['question']
            gold_answer = item['answer']
            print(f"\n{'#'*60}")
            print(f"# 第 {i+1}/{len(test_dataset)} 题  |  标准答案: {gold_answer}")
            print(f"{'#'*60}")

            # 功能点：同一题先跑 DICE（use_tk=True，每步检索示例），再跑 baseline（use_tk=False，每题随机 6 示例）
            ans_tk = agent.run_task(question, max_steps=max_steps, use_tk=True)
            tk_match = exact_match(ans_tk, gold_answer) if ans_tk else False
            if tk_match:
                dice_correct += 1

            ans_no = agent.run_task(question, max_steps=max_steps, use_tk=False)
            no_match = exact_match(ans_no, gold_answer) if ans_no else False
            if no_match:
                base_correct += 1

            # 功能点：记录本题的题目、标准答案、两种模式的答案与 EM 结果，用于汇总与写入
            record = {
                "id": item.get("id") if isinstance(item, dict) else None,
                "question": question,
                "gold": gold_answer,
                "dice_answer": ans_tk,
                "dice_em": tk_match,
                "base_answer": ans_no,
                "base_em": no_match,
            }
            results.append(record)

            tested = i + 1
            print(f"\n📊 进度 [{tested}/{len(test_dataset)}]  "
                  f"DICE: {dice_correct}/{tested} ({dice_correct/tested*100:.1f}%)  "
                  f"Baseline: {base_correct}/{tested} ({base_correct/tested*100:.1f}%)")

        total = len(results)
        print(f"\n\n{'='*70}")
        print(f"{'实验结果汇总':^70}")
        print(f"{'='*70}")
        print(f"  随机种子:    {seed}")
        print(f"  模型:        {MAIN_LLM}")
        print(f"  检索器:      {RETRIEVER_LLM}")
        print(f"  测试题数:    {total} ({difficulty or '全部'} 难度)")
        print(f"  最大步数:    {max_steps}")
        print(f"{'─'*70}")
        print(f"  DICE (+TK)   EM: {dice_correct}/{total} = {dice_correct/total*100:.1f}%")
        print(f"  Baseline     EM: {base_correct}/{total} = {base_correct/total*100:.1f}%")
        print(f"  差值:        {(dice_correct - base_correct)/total*100:+.1f}%")
        print(f"{'='*70}")
        seed_summaries.append((seed, dice_correct, base_correct, total))

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

    print(f"\n{'='*70}")
    print(f"{'随机种子结果汇总':^70}")
    print(f"{'='*70}")
    print(f"{'Seed':<6} {'DICE':<15} {'Baseline':<15} {'Diff':<8}")
    print(f"{'-'*70}")
    for seed, dice_c, base_c, total in seed_summaries:
        diff = dice_c - base_c
        diff_str = f"{diff:+d}"
        print(f"{seed:<6} {dice_c}/{total:<12} {base_c}/{total:<12} {diff_str:<8}")
    print(f"{'='*70}")

    # 功能点：汇总所有 seed 的总题数、DICE 对题数、baseline 对题数，写入 results 目录的 JSON
    total_tests = sum(t for _, _, _, t in seed_summaries)
    dice_total_correct = sum(d for _, d, _, _ in seed_summaries)
    baseline_total_correct = sum(b for _, _, b, _ in seed_summaries)
    results_summary = {
        "total_tests": total_tests,
        "dice_correct": dice_total_correct,
        "baseline_correct": baseline_total_correct,
    }
    output_path = write_experiment_results(
        config_summary, results_summary, output_dir=ROOT_DIR / "results"
    )
    print(f"\n📁 实验结果已写入: {output_path}")


if __name__ == "__main__":
    main()
