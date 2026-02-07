import json
from datasets import load_dataset

def inspect_hotpotqa():
    print("正在通过 Hugging Face下载HotpotQA 数据集...")
    # split='train[:5]' 表示我们只流式加载前5条，速度极快，不会占满内存
    dataset = load_dataset("hotpot_qa", "distractor", split="train[:5]", trust_remote_code=True)
    
    print("\n数据集加载成功")
    print(f"数据对象类型: {type(dataset)}")
    
    # 取出第一条样本
    sample = dataset[1]
    
    # 使用 json.dumps 漂亮地打印字典结构
    print(json.dumps(sample, indent=2, ensure_ascii=False))

    
    print(f"1. [id]: {sample['id']} (唯一标识符)")
    print(f"2. [question]: {sample['question']} (用户的问题)")
    print(f"3. [answer]: {sample['answer']} (标准答案)")
    print(f"4. [supporting_facts]: 这是一个列表，包含了推理依据。")
    print(f"   DICE 需要把这些 facts 转化成 Agent 的 'Observation'。")
    print(f"5. [context]: 包含了所有段落全文（包括干扰项），我们暂时不需要处理这个，太长了。")

if __name__ == "__main__":
    inspect_hotpotqa()