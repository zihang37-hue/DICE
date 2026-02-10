import json
from pathlib import Path

from datasets import load_dataset

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
INSPECT_CFG = get_required(CONFIG, "inspect_data")
DATASET_NAME = get_required(INSPECT_CFG, "dataset_name")
DATASET_CONFIG = get_required(INSPECT_CFG, "dataset_config")
DATASET_SPLIT = get_required(INSPECT_CFG, "split")
SAMPLE_INDEX = int(get_required(INSPECT_CFG, "sample_index"))

def inspect_hotpotqa():
    print("正在通过 Hugging Face下载HotpotQA 数据集...")
    # split='train[:5]' 表示我们只流式加载前5条，速度极快，不会占满内存
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=DATASET_SPLIT, trust_remote_code=True)
    
    print("\n数据集加载成功")
    print(f"数据对象类型: {type(dataset)}")
    
    # 取出第一条样本
    sample = dataset[SAMPLE_INDEX]
    
    # 使用 json.dumps 漂亮地打印字典结构
    print(json.dumps(sample, indent=2, ensure_ascii=False))

    
    print(f"1. [id]: {sample['id']} (唯一标识符)")
    print(f"2. [question]: {sample['question']} (用户的问题)")
    print(f"3. [answer]: {sample['answer']} (标准答案)")
   
if __name__ == "__main__":
    inspect_hotpotqa()
