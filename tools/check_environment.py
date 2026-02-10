import time
from pathlib import Path

import ollama
from sentence_transformers import SentenceTransformer

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
MODELS = get_required(CONFIG, "models")
EMBEDDING_MODEL = get_required(MODELS, "embedding_model")
RETRIEVER_LLM = get_required(MODELS, "retriever_llm")
MAIN_LLM = get_required(MODELS, "main_llm")


def print_memory_warning():
    print("进行内存压力测试...")
    time.sleep(2)


def test_embedding():
    print("\n[1/3] 加载 Embedding 模型 (MiniLM)...")
    # 这会下载约 80MB 的模型文件
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    vec = encoder.encode("Test sentence")
    print(f"Embedding 成功，向量维度: {len(vec)}")
    return encoder


def test_ollama_models():
    print("\n[2/3] 测试 检索器模型 (知识提取器)...")
    # 这一步会把检索器模型加载进内存
    res = ollama.generate(model=RETRIEVER_LLM, prompt='Hello, are you ready?')
    print(f"检索器 回复: {res['response'][:20]}...")

    print("\n[3/3] 测试 主推理模型 (主 Agent)...")
    # 关键点！这一步会尝试把主模型也挤进内存
    # 如果内存不够，Ollama 可能会卸载检索器模型，或者系统开始 Swap
    start_time = time.time()
    res = ollama.generate(model=MAIN_LLM, prompt='Are you ready to work with the retriever?')
    end_time = time.time()

    print(f"主模型 回复: {res['response'][:20]}...")
    print(f"主模型 响应耗时: {end_time - start_time:.2f}s")

    if end_time - start_time > 10:
        print("响应时间过长，可能发生了内存交换 (Swap)")
    else:
        print("内存够用。")


if __name__ == "__main__":
    print_memory_warning()
    test_embedding()
    test_ollama_models()
    print("\n环境检查通过")
