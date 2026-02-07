import time
import ollama
from sentence_transformers import SentenceTransformer


def print_memory_warning():
    print("进行内存压力测试...")
    time.sleep(2)


def test_embedding():
    print("\n[1/3] 加载 Embedding 模型 (MiniLM)...")
    # 这会下载约 80MB 的模型文件
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    vec = encoder.encode("Test sentence")
    print(f"Embedding 成功，向量维度: {len(vec)}")
    return encoder


def test_ollama_models():
    print("\n[2/3] 测试 Gemma-2B (知识提取器)...")
    # 这一步会把 gemma 加载进内存
    res = ollama.generate(model='gemma:2b', prompt='Hello, are you ready?')
    print(f"Gemma 回复: {res['response'][:20]}...")

    print("\n[3/3] 测试 Llama-3.1-8B (主 Agent)...")
    # 关键点！这一步会尝试把 Llama 也挤进内存
    # 如果内存不够，Ollama 可能会卸载 Gemma，或者系统开始 Swap
    start_time = time.time()
    res = ollama.generate(model='llama3.1:8b', prompt='Are you ready to work with Gemma?')
    end_time = time.time()

    print(f"Llama 回复: {res['response'][:20]}...")
    print(f"Llama 响应耗时: {end_time - start_time:.2f}s")

    if end_time - start_time > 10:
        print("响应时间过长，可能发生了内存交换 (Swap)")
    else:
        print("内存够用。")


if __name__ == "__main__":
    print_memory_warning()
    test_embedding()
    test_ollama_models()
    print("\n环境检查通过")