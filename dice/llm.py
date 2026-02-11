import ollama


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
