from pathlib import Path

# 功能点：项目根目录（dice 的上一级），用于拼接 config 与 db 路径
ROOT_DIR = Path(__file__).resolve().parents[1]


def load_config():
    """从项目根目录读取 config.yaml，解析为字典；缺文件或非 dict 时抛错或返回空 dict"""
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
    """按 keys 逐层取配置值，任一层缺失则抛 KeyError（严格模式，无默认值）"""
    cur = cfg
    path = []
    for key in keys:
        path.append(str(key))
        if not isinstance(cur, dict) or key not in cur:
            raise KeyError(f"Missing config: {'.'.join(path)}")
        cur = cur[key]
    return cur


# 功能点：启动时加载并解析 config，后续模块直接使用下列常量
CONFIG = load_config()
MODELS = get_required(CONFIG, "models")
AGENT_CFG = get_required(CONFIG, "agent")

# 功能点：向量库路径与集合名（建库、检索、实验共用）
DB_PATH = str(ROOT_DIR / get_required(CONFIG, "db_path"))
COLLECTION_NAME = get_required(CONFIG, "collection_name")

# 功能点：主推理模型 / TK 预测与提取模型 / 嵌入模型
MAIN_LLM = get_required(MODELS, "main_llm")
RETRIEVER_LLM = get_required(MODELS, "retriever_llm")
EMBEDDING_MODEL = get_required(MODELS, "embedding_model")

# 功能点：DICE 每步检索条数、baseline 每题随机示例数、单次 Observation 最大字符数
DICE_TOP_K = int(get_required(AGENT_CFG, "dice_top_k"))
BASELINE_RANDOM_K = int(get_required(AGENT_CFG, "baseline_random_k"))
OBSERVATION_MAX_CHARS = int(get_required(AGENT_CFG, "observation_max_chars"))
