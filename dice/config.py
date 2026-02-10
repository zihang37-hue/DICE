from pathlib import Path

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

DB_PATH = str(ROOT_DIR / get_required(CONFIG, "db_path"))
COLLECTION_NAME = get_required(CONFIG, "collection_name")

MAIN_LLM = get_required(MODELS, "main_llm")
RETRIEVER_LLM = get_required(MODELS, "retriever_llm")
EMBEDDING_MODEL = get_required(MODELS, "embedding_model")
