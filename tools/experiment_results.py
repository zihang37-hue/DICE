"""
实验结束后将配置与结果写入 results 目录下的 JSON 文件。
"""
import json
from datetime import datetime
from pathlib import Path


def write_experiment_results(config_summary, results_summary, output_dir=None):
    """
    将实验配置与汇总结果写入 JSON 文件。
    """
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_results_{timestamp}.json"
    output_path = output_dir / filename

    payload = {
        "config": config_summary,
        "results": results_summary,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(output_path)
