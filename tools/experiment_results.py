"""
实验结束后将配置与汇总结果写入 results 目录下的 JSON 文件。
"""
import json
from datetime import datetime
from pathlib import Path


def write_experiment_results(config_summary, results_summary, output_dir=None):
    """将实验配置与汇总结果写入 JSON；未指定 output_dir 时使用本文件所在目录。"""
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 功能点：用时间戳生成唯一文件名，避免覆盖
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_results_{timestamp}.json"
    output_path = output_dir / filename

    # 功能点：组装 config + results 两个顶层键，便于后续分析或复现
    payload = {
        "config": config_summary,
        "results": results_summary,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return str(output_path)
