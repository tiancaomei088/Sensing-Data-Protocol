import json

from importlib.resources import files
from sdp import configs


def load_config(dataset_name: str) -> dict:
    try:
        json_path = files(configs) / "model_params.json"

        json_content = json_path.read_text(encoding="utf-8")
        data = json.loads(json_content)

        if dataset_name not in data:
            raise ValueError(f"no such dataset_name in config file: {dataset_name}. \
                                all dataset available: {list(data.keys())}")

        return data[dataset_name]

    except FileNotFoundError:
        raise RuntimeError("no config file, please contact developers")