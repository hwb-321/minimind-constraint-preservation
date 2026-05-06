from pathlib import Path
import importlib

import yaml


_CONFIG_CACHE = None


def find_project_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for path in (current, *current.parents):
        if (path / "config.yaml").exists():
            return path
    return current


def load_project_config(config_path: str | Path | None = None) -> dict:
    global _CONFIG_CACHE
    if config_path is None and _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = Path(config_path) if config_path is not None else find_project_root() / "config.yaml"
    if not path.exists():
        config = {}
    else:
        with path.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}

    if config_path is None:
        _CONFIG_CACHE = config
    return config


def config_get(config: dict, dotted_key: str, default=None):
    value = config
    for key in dotted_key.split("."):
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def config_bool(config: dict, dotted_key: str, default: bool = False) -> bool:
    value = config_get(config, dotted_key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def project_path(path_value: str | Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(find_project_root() / path)


def load_model_classes(config: dict):
    module_name = config_get(config, "model.module", "model.model_minimind")
    config_class_name = config_get(config, "model.config_class", "MiniMindConfig")
    model_class_name = config_get(config, "model.model_class", "MiniMindForCausalLM")
    module = importlib.import_module(module_name)
    return getattr(module, config_class_name), getattr(module, model_class_name)


def model_config_kwargs(config: dict) -> dict:
    model_keys = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "intermediate_size",
        "rms_norm_eps",
        "use_qk_norm",
        "use_moe"
    ]
    kwargs = {}
    for key in model_keys:
        dotted = f"model.{key}"
        value = config_get(config, dotted, None)
        if value is not None:
            kwargs[key] = value
    kwargs["use_attention_gate"] = config_bool(config, "attention_gate.enabled", False)
    kwargs["attention_gate_scale"] = config_get(config, "attention_gate.scale", 2.0)
    return kwargs
