import json


def load_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def select_section(config, section):
    if isinstance(config, dict) and isinstance(config.get(section), dict):
        return config[section]
    return config if isinstance(config, dict) else {}


def get_cfg(config, key, default=None):
    if not key:
        return default
    current = config
    for part in key.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    return current


def _set_cfg(config, key, value):
    parts = key.split(".")
    current = config
    for part in parts[:-1]:
        if part not in current or not isinstance(current[part], dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def apply_overrides(config, overrides):
    if not overrides:
        return config if isinstance(config, dict) else {}
    merged = dict(config) if isinstance(config, dict) else {}
    for key, value in overrides.items():
        if value is None:
            continue
        _set_cfg(merged, key, value)
    return merged

