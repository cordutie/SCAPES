import ast
import json
from pathlib import Path


def _set_nested(config, dotted_key, value):
    keys = [k for k in dotted_key.split(".") if k]
    current = config
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    if keys:
        current[keys[-1]] = value


def _parse_gin_value(raw_value):
    value = raw_value.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ["none", "null"]:
        return None
    try:
        return ast.literal_eval(value)
    except Exception:
        return value


def _strip_inline_comment(line):
    result = []
    in_single = False
    in_double = False
    escape = False
    for ch in line:
        if escape:
            result.append(ch)
            escape = False
            continue
        if ch == "\\":
            result.append(ch)
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        if ch == "#" and not in_single and not in_double:
            break
        result.append(ch)
    return "".join(result).strip()


def _is_value_complete(value):
    stack = []
    in_single = False
    in_double = False
    escape = False
    for ch in value:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            continue
        if in_single or in_double:
            continue
        if ch in "[({":
            stack.append(ch)
        elif ch in "])}":
            if stack:
                stack.pop()
    return not stack and not in_single and not in_double


def parse_gin_config(path: Path):
    config = {}
    current_key = None
    value_parts = []
    with open(path, "r") as f:
        for raw_line in f:
            line = _strip_inline_comment(raw_line)
            if not line:
                continue

            if current_key is None:
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if not key:
                    continue
                current_key = key
                value_parts = [value]
            else:
                value_parts.append(line.strip())

            value_str = " ".join(value_parts)
            if _is_value_complete(value_str):
                _set_nested(config, current_key, _parse_gin_value(value_str))
                current_key = None
                value_parts = []

    if current_key is not None and value_parts:
        value_str = " ".join(value_parts)
        _set_nested(config, current_key, _parse_gin_value(value_str))

    return config


def load_config(config_dir: Path):
    for name in [
        "dataprep.gin",
        # "config.gin",
        # "config.json",
        # "config.kson",
        # "dataprep.json",
    ]:
        path = config_dir / name
        if not path.exists():
            continue
        if path.suffix == ".gin":
            return parse_gin_config(path), path
        with open(path, "r") as f:
            return json.load(f), path
    return {}, None
