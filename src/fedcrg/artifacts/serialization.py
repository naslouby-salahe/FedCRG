"""Atomic serialization helpers for typed scientific evidence."""

from __future__ import annotations

import json
import os
from dataclasses import fields, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Mapping, TypeAlias

from fedcrg.core.ids import AttackGroupId, ClientId, RowId, RunId, Sha256

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def _json_key(value: object) -> str:
    converted = to_json_value(value)
    if isinstance(converted, str):
        return converted
    if isinstance(converted, (int, float, bool)):
        return str(converted)
    raise TypeError(f"Unsupported JSON object key: {type(value).__name__}")


def to_json_value(value: object) -> JsonValue:
    """Convert domain objects to JSON without weakening internal type contracts."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return to_json_value(value.value)
    if isinstance(value, (ClientId, RowId, AttackGroupId, RunId, Sha256)):
        return value.value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: to_json_value(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {
            _json_key(key): to_json_value(item)
            for key, item in value.items()
        }
    if isinstance(value, (tuple, list, set, frozenset)):
        return [to_json_value(item) for item in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return to_json_value(model_dump(mode="json"))
    raise TypeError(f"Unsupported JSON evidence type: {type(value).__name__}")


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp")
    temp.write_text(content, encoding="utf-8")
    os.replace(temp, path)


def atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path,
        json.dumps(to_json_value(payload), indent=2, sort_keys=True) + "\n",
    )
