"""Small, solver-free I/O boundary for generated metric artifacts."""

import hashlib
import json
import math
import os
from pathlib import Path
from uuid import uuid4


def io_path(path: Path) -> Path:
    """Support the repository's long Windows result paths, including UNC."""
    value = str(path.resolve())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = ("\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\")
                 else "\\\\?\\" + value)
    return Path(value)


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_json(path: Path):
    return json.loads(io_path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload):
    """Replace only the caller's generated JSON, with strict finite JSON."""
    path = io_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(clean(payload), indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def digest_file(path: Path) -> str:
    return hashlib.sha256(io_path(path).read_bytes()).hexdigest()


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def display_path(path: Path) -> str:
    value = str(path.resolve())
    if value.startswith("\\\\?\\UNC\\"):
        return "\\\\" + value[8:]
    return value.removeprefix("\\\\?\\")
