"""Sandboxed artefact paths for notebook runs.

Mọi thao tác đọc/ghi CSV hoặc model trong notebooks 01-04 đều đi qua hai helper này:

    rpath(rel)  -> ĐỌC : file trong sandbox nếu tồn tại, ngược lại đọc file canonical
    wpath(rel)  -> GHI : chỉ ghi vào sandbox (không bao giờ ghi đè canonical)

``rel`` là đường dẫn tương đối từ data root, ví dụ ``"processed/customer_features_scaled.csv"``.
Raw input trong ``data/raw/`` được đọc trực tiếp, không sandboxed.

Escape hatch: set ``SEG_CANONICAL_WRITE=1`` để ``wpath()`` ghi vào canonical tree.
"""

from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CANONICAL = _ROOT / "data"
_SANDBOX = _ROOT / "data" / "_sandbox"


def _write_canonical() -> bool:
    return os.environ.get("SEG_CANONICAL_WRITE") == "1"


def wpath(rel: str) -> str:
    """Absolute path to WRITE ``rel``. Sandbox by default; creates parent dirs."""
    base = _CANONICAL if _write_canonical() else _SANDBOX
    p = base / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def rpath(rel: str) -> str:
    """Absolute path to READ ``rel``: sandbox copy if present, else canonical."""
    sandbox = _SANDBOX / rel
    return str(sandbox if sandbox.exists() else _CANONICAL / rel)
