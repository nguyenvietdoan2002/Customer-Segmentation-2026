# -*- coding: utf-8 -*-
"""Smoke tests for the sandboxed notebook I/O helpers."""

from pathlib import Path

import notebook_io


def test_wpath_targets_sandbox_and_creates_parent(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox"
    canonical = tmp_path / "canonical"
    monkeypatch.setattr(notebook_io, "_SANDBOX", sandbox)
    monkeypatch.setattr(notebook_io, "_CANONICAL", canonical)
    monkeypatch.delenv("SEG_CANONICAL_WRITE", raising=False)

    p = Path(notebook_io.wpath("processed/out.csv"))

    assert p == sandbox / "processed" / "out.csv"
    assert p.parent.is_dir()  # parent created eagerly


def test_rpath_prefers_sandbox_else_canonical(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox"
    canonical = tmp_path / "canonical"
    monkeypatch.setattr(notebook_io, "_SANDBOX", sandbox)
    monkeypatch.setattr(notebook_io, "_CANONICAL", canonical)

    # Missing in sandbox -> falls back to canonical.
    assert Path(notebook_io.rpath("x.csv")) == canonical / "x.csv"

    # Present in sandbox -> sandbox wins.
    (sandbox).mkdir(parents=True, exist_ok=True)
    (sandbox / "x.csv").write_text("data")
    assert Path(notebook_io.rpath("x.csv")) == sandbox / "x.csv"


def test_wpath_canonical_escape_hatch(tmp_path, monkeypatch):
    sandbox = tmp_path / "sandbox"
    canonical = tmp_path / "canonical"
    monkeypatch.setattr(notebook_io, "_SANDBOX", sandbox)
    monkeypatch.setattr(notebook_io, "_CANONICAL", canonical)
    monkeypatch.setenv("SEG_CANONICAL_WRITE", "1")

    p = Path(notebook_io.wpath("processed/out.csv"))
    assert p == canonical / "processed" / "out.csv"
