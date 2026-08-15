"""Records reward-model training runs so metrics can be compared across approaches
over time. Each run is written as one JSON file under a runs directory; visualize.py
reads them back to build a comparison table/chart."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        return result.stdout.strip()
    except Exception:
        return None


def log_run(
    runs_dir: str | Path,
    *,
    run_name: str,
    model_class: str,
    config: dict[str, Any],
    metrics: dict[str, Any],
    training: dict[str, Any] | None = None,
) -> Path:
    """Writes one JSON record for this run into runs_dir, named so runs sort
    chronologically by filename. Returns the written path.

    `training` holds fit-time diagnostics (convergence, timing, dataset shape at fit
    time) that are worth keeping on record for future debugging/comparison even though
    they're not part of the headline evaluate() metrics and aren't charted by
    visualize.py."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{run_name}"
    record = {
        "run_id": run_id,
        "timestamp": timestamp,
        "run_name": run_name,
        "model_class": model_class,
        "git_commit": _git_commit(),
        "config": config,
        "metrics": metrics,
        "training": training or {},
    }
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    path = runs_dir / f"{run_id}.json"
    path.write_text(json.dumps(record, indent=2))
    return path


def load_runs(runs_dir: str | Path) -> list[dict[str, Any]]:
    """Loads every run record from runs_dir, sorted chronologically (filenames are
    timestamp-prefixed, so a plain sort suffices)."""
    runs_dir = Path(runs_dir)
    if not runs_dir.exists():
        return []
    paths = sorted(runs_dir.glob("*.json"))
    return [json.loads(p.read_text()) for p in paths]
