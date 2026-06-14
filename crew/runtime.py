from __future__ import annotations

import os
from pathlib import Path


def ensure_crewai_storage_writable(fallback_home: Path | str = ".crewai_home") -> Path:
    """Ensure CrewAI's SQLite task-output storage can be created.

    CrewAI uses the platform application-support directory by default. That is
    fine for normal local runs, but sandboxed executions may not be allowed to
    write there. In that case, fall back to a workspace-local HOME so CrewAI's
    appdirs path remains writable.
    """
    from crewai_core.paths import db_storage_path

    try:
        return _assert_writable(Path(db_storage_path()))
    except Exception:
        local_home = Path(os.getenv("TU_STUDY_ASSISTANT_CREWAI_HOME", str(fallback_home))).resolve()
        local_home.mkdir(parents=True, exist_ok=True)
        os.environ["HOME"] = str(local_home)
        return _assert_writable(Path(db_storage_path()))


def _assert_writable(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    probe = path / ".write_test"
    probe.write_text("", encoding="utf-8")
    probe.unlink(missing_ok=True)
    return path
