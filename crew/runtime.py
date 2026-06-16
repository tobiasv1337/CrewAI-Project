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


def _apply_context_propagation_patches() -> None:
    """Propagate ContextVars context across threads and ThreadPoolExecutor tasks.

    Python's ContextVars are thread-local by default and do not propagate to
    threads spawned by Thread or ThreadPoolExecutor. Since CrewAI runs agents
    and tools in separate threads/executors, this patch ensures that context
    variables like the active profile slug and recorded course proposals are
    shared correctly.
    """
    import threading
    import contextvars
    from concurrent.futures import ThreadPoolExecutor

    original_init = threading.Thread.__init__
    original_run = threading.Thread.run

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._context = contextvars.copy_context()

    def patched_run(self):
        if hasattr(self, "_context"):
            self._context.run(original_run, self)
        else:
            original_run(self)

    threading.Thread.__init__ = patched_init
    threading.Thread.run = patched_run

    original_submit = ThreadPoolExecutor.submit

    def patched_submit(self, fn, *args, **kwargs):
        ctx = contextvars.copy_context()
        def wrapper(*w_args, **w_kwargs):
            return ctx.run(fn, *w_args, **w_kwargs)
        return original_submit(self, wrapper, *args, **kwargs)

    ThreadPoolExecutor.submit = patched_submit


_apply_context_propagation_patches()
