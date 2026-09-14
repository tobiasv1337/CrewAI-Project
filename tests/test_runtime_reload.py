import subprocess
import sys


def test_context_patch_is_idempotent_after_reload():
    result = subprocess.run([sys.executable, "-c", '''
import contextvars
import importlib
import threading
from concurrent.futures import ThreadPoolExecutor
import crew.runtime
initial_run = threading.Thread.run
importlib.reload(crew.runtime)
assert threading.Thread.run is initial_run
value = contextvars.ContextVar("test_value", default="missing")
value.set("inherited")
received = []
thread = threading.Thread(target=lambda: received.append(value.get()))
thread.start()
thread.join()
assert received == ["inherited"]
with ThreadPoolExecutor(max_workers=1) as pool:
    assert pool.submit(value.get).result() == "inherited"
'''], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
