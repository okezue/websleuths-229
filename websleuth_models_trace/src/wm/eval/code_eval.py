from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class CodeResult:
    passed: bool
    stdout: str
    stderr: str
    returncode: int
    timed_out: bool = False


def _limits(cpu_seconds: int, memory_mb: int):
    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        memory = memory_mb * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_FSIZE, (10 * 1024 * 1024, 10 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    return apply


class SandboxedPythonRunner:
    """Best-effort local process isolation. Use a container/VM for untrusted public evaluation."""

    def __init__(self, timeout_seconds: int = 10, memory_mb: int = 512):
        self.timeout_seconds = timeout_seconds
        self.memory_mb = memory_mb

    def run(self, code: str, tests: str | Iterable[str]) -> CodeResult:
        tests_text = tests if isinstance(tests, str) else "\n".join(tests)
        program = code + "\n\n" + tests_text + "\n"
        with tempfile.TemporaryDirectory(prefix="websleuth_code_") as tmp:
            path = Path(tmp) / "solution.py"
            path.write_text(program, encoding="utf-8")
            env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "HOME": tmp, "TMPDIR": tmp}
            try:
                proc = subprocess.run(
                    [sys.executable, "-I", str(path)],
                    cwd=tmp,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=self.timeout_seconds,
                    preexec_fn=_limits(self.timeout_seconds, self.memory_mb) if os.name == "posix" else None,
                )
                return CodeResult(proc.returncode == 0, proc.stdout[-8000:], proc.stderr[-8000:], proc.returncode)
            except subprocess.TimeoutExpired as exc:
                return CodeResult(False, exc.stdout or "", exc.stderr or "", -1, timed_out=True)
