from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wm.core.io import ensure_dir


@dataclass
class LocalJob:
    command: list[str]
    workdir: str = "."
    env: dict[str, str] = field(default_factory=dict)
    stdout_path: str | None = None
    stderr_path: str | None = None


@dataclass
class LocalJobResult:
    returncode: int
    command: list[str]
    stdout_path: str | None
    stderr_path: str | None


class LocalRunner:
    def run(self, job: LocalJob, check: bool = True) -> LocalJobResult:
        env = os.environ.copy()
        env.update(job.env)
        stdout_handle: Any = None
        stderr_handle: Any = None
        try:
            if job.stdout_path:
                path = Path(job.stdout_path)
                ensure_dir(path.parent)
                stdout_handle = path.open("w", encoding="utf-8")
            if job.stderr_path:
                path = Path(job.stderr_path)
                ensure_dir(path.parent)
                stderr_handle = path.open("w", encoding="utf-8")
            proc = subprocess.run(
                job.command,
                cwd=job.workdir,
                env=env,
                stdout=stdout_handle,
                stderr=stderr_handle,
                text=True,
                check=check,
            )
            return LocalJobResult(proc.returncode, job.command, job.stdout_path, job.stderr_path)
        finally:
            if stdout_handle:
                stdout_handle.close()
            if stderr_handle:
                stderr_handle.close()

    @staticmethod
    def shell(command: str, **kwargs: Any) -> LocalJob:
        return LocalJob(command=shlex.split(command), **kwargs)
