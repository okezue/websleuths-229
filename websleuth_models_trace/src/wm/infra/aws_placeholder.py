from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AWSJobSpec:
    name: str
    command: list[str]
    instance_type: str
    image_uri: str
    environment: dict[str, str]
    storage: dict[str, Any]


class AWSRunner:
    """Integration boundary for the project's AWS/compute control plane."""

    def submit(self, spec: AWSJobSpec) -> str:
        raise NotImplementedError(
            "Connect this interface to your AWS Batch, EKS, SageMaker, or internal scheduler. "
            "All model, data, checkpoint, evaluation, and reporting code is implemented elsewhere."
        )

    def status(self, job_id: str) -> dict[str, Any]:
        raise NotImplementedError("Implement scheduler-specific job status lookup")

    def cancel(self, job_id: str) -> None:
        raise NotImplementedError("Implement scheduler-specific cancellation")
