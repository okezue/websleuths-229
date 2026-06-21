# AWS integration boundary

All research logic, local execution, checkpointing, evaluation, and reporting are implemented in this repository. AWS-specific concerns are intentionally isolated:

- job submission and queues;
- instance and accelerator selection;
- EFS/S3 mounting and credential wiring;
- spot interruption handling;
- distributed rendezvous and elastic launch;
- cluster observability.

`wm.infra.aws_placeholder` defines the expected interface and raises a clear `NotImplementedError`. No other module depends on AWS.
