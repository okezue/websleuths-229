from __future__ import annotations

import torch
import torch.nn.functional as F


def js_divergence(student_logits: torch.Tensor, teacher_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Symmetric Jensen-Shannon divergence over aligned answer-token logits."""
    length = min(student_logits.shape[-2], teacher_logits.shape[-2])
    student = student_logits[..., :length, :] / temperature
    teacher = teacher_logits[..., :length, :] / temperature
    log_p = F.log_softmax(student, dim=-1)
    log_q = F.log_softmax(teacher.detach(), dim=-1)
    p = log_p.exp()
    q = log_q.exp()
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(1e-12))
    js = 0.5 * (p * (log_p - log_m)).sum(-1) + 0.5 * (q * (log_q - log_m)).sum(-1)
    return js.mean() * (temperature**2)


def orthogonality_penalty(new_parameters: list[torch.Tensor], old_vectors: list[torch.Tensor]) -> torch.Tensor:
    if not new_parameters or not old_vectors:
        device = new_parameters[0].device if new_parameters else torch.device("cpu")
        return torch.tensor(0.0, device=device)
    new = torch.cat([parameter.reshape(-1) for parameter in new_parameters])
    penalty = torch.tensor(0.0, device=new.device)
    for old in old_vectors:
        old = old.to(new.device).reshape(-1)
        n = min(new.numel(), old.numel())
        if n:
            penalty = penalty + F.cosine_similarity(new[:n].unsqueeze(0), old[:n].unsqueeze(0)).abs().mean()
    return penalty / max(len(old_vectors), 1)
