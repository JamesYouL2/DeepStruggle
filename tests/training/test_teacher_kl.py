"""P32: distilling a frozen teacher's policy into one seat's own decisions.

The term is KL(teacher || policy) over the rows it selects; the rest of the update is untouched.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ai.training.nash_pg import teacher_kl


def _masked_log_p(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.log_softmax(logits.masked_fill(~mask, -1e9), dim=-1)


def test_zero_when_the_policy_is_the_teacher() -> None:
    g = torch.Generator().manual_seed(0)
    mask = torch.rand(6, 9, generator=g) > 0.3
    mask[:, 0] = True
    lp = _masked_log_p(torch.randn(6, 9, generator=g), mask)
    kl, n = teacher_kl(lp, lp, torch.ones(6, dtype=torch.bool))
    assert float(n) == 6.0 and abs(float(kl)) < 1e-6


def test_only_selected_rows_count_and_masked_actions_add_nothing() -> None:
    """A masked action carries -1e9 in both log-probabilities; it must not turn into inf or NaN."""
    g = torch.Generator().manual_seed(1)
    mask = torch.ones(4, 7, dtype=torch.bool)
    mask[:, 5:] = False
    student = _masked_log_p(torch.randn(4, 7, generator=g), mask)
    teacher = _masked_log_p(torch.randn(4, 7, generator=g), mask)
    rows = torch.tensor([True, False, True, False])
    kl, n = teacher_kl(student, teacher, rows)
    want = torch.stack([(teacher[i].exp() * (teacher[i] - student[i]))[:5].sum() for i in (0, 2)]).mean()
    assert float(n) == 2.0 and torch.isfinite(kl) and abs(float(kl) - float(want)) < 1e-6
    none, n0 = teacher_kl(student, teacher, torch.zeros(4, dtype=torch.bool))
    assert float(n0) == 0.0 and float(none) == 0.0


def test_its_gradient_moves_the_policy_toward_the_teacher() -> None:
    logits = torch.zeros(1, 3, requires_grad=True)
    teacher = torch.log(torch.tensor([[0.8, 0.1, 0.1]]))
    kl, _ = teacher_kl(F.log_softmax(logits, dim=-1), teacher, torch.ones(1, dtype=torch.bool))
    kl.backward()
    assert logits.grad is not None and float(logits.grad[0, 0]) < 0.0   # raise the teacher's pick
