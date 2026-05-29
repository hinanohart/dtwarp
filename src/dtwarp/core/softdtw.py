"""Soft-DTW forward, custom Cuturi-Blondel backward, and the divergence loss core.

The public loss is the soft-DTW *divergence* (Blondel, Mensch, Vert; AISTATS 2021,
arXiv:2010.08354):

    D_gamma(x, y) = SDTW_gamma(x, y) - 0.5 * (SDTW_gamma(x, x) + SDTW_gamma(y, y))

which is non-negative and equals 0 iff x == y. Raw soft-DTW (Cuturi & Blondel, ICML 2017)
is NOT a valid regression loss (not minimized at x=y, can be negative) and is exposed only
behind a debug flag.

Variable-length / padding is handled by an *endpoint-seeded* wavefront: the value is read at
each sample's valid endpoint R[b, n-1, n-1] and the backward soft-alignment E is seeded there,
so padded positions never enter the value or the gradient (zero pad-gradient by construction).
See DESIGN_NOTES.md section 2 for why this supersedes the BIG-block proposal.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = ["soft_dtw", "softdtw_divergence_from_costs", "softdtw_divergence", "MIN_GAMMA"]

MIN_GAMMA = 1e-3


def _clamp_gamma(gamma: float) -> float:
    g = float(gamma)
    if g < MIN_GAMMA:
        g = MIN_GAMMA
    return g


def _resolve_lengths(n: Tensor | None, batch: int, m: int, device: torch.device) -> Tensor:
    """Per-sample valid length (1-indexed endpoint). None -> full length m."""
    if n is None:
        return torch.full((batch,), m, dtype=torch.long, device=device)
    n = n.to(device=device, dtype=torch.long)
    if n.shape != (batch,):
        raise ValueError(f"valid-length tensor must have shape ({batch},), got {tuple(n.shape)}")
    if int(n.min()) < 1 or int(n.max()) > m:
        raise ValueError(f"valid lengths must lie in [1, {m}], got [{int(n.min())}, {int(n.max())}]")
    return n


def _sdtw_forward(cost: Tensor, gamma: float) -> Tensor:
    """Batched anti-diagonal soft-DTW forward.

    cost: (B, M, N) finite, fp32.  Returns R: (B, M+1, N+1) with R[:, 0, 0] = 0 and a +inf
    border, R[:, i, j] = cost[i-1, j-1] + softmin_gamma(R[i-1,j], R[i,j-1], R[i-1,j-1]).
    """
    b, m, n = cost.shape
    device, dtype = cost.device, cost.dtype
    r = torch.full((b, m + 1, n + 1), float("inf"), device=device, dtype=dtype)
    r[:, 0, 0] = 0.0
    for d in range(2, m + n + 1):
        i_lo = max(1, d - n)
        i_hi = min(m, d - 1)
        if i_lo > i_hi:
            continue
        i_idx = torch.arange(i_lo, i_hi + 1, device=device)
        j_idx = d - i_idx
        r_diag = r[:, i_idx - 1, j_idx - 1]
        r_up = r[:, i_idx - 1, j_idx]
        r_left = r[:, i_idx, j_idx - 1]
        c = cost[:, i_idx - 1, j_idx - 1]
        stacked = torch.stack([r_diag, r_up, r_left], dim=0)  # (3, B, k)
        m_min = stacked.min(dim=0).values  # (B, k); finite for all valid cells
        soft = m_min - gamma * torch.log(torch.exp(-(stacked - m_min) / gamma).sum(dim=0))
        r[:, i_idx, j_idx] = c + soft
    return r


def _edge_weight(rp: Tensor, cp: Tensor, ci: Tensor, cj: Tensor, r_ij: Tensor, inv_g: float) -> Tensor:
    """Soft-alignment edge weight from child cell (ci, cj) onto its parent (the cell at r_ij).

    Out-of-grid / unreached children carry e_child == 0, so their +inf R is sanitized to r_ij
    (giving a finite weight) to avoid 0 * inf = NaN.
    """
    r_child = rp[:, ci, cj]
    r_child_s = torch.where(torch.isinf(r_child), r_ij, r_child)
    return torch.exp((r_child_s - cp[:, ci, cj] - r_ij) * inv_g)


def _sdtw_backward(cost: Tensor, r: Tensor, gamma: float, rx: Tensor, ry: Tensor) -> Tensor:
    """Cuturi-Blondel soft-alignment backward, seeded at per-sample endpoints (rx, ry).

    Returns grad_cost (B, M, N) = dValue/dcost where Value[b] = r[b, rx[b], ry[b]].
    """
    b, m, n = cost.shape
    device, dtype = cost.device, cost.dtype
    inv_g = 1.0 / gamma

    rp = torch.full((b, m + 2, n + 2), float("inf"), device=device, dtype=dtype)
    rp[:, : m + 1, : n + 1] = r
    cp = torch.zeros((b, m + 2, n + 2), device=device, dtype=dtype)
    cp[:, 1 : m + 1, 1 : n + 1] = cost  # cp[i, j] = cost[i-1, j-1] for the cell at (i, j)
    e = torch.zeros((b, m + 2, n + 2), device=device, dtype=dtype)
    bidx = torch.arange(b, device=device)
    e[bidx, rx, ry] = 1.0  # seed dValue/dR[endpoint] = 1

    for d in range(m + n, 1, -1):
        i_lo = max(1, d - n)
        i_hi = min(m, d - 1)
        if i_lo > i_hi:
            continue
        i_idx = torch.arange(i_lo, i_hi + 1, device=device)
        j_idx = d - i_idx
        r_ij = rp[:, i_idx, j_idx]  # (B, k); finite for valid cells
        w_up = _edge_weight(rp, cp, i_idx + 1, j_idx, r_ij, inv_g)
        w_left = _edge_weight(rp, cp, i_idx, j_idx + 1, r_ij, inv_g)
        w_diag = _edge_weight(rp, cp, i_idx + 1, j_idx + 1, r_ij, inv_g)
        contrib = (
            e[:, i_idx + 1, j_idx] * w_up
            + e[:, i_idx, j_idx + 1] * w_left
            + e[:, i_idx + 1, j_idx + 1] * w_diag
        )
        e[:, i_idx, j_idx] = e[:, i_idx, j_idx] + contrib

    return e[:, 1 : m + 1, 1 : n + 1]


class _SoftDTW(torch.autograd.Function):
    """Differentiable soft-DTW value read at per-sample endpoints, w.r.t. the cost matrix."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        cost: Tensor,
        gamma: float,
        rx: Tensor,
        ry: Tensor,
    ) -> Tensor:
        cost32 = (
            cost.detach().to(torch.float32)
            if cost.dtype not in (torch.float32, torch.float64)
            else cost.detach()
        )
        r = _sdtw_forward(cost32, gamma)
        bidx = torch.arange(cost.shape[0], device=cost.device)
        value = r[bidx, rx, ry]
        ctx.save_for_backward(cost32, r, rx, ry)
        ctx.gamma = gamma  # type: ignore[attr-defined]
        return value.to(cost.dtype)

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor
    ) -> tuple[Tensor | None, None, None, None]:
        cost, r, rx, ry = ctx.saved_tensors  # type: ignore[attr-defined]
        grad_cost = _sdtw_backward(cost, r, ctx.gamma, rx, ry)  # type: ignore[attr-defined]
        grad_cost = grad_cost.to(grad_output.dtype) * grad_output[:, None, None]
        return grad_cost, None, None, None


def soft_dtw(cost: Tensor, gamma: float = 0.1, valid_lengths: Tensor | None = None) -> Tensor:
    """Soft-DTW value for a batch of cost matrices, read at each sample's valid endpoint.

    Args:
        cost: (B, M, N) pairwise ground-cost matrices (e.g. squared-euclidean).
        gamma: smoothing (clamped to >= MIN_GAMMA for numerical stability).
        valid_lengths: (B,) per-sample valid length n; the value is read at (n, n) if M == N,
            else interpreted as the shared endpoint. None -> full length. Padded positions
            (index >= n) never influence the value or its gradient.

    Returns:
        (B,) soft-DTW values.
    """
    if cost.dim() != 3:
        raise ValueError(f"cost must be (B, M, N), got {tuple(cost.shape)}")
    g = _clamp_gamma(gamma)
    b, m, n = cost.shape
    rx = _resolve_lengths(valid_lengths, b, m, cost.device)
    ry = _resolve_lengths(valid_lengths, b, n, cost.device)
    out: Tensor = _SoftDTW.apply(cost, g, rx, ry)  # type: ignore[no-untyped-call]
    return out


def softdtw_divergence_from_costs(
    cost_xy: Tensor,
    cost_xx: Tensor,
    cost_yy: Tensor,
    gamma: float = 0.1,
    valid_lengths: Tensor | None = None,
    raw: bool = False,
) -> Tensor:
    """Soft-DTW divergence from precomputed cost matrices.

    D = SDTW(x, y) - 0.5 * (SDTW(x, x) + SDTW(y, y)), non-negative, 0 iff x == y.
    If ``raw`` is True, returns SDTW(x, y) only (debug; NOT a valid loss).
    """
    g = _clamp_gamma(gamma)
    sxy = soft_dtw(cost_xy, g, valid_lengths)
    if raw:
        return sxy
    sxx = soft_dtw(cost_xx, g, valid_lengths)
    syy = soft_dtw(cost_yy, g, valid_lengths)
    return sxy - 0.5 * (sxx + syy)


def softdtw_divergence(
    x: Tensor,
    y: Tensor,
    gamma: float = 0.1,
    valid_lengths: Tensor | None = None,
    raw: bool = False,
    w_deriv: float = 0.0,
    per_channel: bool = False,
) -> Tensor:
    """Soft-DTW divergence between two action sequences.

    Args:
        x, y: (B, T, A) sequences (e.g. predicted vs. target actions / velocities).
        gamma: smoothing.
        valid_lengths: (B,) valid length per sample (from ``~action_is_pad``). None -> full T.
        raw: if True return raw SDTW(x, y) (debug only, not a loss).
        w_deriv: weight on a first-difference (velocity) cost channel (default 0.0 = off).
        per_channel: experimental per-action-dim cost (default joint squared-euclidean).

    Returns:
        (B,) per-sample divergence values.
    """
    from dtwarp.core.cost import pairwise_cost

    cost_xy = pairwise_cost(x, y, w_deriv=w_deriv, per_channel=per_channel)
    if raw:
        return softdtw_divergence_from_costs(cost_xy, cost_xy, cost_xy, gamma, valid_lengths, raw=True)
    cost_xx = pairwise_cost(x, x, w_deriv=w_deriv, per_channel=per_channel)
    cost_yy = pairwise_cost(y, y, w_deriv=w_deriv, per_channel=per_channel)
    return softdtw_divergence_from_costs(cost_xy, cost_xx, cost_yy, gamma, valid_lengths, raw=False)
