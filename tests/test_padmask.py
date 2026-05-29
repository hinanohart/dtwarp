"""Padding / variable-length: non-vacuous gradient, endpoint-runtime == per-sample slice."""

from __future__ import annotations

import torch

from dtwarp import softdtw_divergence
from dtwarp.core.padmask import sliced_divergence, valid_lengths_from_pad


def test_valid_lengths_from_pad() -> None:
    pad = torch.zeros(3, 8, dtype=torch.bool)
    pad[0, 5:] = True
    pad[1, 6:] = True
    n = valid_lengths_from_pad(pad, 3, 8, torch.device("cpu"))
    assert n.tolist() == [5, 6, 8]
    # None -> full length
    n2 = valid_lengths_from_pad(None, 3, 8, torch.device("cpu"))
    assert n2.tolist() == [8, 8, 8]


def test_non_vacuous_gradient_under_padding() -> None:
    """Padded positions get exactly zero gradient; valid positions get non-zero gradient."""
    torch.manual_seed(0)
    b, t, a = 3, 8, 2
    x = torch.randn(b, t, a, dtype=torch.float64, requires_grad=True)
    y = torch.randn(b, t, a, dtype=torch.float64)
    pad = torch.zeros(b, t, dtype=torch.bool)
    pad[0, 5:] = True
    pad[1, 6:] = True
    n = valid_lengths_from_pad(pad, b, t, x.device)
    softdtw_divergence(x, y, gamma=0.2, valid_lengths=n).sum().backward()
    g = x.grad
    assert g is not None
    for i in range(b):
        ni = int(n[i])
        if ni < t:
            assert float(g[i, ni:].abs().max()) == 0.0, ("padded grad nonzero", i)
        assert float(g[i, :ni].abs().sum()) > 0.0, ("valid grad vacuous", i)


def test_endpoint_runtime_matches_per_sample_slice() -> None:
    """The batched endpoint-seeded runtime equals computing each sample on its sliced prefix."""
    torch.manual_seed(1)
    b, t, a = 4, 10, 3
    x = torch.randn(b, t, a, dtype=torch.float64)
    y = torch.randn(b, t, a, dtype=torch.float64)
    pad = torch.zeros(b, t, dtype=torch.bool)
    pad[0, 4:] = True
    pad[1, 7:] = True
    pad[2, 9:] = True
    n = valid_lengths_from_pad(pad, b, t, x.device)
    runtime = softdtw_divergence(x, y, gamma=0.2, valid_lengths=n)
    sliced = sliced_divergence(x, y, gamma=0.2, valid_lengths=n)
    assert torch.allclose(runtime, sliced, atol=1e-9), (runtime - sliced).abs().max()


def test_padded_target_perturbation_is_invariant() -> None:
    """Changing values in padded positions must not change the loss."""
    torch.manual_seed(2)
    b, t, a = 2, 9, 2
    x = torch.randn(b, t, a, dtype=torch.float64)
    y = torch.randn(b, t, a, dtype=torch.float64)
    pad = torch.zeros(b, t, dtype=torch.bool)
    pad[:, 5:] = True
    n = valid_lengths_from_pad(pad, b, t, x.device)
    base = softdtw_divergence(x, y, gamma=0.2, valid_lengths=n)
    y2 = y.clone()
    y2[:, 5:] += 100.0  # garbage in padded region
    x2 = x.clone()
    x2[:, 5:] -= 50.0
    after = softdtw_divergence(x2, y2, gamma=0.2, valid_lengths=n)
    assert torch.allclose(base, after, atol=1e-9), (base - after).abs().max()
