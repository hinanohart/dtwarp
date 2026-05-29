"""Tier-1 empirical harness: paired baseline-vs-dtwarp training + held-out action error.

Two dataset sources share one paired protocol:

* ``synthetic`` (always available, CPU-only, no LeRobot) — smooth random trajectories with
  injected tempo jitter; deterministic, fast, used in CI. Labeled ``mode="synthetic-injected-tempo"``.
* ``pusht_keypoints`` (optional, needs ``dtwarp[eval]``) — real ``lerobot/pusht_keypoints``
  trajectories cut into action chunks. Labeled ``mode="real-pusht_keypoints"``.

Protocol (paired): for each seed, train two TinyChunkPolicy models from the SAME initialization
on the SAME minibatch order and the SAME train/held-out split, differing ONLY in the loss
(``blend=0`` native L1 baseline vs. ``blend>0`` dtwarp). The held-out per-sample plain (masked)
action MSE is the paired unit; the delta ``baseline - dtwarp`` is bootstrapped. The headline metric
is the neutral plain MSE (NOT the DTW-aligned error, which would favor dtwarp by construction).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from dtwarp.core.softdtw import softdtw_divergence
from dtwarp.eval.bootstrap import paired_bootstrap_ci
from dtwarp.loss.heads import masked_base_loss

__all__ = ["Tier1Config", "make_synthetic_tempo_dataset", "load_pusht_chunks", "run_tier1"]


@dataclass
class Tier1Config:
    chunk: int = 16
    blend: float = 0.5
    gamma: float = 0.1
    steps: int = 1200
    batch: int = 64
    hidden: int = 128
    lr: float = 1e-3
    seeds: tuple[int, ...] = (0, 1, 2, 3)
    heldout_frac: float = 0.25
    n_resamples: int = 2000
    extra: dict[str, object] = field(default_factory=dict)


# --------------------------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------------------------
def make_synthetic_tempo_dataset(
    n_samples: int = 800, chunk: int = 16, adim: int = 2, n_proto: int = 6, seed: int = 0
) -> tuple[Tensor, Tensor, Tensor]:
    """Smooth trajectories with injected tempo/phase jitter.

    Each sample is a prototype sinusoidal motion replayed at a random tempo and phase. The input
    is (one-hot prototype, phase, tempo); the target is the resulting action chunk. No structure is
    added to favor any loss — soft-DTW may or may not help generalization across tempo.
    """
    rng = np.random.default_rng(seed)
    freqs = rng.uniform(0.5, 1.5, size=(n_proto, adim))
    phases0 = rng.uniform(0, 2 * np.pi, size=(n_proto, adim))
    proto = rng.integers(0, n_proto, size=n_samples)
    tempo = rng.uniform(0.7, 1.3, size=n_samples)
    phase = rng.uniform(0, 2 * np.pi, size=n_samples)
    t = np.arange(chunk)[None, :]  # (1, chunk)
    chunks = np.zeros((n_samples, chunk, adim), dtype=np.float32)
    for a in range(adim):
        f = freqs[proto, a][:, None] * tempo[:, None]
        ph = phases0[proto, a][:, None] + phase[:, None]
        chunks[:, :, a] = np.sin(f * t * 0.4 + ph).astype(np.float32)
    chunks += rng.normal(0, 0.02, size=chunks.shape).astype(np.float32)
    onehot = np.eye(n_proto, dtype=np.float32)[proto]
    states = np.concatenate(
        [onehot, phase[:, None].astype(np.float32), tempo[:, None].astype(np.float32)], axis=1
    )
    pads = np.zeros((n_samples, chunk), dtype=bool)  # synthetic chunks are full-length
    return torch.from_numpy(states), torch.from_numpy(chunks), torch.from_numpy(pads)


def load_pusht_chunks(
    chunk: int = 16, max_episodes: int | None = None, stride: int = 2, seed: int = 0
) -> tuple[Tensor, Tensor, Tensor]:
    """Load real ``lerobot/pusht_keypoints`` and cut (state -> action chunk) samples.

    Input state = concat(observation.state, observation.environment_state). Target = the next
    ``chunk`` actions; chunks that run past the episode end are right-padded and masked
    (``action_is_pad``), exactly as LeRobot does. Requires ``dtwarp[eval]``.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except Exception as exc:  # pragma: no cover - exercised only without lerobot
        raise RuntimeError("load_pusht_chunks requires `pip install 'dtwarp[eval]'` (lerobot).") from exc

    ds = LeRobotDataset("lerobot/pusht_keypoints")
    hf = ds.hf_dataset.with_format("numpy")
    actions = np.asarray(hf["action"], dtype=np.float32)
    obs_state = np.asarray(hf["observation.state"], dtype=np.float32)
    env_state = np.asarray(hf["observation.environment_state"], dtype=np.float32)
    ep_index = np.asarray(hf["episode_index"]).astype(np.int64)
    states_full = np.concatenate([obs_state, env_state], axis=1)

    uniq = np.unique(ep_index)
    if max_episodes is not None:
        rng = np.random.default_rng(seed)
        uniq = rng.permutation(uniq)[:max_episodes]
    adim = actions.shape[1]
    s_list, c_list, p_list = [], [], []
    for ep in uniq:
        rows = np.flatnonzero(ep_index == ep)
        for start in range(0, len(rows), stride):
            r = rows[start]
            chunk_rows = rows[start : start + chunk]
            ck = np.zeros((chunk, adim), dtype=np.float32)
            pad = np.ones((chunk,), dtype=bool)
            ck[: len(chunk_rows)] = actions[chunk_rows]
            pad[: len(chunk_rows)] = False
            s_list.append(states_full[r])
            c_list.append(ck)
            p_list.append(pad)
    states = np.stack(s_list).astype(np.float32)
    chunks = np.stack(c_list).astype(np.float32)
    pads = np.stack(p_list)
    return torch.from_numpy(states), torch.from_numpy(chunks), torch.from_numpy(pads)


# --------------------------------------------------------------------------------------------
# tiny chunk policy + paired training
# --------------------------------------------------------------------------------------------
class TinyChunkPolicy(nn.Module):
    """Minimal action-chunking regressor: state -> flattened (chunk, adim)."""

    def __init__(self, state_dim: int, chunk: int, adim: int, hidden: int = 128) -> None:
        super().__init__()
        self.chunk = chunk
        self.adim = adim
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, chunk * adim),
        )

    def forward(self, state: Tensor) -> Tensor:
        out: Tensor = self.net(state)
        return out.view(-1, self.chunk, self.adim)


def _dtwarp_train_loss(pred: Tensor, target: Tensor, pad: Tensor, blend: float, gamma: float) -> Tensor:
    base = masked_base_loss(pred, target, pad, base="l1", pad_aware=True)
    if blend == 0.0:
        return base
    n = (~pad).sum(dim=1).clamp_min(1).to(pred.dtype)
    div = softdtw_divergence(pred, target, gamma=gamma, valid_lengths=(~pad).sum(dim=1).clamp_min(1))
    return (1.0 - blend) * base + blend * (div / n).mean()


def _heldout_plain_mse(pred: Tensor, target: Tensor, pad: Tensor) -> NDArray[np.float64]:
    """Per-sample masked plain action MSE (the neutral headline metric — favors neither loss)."""
    err = (pred - target).pow(2)  # (B, T, A)
    valid = (~pad).unsqueeze(-1).to(err.dtype)
    denom = (valid.sum(dim=(1, 2)) * err.shape[-1]).clamp_min(1)
    per = (err * valid).sum(dim=(1, 2)) / denom
    return per.detach().cpu().numpy()


def _heldout_dtw_aligned(pred: Tensor, target: Tensor, pad: Tensor, gamma: float) -> NDArray[np.float64]:
    """Per-sample length-normalized soft-DTW divergence (the alignment-tolerant metric dtwarp
    optimizes — reported as secondary context, NOT the neutral headline)."""
    n = (~pad).sum(dim=1).clamp_min(1)
    div = softdtw_divergence(pred, target, gamma=gamma, valid_lengths=n)
    return (div / n.to(div.dtype)).detach().cpu().numpy()


def _train_and_eval(
    states: Tensor, chunks: Tensor, pads: Tensor, cfg: Tier1Config, blend: float, seed: int
) -> dict[str, NDArray[np.float64]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = states.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_held = max(2, int(n * cfg.heldout_frac))
    held_idx, train_idx = perm[:n_held], perm[n_held:]
    s_tr, c_tr, p_tr = states[train_idx], chunks[train_idx], pads[train_idx]
    s_he, c_he, p_he = states[held_idx], chunks[held_idx], pads[held_idx]

    torch.manual_seed(seed)  # identical init for baseline & dtwarp (paired)
    model = TinyChunkPolicy(states.shape[1], cfg.chunk, chunks.shape[2], cfg.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    step_gen = torch.Generator().manual_seed(seed + 10_000)  # identical minibatch order (paired)
    n_tr = s_tr.shape[0]
    for _ in range(cfg.steps):
        bidx = torch.randint(0, n_tr, (min(cfg.batch, n_tr),), generator=step_gen)
        pred = model(s_tr[bidx])
        loss = _dtwarp_train_loss(pred, c_tr[bidx], p_tr[bidx], blend, cfg.gamma)
        opt.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        opt.step()
    model.eval()
    with torch.no_grad():
        pred_he = model(s_he)
    return {
        "plain_mse": _heldout_plain_mse(pred_he, c_he, p_he),
        "dtw_aligned": _heldout_dtw_aligned(pred_he, c_he, p_he, cfg.gamma),
    }


def run_tier1(
    dataset_kind: str = "synthetic",
    cfg: Tier1Config | None = None,
    states: Tensor | None = None,
    chunks: Tensor | None = None,
    pads: Tensor | None = None,
) -> dict[str, object]:
    """Run the paired Tier-1 experiment and return a results dict (no I/O).

    ``dataset_kind`` in {"synthetic", "pusht_keypoints"} (or pass tensors directly). The headline
    delta is the held-out plain-MSE improvement (baseline - dtwarp) with a paired bootstrap CI.
    """
    cfg = cfg or Tier1Config()
    if states is None:
        if dataset_kind == "synthetic":
            states, chunks, pads = make_synthetic_tempo_dataset(chunk=cfg.chunk)
            mode = "synthetic-injected-tempo"
        elif dataset_kind == "pusht_keypoints":
            states, chunks, pads = load_pusht_chunks(chunk=cfg.chunk, **cfg.extra)  # type: ignore[arg-type]
            mode = "real-pusht_keypoints"
        else:
            raise ValueError(f"unknown dataset_kind {dataset_kind!r}")
    else:
        assert chunks is not None and pads is not None
        mode = f"provided:{dataset_kind}"

    plain_deltas: list[NDArray[np.float64]] = []
    aligned_deltas: list[NDArray[np.float64]] = []
    per_seed: list[dict[str, float]] = []
    for seed in cfg.seeds:
        base = _train_and_eval(states, chunks, pads, cfg, blend=0.0, seed=seed)
        dtw = _train_and_eval(states, chunks, pads, cfg, blend=cfg.blend, seed=seed)
        # positive delta => dtwarp lowered held-out error (paired: same init/order/split)
        plain_deltas.append(base["plain_mse"] - dtw["plain_mse"])
        aligned_deltas.append(base["dtw_aligned"] - dtw["dtw_aligned"])
        per_seed.append(
            {
                "seed": float(seed),
                "baseline_plain_mse": float(base["plain_mse"].mean()),
                "dtwarp_plain_mse": float(dtw["plain_mse"].mean()),
                "baseline_dtw_aligned": float(base["dtw_aligned"].mean()),
                "dtwarp_dtw_aligned": float(dtw["dtw_aligned"].mean()),
            }
        )
    plain = np.concatenate(plain_deltas)
    aligned = np.concatenate(aligned_deltas)
    headline = paired_bootstrap_ci(plain, n_resamples=cfg.n_resamples, seed=0)
    secondary = paired_bootstrap_ci(aligned, n_resamples=cfg.n_resamples, seed=0)

    return {
        "mode": mode,
        "dataset_kind": dataset_kind,
        "n_samples": int(states.shape[0]),
        "n_heldout_units": int(plain.shape[0]),
        "chunk": cfg.chunk,
        "blend": cfg.blend,
        "gamma": cfg.gamma,
        "steps": cfg.steps,
        "seeds": list(cfg.seeds),
        "paired": True,
        "metric": "heldout_plain_masked_action_mse_delta(baseline-dtwarp) [HEADLINE,neutral]",
        "secondary_metric": "heldout_softdtw_divergence_delta(baseline-dtwarp) [context:dtwarp's own metric]",
        "bootstrap": headline.as_dict(),
        "secondary_bootstrap": secondary.as_dict(),
        "per_seed": per_seed,
    }
