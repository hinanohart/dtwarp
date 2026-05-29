# dtwarp

**A soft-DTW divergence loss head for [LeRobot](https://github.com/huggingface/lerobot) imitation learning.**

Temporally-elastic, phase-invariant, pad-aware, and a drop-in for ACT and flow-matching
(SmolVLA / pi0) policies. Pure PyTorch, CPU-trainable, no vendored kernels.

> **Status: v0.1.0a1 (pre-alpha).** Apache-2.0. The only correctness *claim* is the
> divergence mathematics and the LeRobot-faithful, pad-aware reductions — all machine-tested.
> Whether the elastic loss improves a policy is an *empirical* question this repo measures
> honestly (paired bootstrap CI); it is **not** asserted as a built-in fact. See
> [Empirical results](#empirical-results).

## Why

Standard behaviour cloning penalizes actions step-by-step (L1 for ACT, MSE on the
flow-matching velocity for SmolVLA/pi0). That makes the loss sensitive to small temporal
*phase* shifts between the demonstration and the prediction, even when the *shape* of the
trajectory is right. Soft-DTW measures distance under an optimal soft time-alignment, so
it is forgiving of phase while staying sensitive to shape.

dtwarp packages the **soft-DTW divergence** (Blondel/Mensch/Vert 2021) — the only form
that is a valid regression loss (non-negative, zero iff equal) — with LeRobot's exact
padding/masking conventions, so it drops in without forking LeRobot.

## Install

```bash
pip install dtwarp                 # core (CPU)
pip install "dtwarp[lerobot]"      # + LeRobot integration
pip install "dtwarp[cuda]"         # + optional CUDA fast-path (lazy, never vendored)
pip install "dtwarp[eval]"         # + empirical harness deps
```

## Quickstart

```python
import dtwarp

# 1) Use the loss head directly (framework-agnostic building block)
loss = dtwarp.softdtw_divergence(pred, target, gamma=0.1)   # pred,target: (B, T, A)

# 2) Drop into a LeRobot ACT policy (blend = convex mix of L1 and soft-DTW divergence)
policy = dtwarp.wrap_policy(policy, loss="softdtw_divergence", blend=0.5, gamma=0.1)
# blend=0.0 reproduces LeRobot's native masked loss bit-for-bit (a built-in regression guard).
```

`blend=0.0` is a hard guarantee: the wrapped loss equals LeRobot's own masked reduction
exactly, so adopting dtwarp is risk-free to try (anneal `blend` up from 0).

## What is wired (v0.1)

| Policy | Loss replaced | Status |
|---|---|---|
| ACT | masked L1 on `actions_hat` | drop-in `wrap_policy` |
| SmolVLA / pi0 | masked MSE on flow velocity `u_t = noise - actions` | `flow_matching_head` building block + adapter |
| Diffusion-Policy (epsilon) | — | **forbidden** (white-noise target has no temporal shape) |
| VQ-BeT / discrete | — | not supported (raises) |

## Empirical results

<!--MEASURED@S7--> Results are generated only from `eval/results/*.json` after the Tier-1
harness runs. Until then this section is intentionally empty — no hand-written numbers.

- Tier-1 (CPU, always): paired baseline-vs-dtwarp training on `lerobot/pusht_keypoints`,
  reported as a **paired bootstrap CI** on held-out / DTW-aligned action error.
- Tier-2 (closed-loop, conditional): `gym-pusht` success rate, or marked `deferred`.

If the CI for the improvement crosses zero, dtwarp is reported honestly as
"an alternative elastic loss", not "better".

## License & attribution

Apache-2.0. The soft-DTW core is an independent re-implementation; see [`NOTICE`](NOTICE)
for attribution to mblondel (BSD-2) and Maghoumi (MIT). Neither is vendored.

## Citation

<!--MEASURED@S7-->
