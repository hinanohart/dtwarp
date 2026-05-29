# dtwarp — design notes & frozen decisions

Source of truth for the *why*. The frozen architecture lives in the project memory
`project_dtwarp_architecture_2026-05-29`; this file records the implementation-level
decisions made during the build, especially where the realization deviates from the
literal architecture text in order to satisfy its *intent* (the tests) correctly.

## 1. Soft-DTW divergence is the only loss core (P1)

We implement the **divergence** form (Blondel/Mensch/Vert 2021, arXiv:2010.08354):

    D_gamma(x, y) = SDTW_gamma(x, y) - 0.5 * (SDTW_gamma(x, x) + SDTW_gamma(y, y))

Raw soft-DTW is *not* a valid behaviour-cloning loss: it is not minimized at x=y and
can be negative. The divergence is non-negative and equals 0 iff x=y. Raw SDTW is only
reachable behind `raw_softdtw=True` (debug flag), never the public default.

- `gamma` default **0.1** (tslearn's 1.0 is tuned for classification and over-smooths,
  blurring action onsets — an HONEST RISK we document; a sweep grid
  `{0.01, 0.1, 0.5, 1.0}` is exposed).
- Length normalization uses **T_valid** (post-mask valid length), matching LeRobot's
  `num_valid` convention — not T_pad, not sqrt(T).
- Ground cost: **joint squared-euclidean over action_dim** by default. `per_channel`
  (independent-per-dimension warping) and the `gak` (global-alignment-kernel) variant are
  **deferred to v0.2 and currently raise `NotImplementedError`** — the v0.1 risk they would
  address (onset crispness) is already covered by the optional derivative cost channel.

## 2. Padding & variable length (P2) — IMPLEMENTATION DEVIATION, justified

The architecture text proposed a "BIG=1e9 cost, path-blocked, single rectangular
anti-diagonal wavefront, read at the global corner". **That mechanism is mathematically
incorrect for variable-length sequences**: soft-DTW's value is read at the bottom-right
corner `R[T_pad-1, T_pad-1]`, and any path reaching the *padded* corner must traverse
padded cells, so a BIG cost there inflates the loss — it does **not** reproduce the
soft-DTW of the valid prefix.

We therefore realize the frozen *intent* (the P2 test spec: `D(x,x)==0`, padded-gradient
is exactly zero, valid-gradient is non-zero, and the result matches per-sample slicing)
with an **endpoint-seeded wavefront**:

- Forward: compute the full `(B, T, T)` recursion `R` with a batched anti-diagonal
  wavefront, then **gather the loss per sample at its valid endpoint** `R[b, n[b]-1, n[b]-1]`,
  where `n[b]` is the valid length from `~action_is_pad`.
- Because `R[b, n-1, n-1]` depends only on cells `(i<=n-1, j<=n-1)` — all valid — the
  padded inputs **never enter the computed value**, so their gradient is exactly zero
  *by construction* (no BIG cost needed, no numerical hazard).
- Backward (custom): the soft-alignment matrix `E` is **seeded at the per-sample
  endpoint** `(n-1, n-1)` and propagated up-left; cells beyond the endpoint stay zero.

This is provably equivalent to computing soft-DTW on each sample's sliced valid prefix.
The **per-sample slice** computation is retained as a TEST ORACLE (`tests/test_padmask.py`),
and a BIG-block variant is kept only as an additional cross-check, never as the runtime.
This resolves the synthesis-critic's objection that a BIG-block runtime and a per-sample
slice runtime cannot both be the live path for variable `T_valid`.

## 3. Custom autograd (Cuturi-Blondel backward)

`SoftDTW(torch.autograd.Function)` computes `R` in the forward and the soft-alignment
matrix `E` in a reverse recursion (the standard Cuturi-Blondel gradient,
`dSDTW/dC = E`). We validate the hand-written backward with `torch.autograd.gradcheck`
in float64 on **all three** divergence terms (SDTW(x,y), SDTW(x,x), SDTW(y,y) via D).
fp32 is forced in the kernel; gamma is clamped to `>= 1e-3` to avoid softmin overflow.

## 4. LeRobot integration (P3) — what is wired vs. what is forbidden

Anchors resolved against LeRobot SHA `24017e96` (v0.5.2); see `anchors.json`.

- **ACT** (`modeling_act.py:145-150`): clean drop-in. `policy.forward` produces
  `actions_hat`; we replace the masked L1 with the blended soft-DTW divergence.
  `blend=0` reproduces LeRobot's masked-L1 **bit-for-bit** (num_valid = valid*A).
- **flow-matching (pi0 / SmolVLA)**: the loss `F.mse_loss(u_t, v_t, reduction='none')`
  is computed *inside* the flow model with `u_t = noise - actions` (a velocity that
  **does** carry temporal shape), so `softdtw(v_t, u_t)` is meaningful. `flow_matching_head`
  takes `(v_t, u_t, action_is_pad)`. `blend=0` reproduces the masked MSE mean
  (SmolVLA-style masked, or pi0-style plain mean via `pad_aware`).
  NOTE: because the MSE is computed deep in the flow model, the head is supplied as a
  composable building block + a documented adapter; it is **not** auto-monkeypatched into
  the VLM forward (which would require running PaliGemma). This is stated honestly.
- **Diffusion-Policy epsilon**: **forbidden**. The target is white noise with no temporal
  shape; `softdtw(eps_hat, eps)` is meaningless. The only sound DP route is an auxiliary
  term on the reconstructed `x0_hat = (x_t - sqrt(1-abar)*eps_hat)/sqrt(abar)`, but
  `sqrt(abar) -> 0` at high-noise t amplifies the error. This is a **v0.2** feature that
  REQUIRES an SNR/timestep clamp; v0.1 ships only a warned experimental stub that is not
  a CLAIM. `VQ-BeT`/discrete policies are dropped (smooth-trajectory assumption breaks)
  and raise.

## 5. Honesty / claims (P4)

The **only correctness CLAIM** is the divergence mathematics (non-negative, zero iff
equal) plus pad-mask correctness and the bit-exact `blend=0` reductions — all machine-tested.

We do **not** claim better sample-efficiency or rollout fidelity as a built-in fact. The
Tier-1 empirical harness trains paired (same data, same seed, loss swapped only) baselines
vs. dtwarp and reports a **paired bootstrap CI**. If `ci_low <= 0 <= ci_high`, the claim
tier auto-degrades to `alternative-elastic-loss` and the word "better" is forbidden
(grep-enforced in CI). README numbers are generated only from the results JSON.

## 6. Scope

- **v0.1**: divergence (gamma 0.1, sweep grid), endpoint-seeded pad-mask, ACT + pi0/SmolVLA
  flow heads, convex-blend knob (lambda=0 is a bit-exact regression guard), opt-in
  derivative cost channel (w_deriv=0 default), optional Maghoumi CUDA fast-path
  (git-install / not on PyPI, lazy/guarded/never-vendored), Tier-1 CPU harness with paired
  bootstrap CI + auto-degrade.
- **v0.2**: DP `dp_x0_aux` with SNR clamp, offline DBA barycenter relabel, learnable
  Mahalanobis cost, annealed blend schedule, Tier-2 closed-loop.
