#!/usr/bin/env python3
"""Run the Tier-1 paired experiment and write a results JSON (+ print the README block).

Usage:
    python scripts/run_tier1.py --kind pusht_keypoints --steps 500 --seeds 0 1 2 --max-episodes 60
    python scripts/run_tier1.py --kind synthetic --steps 800

Real-pusht needs `pip install 'dtwarp[eval]'` (lerobot). The headline claim is auto-derived from
the paired bootstrap CI (auto-degrades to 'alternative-elastic-loss' if the CI contains 0).
"""

from __future__ import annotations

import argparse

from dtwarp.eval.harness import Tier1Config, run_tier1
from dtwarp.eval.report import build_results, results_to_readme_block, write_results_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["synthetic", "pusht_keypoints"], default="synthetic")
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--blend", type=float, default=0.5)
    ap.add_argument("--gamma", type=float, default=0.1)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--max-episodes", type=int, default=60)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    extra: dict[str, object] = {}
    if args.kind == "pusht_keypoints":
        extra = {"max_episodes": args.max_episodes, "stride": args.stride}
    cfg = Tier1Config(
        chunk=args.chunk,
        batch=args.batch,
        steps=args.steps,
        blend=args.blend,
        gamma=args.gamma,
        seeds=tuple(args.seeds),
        extra=extra,
    )
    print(f"running Tier-1 kind={args.kind} steps={args.steps} seeds={args.seeds} ...", flush=True)
    res = run_tier1(args.kind, cfg)
    full = build_results(res)
    out = args.out or f"src/dtwarp/eval/results/tier1_{args.kind}.json"
    path = write_results_json(full, out)
    print(f"\nwrote {path}\n")
    print(results_to_readme_block(full))


if __name__ == "__main__":
    main()
