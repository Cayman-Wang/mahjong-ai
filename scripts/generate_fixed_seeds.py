"""Generate a fixed seed set for reproducible evaluation."""

from __future__ import annotations

import argparse
import json


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n", type=int, default=200)
    args = p.parse_args()

    seeds = [args.seed + i for i in range(args.n)]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"seeds": seeds}, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
