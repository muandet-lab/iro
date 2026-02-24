#!/usr/bin/env python3
"""Collect CMNIST sweep results from iro JSONL artifacts.

Model selection mirrors the legacy flow:
- group runs by algorithm + args_id
- select args_id with best mean accuracy on model-selection env
- report mean/std over seeds for requested test envs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev


def env_key(value: float, *, ms_type: str) -> str:
    return f"{value:g}_acc_{ms_type}"


def as_env_label(value: float) -> str:
    return f"{value:g}"


def is_oracle_row(record: dict) -> bool:
    if str(record.get("algorithm", "")).lower() != "erm":
        return False
    args = record.get("args") or {}
    data = args.get("data") or {}
    envs = data.get("cmnist_train_envs")
    if isinstance(envs, str):
        return envs.strip().lower() == "gray"
    if isinstance(envs, list):
        return [float(x) for x in envs] == [0.5, 0.5]
    return False


def canonical_alg(record: dict) -> str:
    if is_oracle_row(record):
        return "oracle"
    name = str(record.get("algorithm", "")).lower()
    if name == "inftask":
        return "inf-task"
    if name == "iid":
        return "bayes_erm_iid"
    return name


def load_records(results_dir: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(results_dir.glob("*.jsonl")):
        if not path.is_file() or path.stat().st_size == 0:
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") != "ok":
                    continue
                if "args_id" not in rec:
                    continue
                if "algorithm" not in rec:
                    continue
                records.append(rec)
    return records


def group_by_alg_args(records: list[dict]) -> dict[str, dict[str, list[dict]]]:
    grouped: dict[str, dict[str, list[dict]]] = {}
    for rec in records:
        alg = canonical_alg(rec)
        args_id = str(rec["args_id"])
        grouped.setdefault(alg, {}).setdefault(args_id, []).append(rec)
    return grouped


def pick_best_args_id(
    groups: dict[str, list[dict]],
    *,
    select_env: float,
    ms_type: str,
) -> str | None:
    key = env_key(select_env, ms_type=ms_type)
    best_id: str | None = None
    best_score: float | None = None

    for args_id, recs in groups.items():
        vals = [float(r[key]) for r in recs if key in r]
        if not vals:
            continue
        score = mean(vals)
        if best_score is None or score > best_score:
            best_score = score
            best_id = args_id
    return best_id


def summarize_group(recs: list[dict], envs: list[float], *, ms_type: str) -> list[str]:
    out: list[str] = []
    for env in envs:
        key = env_key(env, ms_type=ms_type)
        vals = [float(r[key]) for r in recs if key in r]
        if not vals:
            out.append("NA")
            continue
        mu = mean(vals)
        if len(vals) > 1:
            sd = stdev(vals)
        else:
            sd = 0.0
        out.append(f"{mu:.3f} +- {sd:.3f}")
    return out


def print_table(headers: list[str], rows: dict[str, list[str]]) -> None:
    col0w = max([len("algorithm")] + [len(k) for k in rows.keys()])
    colw = 16

    def fmt(parts: list[str]) -> str:
        return "  ".join(
            [parts[0].ljust(col0w)] + [p.ljust(colw) for p in parts[1:]]
        )

    print(fmt(["algorithm"] + headers))
    for alg in sorted(rows.keys()):
        print(fmt([alg] + rows[alg]))


def main() -> None:
    p = argparse.ArgumentParser(description="Collect CMNIST results into a table.")
    p.add_argument("results_dir", type=Path, help="Directory with per-run JSONL files.")
    p.add_argument(
        "--model-selection-env",
        type=float,
        default=0.9,
        help="Environment value used for selecting best hyperparameters.",
    )
    p.add_argument(
        "--model-selection-type",
        type=str,
        choices=["best", "final"],
        default="best",
    )
    p.add_argument(
        "--test-envs",
        type=str,
        default="all",
        help="Comma-separated env list or 'all' for 0.0..1.0.",
    )
    p.add_argument(
        "--algorithms",
        type=str,
        default="",
        help="Comma-separated canonical algorithm names to include.",
    )
    args = p.parse_args()

    results_dir = args.results_dir
    if not results_dir.exists() or not results_dir.is_dir():
        raise SystemExit(f"results_dir not found: {results_dir}")

    if args.test_envs.strip().lower() == "all":
        envs = [i / 10.0 for i in range(11)]
    else:
        envs = [float(x.strip()) for x in args.test_envs.split(",") if x.strip()]
    env_headers = [as_env_label(v) for v in envs]

    allowed_algs = {x.strip().lower() for x in args.algorithms.split(",") if x.strip()}

    records = load_records(results_dir)
    if not records:
        raise SystemExit("No successful JSONL records found.")

    grouped = group_by_alg_args(records)
    rows: dict[str, list[str]] = {}

    for alg, by_args in grouped.items():
        if allowed_algs and alg not in allowed_algs:
            continue
        best_args_id = pick_best_args_id(
            by_args,
            select_env=args.model_selection_env,
            ms_type=args.model_selection_type,
        )
        if best_args_id is None:
            continue
        rows[alg] = summarize_group(
            by_args[best_args_id],
            envs,
            ms_type=args.model_selection_type,
        )

    if not rows:
        raise SystemExit("No rows to print after filtering/model-selection.")

    print_table(env_headers, rows)


if __name__ == "__main__":
    main()
