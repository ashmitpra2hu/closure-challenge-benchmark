#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

NASA_CASE = Path("data/NASA_2DWMH")
NASA_HIDDEN = Path("data/_NASA_2DWMH_hidden_for_no_cheat")
NASA_EVAL_POINTS = Path("data/evaluation_points/NASA_2DWMH_points.csv")
OUTPUT_CSV = Path("submissions/codex_no_cheat/NASA_2DWMH.csv")
REPORT_JSON = Path("submissions/codex_no_cheat/NASA_2DWMH_efficiency_report.json")


def parse_openfoam_vector_field(filepath: Path) -> np.ndarray:
    with filepath.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().isdigit():
            start_idx = i + 2
            break
    if start_idx is None:
        raise ValueError(f"Could not find OpenFOAM internal field count in {filepath}")

    data = []
    for line in lines[start_idx:]:
        row = line.strip()
        if row in {");", ")"}:
            break
        if row.startswith("(") and row.endswith(")"):
            vals = row.strip("()").split()
            if len(vals) != 3:
                continue
            data.append([float(vals[0]), float(vals[1]), float(vals[2])])

    return np.asarray(data, dtype=float)


def load_case(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    coords = parse_openfoam_vector_field(case_dir / "0" / "C")
    vel_file = case_dir / "0" / "U_LES"
    try:
        vel = parse_openfoam_vector_field(vel_file)
    except Exception:
        text = vel_file.read_text(encoding="utf-8")
        include_line = None
        for line in text.splitlines():
            row = line.strip()
            if row.startswith('#include') and 'U_internalField' in row:
                include_line = row
                break
        if include_line is None:
            raise
        rel = include_line.split('"')[1]
        vel = parse_openfoam_vector_field(vel_file.parent / rel)

    if len(coords) != len(vel):
        raise ValueError(f"Coordinate/velocity length mismatch in {case_dir}: {len(coords)} vs {len(vel)}")
    return coords, vel


def discover_training_cases(data_root: Path) -> list[Path]:
    cases = []
    for vel in data_root.glob("**/0/U_LES"):
        case_dir = vel.parent.parent
        p = case_dir.as_posix()
        if "NASA_2DWMH" in p or str(NASA_HIDDEN) in p:
            continue
        if (case_dir / "0" / "C").exists():
            cases.append(case_dir)
    return sorted(set(cases))


def filter_training_cases(cases: list[Path], include_patterns: list[str]) -> list[Path]:
    if not include_patterns:
        return cases
    filtered = []
    for case in cases:
        p = case.as_posix()
        if any(token in p for token in include_patterns):
            filtered.append(case)
    return filtered


def sample_case(coords: np.ndarray, vel: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if len(coords) <= n:
        return coords, vel
    idx = rng.choice(len(coords), size=n, replace=False)
    return coords[idx], vel[idx]


def knn_idw_predict(
    train_xyz: np.ndarray,
    train_u: np.ndarray,
    query_xyz: np.ndarray,
    k: int,
    chunk_size: int,
    distance_power: float,
) -> np.ndarray:
    out = np.zeros((len(query_xyz), 3), dtype=float)
    train_xz = train_xyz[:, [0, 2]]
    query_xz = query_xyz[:, [0, 2]]

    for start in range(0, len(query_xz), chunk_size):
        stop = min(start + chunk_size, len(query_xz))
        q = query_xz[start:stop]
        d2 = np.sum((q[:, None, :] - train_xz[None, :, :]) ** 2, axis=2)
        kk = min(k, d2.shape[1])
        idx = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
        d2_k = np.take_along_axis(d2, idx, axis=1)
        w = 1.0 / ((np.sqrt(d2_k) + 1e-12) ** distance_power)
        w /= np.sum(w, axis=1, keepdims=True)
        out[start:stop] = np.einsum("ij,ijk->ik", w, train_u[idx])

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="No-cheat NASA hump generation from non-NASA cases.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples-per-case", type=int, default=2500)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--distance-power", type=float, default=1.0)
    parser.add_argument(
        "--include-patterns",
        nargs="*",
        default=["Parm_PH_29", "PH_Breuer"],
        help="Only train on cases whose path contains any of these tokens.",
    )
    args = parser.parse_args()

    if not NASA_CASE.exists():
        raise FileNotFoundError(f"Expected NASA case at {NASA_CASE}")
    if NASA_HIDDEN.exists():
        raise FileExistsError(f"Hidden NASA folder already exists: {NASA_HIDDEN}")

    rng = np.random.default_rng(args.seed)
    t0 = time.perf_counter()
    train_cases = []
    train_counts = {}
    prediction = None

    shutil.move(str(NASA_CASE), str(NASA_HIDDEN))
    try:
        train_cases = discover_training_cases(Path("data"))
        train_cases = filter_training_cases(train_cases, args.include_patterns)
        if not train_cases:
            raise RuntimeError("No non-NASA training cases found after filtering.")

        xyz_parts, u_parts = [], []
        for case in train_cases:
            coords, vel = load_case(case)
            coords, vel = sample_case(coords, vel, args.samples_per_case, rng)
            xyz_parts.append(coords)
            u_parts.append(vel)
            train_counts[case.as_posix()] = int(len(coords))

        train_xyz = np.vstack(xyz_parts)
        train_u = np.vstack(u_parts)
        query_xyz = np.loadtxt(NASA_EVAL_POINTS, delimiter=",")
        prediction = knn_idw_predict(train_xyz, train_u, query_xyz, args.k, args.chunk_size, args.distance_power)

        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(OUTPUT_CSV, prediction, delimiter=",", fmt="%.18e")
    finally:
        if NASA_HIDDEN.exists() and not NASA_CASE.exists():
            shutil.move(str(NASA_HIDDEN), str(NASA_CASE))

    elapsed = time.perf_counter() - t0

    query_xyz = np.loadtxt(NASA_EVAL_POINTS, delimiter=",")
    nasa_coords, nasa_u = load_case(NASA_CASE)
    truth = knn_idw_predict(nasa_coords, nasa_u, query_xyz, 1, args.chunk_size, 1.0)

    mae = float(np.mean(np.abs(prediction - truth)))
    pps = float(len(query_xyz) / max(elapsed, 1e-12))
    efficiency = float(pps / (1.0 + 100.0 * mae))

    report = {
        "no_cheat": True,
        "nasa_hidden_during_training": True,
        "training_case_count": len(train_cases),
        "training_cases": train_counts,
        "include_patterns": args.include_patterns,
        "prediction_output": OUTPUT_CSV.as_posix(),
        "n_points": int(len(query_xyz)),
        "runtime_seconds": elapsed,
        "points_per_second": pps,
        "mae_vs_restored_nasa": mae,
        "efficiency_score": efficiency,
        "efficiency_formula": "points_per_second / (1 + 100 * mae)",
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
