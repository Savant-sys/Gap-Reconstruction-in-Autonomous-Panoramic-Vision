#!/usr/bin/env python3
"""
compile_pipeline_results.py

Compile summary results across multiple full-pipeline output folders, such as:

  full_pipeline_output_dir_val_mask_0.5/
  full_pipeline_output_dir_val_mask_0.7/
  full_pipeline_output_dir_val_mask_0.9/
  full_pipeline_output_dir_val_mask_1.2/
  full_pipeline_output_dir_val_mask_1.5/
  full_pipeline_output_dir_val_mask_2.0/

Each folder is expected to contain files produced by run_full_waymo_gap_pipeline_dir.py:
  - pipeline_manifest.csv
  - all_yolo_summary_rows.csv
  - all_yolo_detections.csv
  - aggregate_yolo_counts.csv

This script creates a compiled results folder with:
  - compiled_mode_summary.csv
  - compiled_sample_recovery.csv
  - compiled_recovery_summary.csv
  - compiled_class_summary.csv
  - compiled_manifest_summary.csv
  - optional plots

Example:
  python compile_pipeline_results.py \
    --root_dir /Users/lindseyraven/Desktop/MSAI_SJSU_Work/CMPE297_CV/Project/New_Pipeline \
    --pattern "full_pipeline_output_dir_val_mask_*" \
    --output_dir compiled_results
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd


VARIANT_ORDER = ["original", "masked", "inpainted"]


def ensure_dir(path: str | Path) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def parse_mask_value(path: Path) -> float:
    m = re.search(r"mask[_-]([0-9]+(?:\.[0-9]+)?)", path.name)
    if m:
        return float(m.group(1))
    return float("nan")


def read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[WARN] Missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        print(f"[WARN] Empty file: {path}")
        return None


def discover_runs(root_dir: Path, pattern: str) -> List[Path]:
    return sorted([p for p in root_dir.glob(pattern) if p.is_dir()])


def coerce_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def summarize_manifest(run_dir: Path, mask_value: float) -> Dict[str, Any]:
    manifest_path = run_dir / "pipeline_manifest.csv"
    manifest = read_csv_if_exists(manifest_path)

    if manifest is None:
        return {
            "run_name": run_dir.name,
            "mask_value": mask_value,
            "num_samples": 0,
            "num_segments": 0,
            "manifest_path": str(manifest_path),
        }

    num_samples = len(manifest)

    if "segment_file" in manifest.columns:
        num_segments = manifest["segment_file"].nunique()
    elif "camera_image_parquet" in manifest.columns:
        num_segments = manifest["camera_image_parquet"].nunique()
    else:
        num_segments = num_samples

    return {
        "run_name": run_dir.name,
        "mask_value": mask_value,
        "num_samples": int(num_samples),
        "num_segments": int(num_segments),
        "manifest_path": str(manifest_path),
    }


def load_summary_rows(run_dir: Path, mask_value: float) -> Optional[pd.DataFrame]:
    path = run_dir / "all_yolo_summary_rows.csv"
    df = read_csv_if_exists(path)
    if df is None:
        return None

    df["run_name"] = run_dir.name
    df["mask_value"] = mask_value

    df = coerce_numeric(
        df,
        [
            "count",
            "mean_confidence",
            "max_confidence",
            "total_box_area",
            "frame_index",
            "segment_index",
        ],
    )

    return df


def load_detections(run_dir: Path, mask_value: float) -> Optional[pd.DataFrame]:
    path = run_dir / "all_yolo_detections.csv"
    df = read_csv_if_exists(path)
    if df is None:
        return None

    df["run_name"] = run_dir.name
    df["mask_value"] = mask_value

    df = coerce_numeric(
        df,
        [
            "confidence",
            "x1", "y1", "x2", "y2",
            "width", "height", "area",
            "frame_index",
            "segment_index",
        ],
    )

    return df


def build_mode_summary(summary_all: pd.DataFrame) -> pd.DataFrame:
    df = summary_all.copy()

    if "variant" not in df.columns and "mode" in df.columns:
        df["variant"] = df["mode"]

    df = df[df["class_name"] == "__ALL__"].copy()

    grouped = (
        df.groupby(["mask_value", "run_name", "variant"], dropna=False)
        .agg(
            num_samples=("count", "size"),
            total_detections=("count", "sum"),
            mean_detections_per_sample=("count", "mean"),
            std_detections_per_sample=("count", "std"),
            mean_confidence=("mean_confidence", "mean"),
            mean_total_box_area=("total_box_area", "mean"),
        )
        .reset_index()
    )

    grouped["variant"] = pd.Categorical(grouped["variant"], VARIANT_ORDER, ordered=True)
    grouped = grouped.sort_values(["mask_value", "variant"]).reset_index(drop=True)
    return grouped


def build_sample_recovery(summary_all: pd.DataFrame) -> pd.DataFrame:
    df = summary_all.copy()

    if "variant" not in df.columns and "mode" in df.columns:
        df["variant"] = df["mode"]

    df = df[df["class_name"] == "__ALL__"].copy()

    if "sample_id" in df.columns:
        sample_col = "sample_id"
    elif "sample_dir" in df.columns:
        sample_col = "sample_dir"
    else:
        df["sample_id_fallback"] = df.groupby(["run_name", "mask_value", "variant"]).cumcount()
        sample_col = "sample_id_fallback"

    pivot_count = df.pivot_table(
        index=["mask_value", "run_name", sample_col],
        columns="variant",
        values="count",
        aggfunc="first",
    ).reset_index()

    pivot_conf = df.pivot_table(
        index=["mask_value", "run_name", sample_col],
        columns="variant",
        values="mean_confidence",
        aggfunc="first",
    ).reset_index()

    for v in VARIANT_ORDER:
        if v not in pivot_count.columns:
            pivot_count[v] = np.nan
        if v not in pivot_conf.columns:
            pivot_conf[v] = np.nan

    out = pivot_count.rename(
        columns={
            "original": "original_count",
            "masked": "masked_count",
            "inpainted": "inpainted_count",
        }
    )

    conf_small = pivot_conf[
        ["mask_value", "run_name", sample_col, "original", "masked", "inpainted"]
    ].rename(
        columns={
            "original": "original_confidence",
            "masked": "masked_confidence",
            "inpainted": "inpainted_confidence",
        }
    )

    out = out.merge(conf_small, on=["mask_value", "run_name", sample_col], how="left")

    out["masked_drop"] = out["original_count"] - out["masked_count"]
    out["recovery_gain"] = out["inpainted_count"] - out["masked_count"]

    out["recovery_ratio"] = np.where(
        out["masked_drop"] > 0,
        out["recovery_gain"] / out["masked_drop"],
        np.nan,
    )

    out["masked_conf_drop"] = out["original_confidence"] - out["masked_confidence"]
    out["inpaint_conf_gain"] = out["inpainted_confidence"] - out["masked_confidence"]

    out = out.rename(columns={sample_col: "sample_id"})
    out = out.sort_values(["mask_value", "sample_id"]).reset_index(drop=True)
    return out


def build_recovery_summary(sample_recovery: pd.DataFrame) -> pd.DataFrame:
    return (
        sample_recovery.groupby(["mask_value", "run_name"], dropna=False)
        .agg(
            num_samples=("sample_id", "size"),
            mean_original_count=("original_count", "mean"),
            mean_masked_count=("masked_count", "mean"),
            mean_inpainted_count=("inpainted_count", "mean"),
            mean_masked_drop=("masked_drop", "mean"),
            mean_recovery_gain=("recovery_gain", "mean"),
            mean_recovery_ratio=("recovery_ratio", "mean"),
            median_recovery_ratio=("recovery_ratio", "median"),
            mean_masked_conf_drop=("masked_conf_drop", "mean"),
            mean_inpaint_conf_gain=("inpaint_conf_gain", "mean"),
        )
        .reset_index()
        .sort_values("mask_value")
    )


def build_class_summary(summary_all: pd.DataFrame) -> pd.DataFrame:
    df = summary_all.copy()

    if "variant" not in df.columns and "mode" in df.columns:
        df["variant"] = df["mode"]

    df = df[df["class_name"] != "__ALL__"].copy()

    if len(df) == 0:
        return pd.DataFrame()

    grouped = (
        df.groupby(["mask_value", "run_name", "variant", "class_name"], dropna=False)
        .agg(
            total_class_detections=("count", "sum"),
            mean_class_detections_per_sample=("count", "mean"),
            mean_confidence=("mean_confidence", "mean"),
            max_confidence=("max_confidence", "max"),
            mean_total_box_area=("total_box_area", "mean"),
        )
        .reset_index()
    )

    grouped["variant"] = pd.Categorical(grouped["variant"], VARIANT_ORDER, ordered=True)
    grouped = grouped.sort_values(
        ["mask_value", "variant", "total_class_detections"],
        ascending=[True, True, False],
    )
    return grouped


def make_plots(mode_summary: pd.DataFrame, recovery_summary: pd.DataFrame, output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] Could not import matplotlib; skipping plots. Error: {e}")
        return

    plt.figure(figsize=(7, 4))
    for variant in VARIANT_ORDER:
        sub = mode_summary[mode_summary["variant"] == variant]
        if len(sub) == 0:
            continue
        plt.plot(sub["mask_value"], sub["mean_detections_per_sample"], marker="o", label=variant)
    plt.xlabel("Mask setting")
    plt.ylabel("Mean detections per sample")
    plt.title("Detection Count vs. Mask Setting")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_detection_count_vs_mask.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 4))
    for variant in VARIANT_ORDER:
        sub = mode_summary[mode_summary["variant"] == variant]
        if len(sub) == 0:
            continue
        plt.plot(sub["mask_value"], sub["mean_confidence"], marker="o", label=variant)
    plt.xlabel("Mask setting")
    plt.ylabel("Mean YOLO confidence")
    plt.title("Mean Confidence vs. Mask Setting")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot_confidence_vs_mask.png", dpi=200)
    plt.close()

    if len(recovery_summary) > 0:
        plt.figure(figsize=(7, 4))
        plt.plot(recovery_summary["mask_value"], recovery_summary["mean_recovery_ratio"], marker="o")
        plt.axhline(0.0, linestyle="--", linewidth=1)
        plt.xlabel("Mask setting")
        plt.ylabel("Mean recovery ratio")
        plt.title("Recovery Ratio vs. Mask Setting")
        plt.tight_layout()
        plt.savefig(output_dir / "plot_recovery_ratio_vs_mask.png", dpi=200)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", default=".", help="Directory containing full_pipeline_output folders.")
    parser.add_argument("--pattern", default="full_pipeline_output_dir_val_mask_*")
    parser.add_argument("--runs", nargs="*", default=None, help="Optional explicit run directories.")
    parser.add_argument("--output_dir", default="compiled_results")
    parser.add_argument("--no_plots", action="store_true")

    args = parser.parse_args()

    root_dir = Path(args.root_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    if args.runs:
        run_dirs = [Path(r).resolve() for r in args.runs]
    else:
        run_dirs = discover_runs(root_dir, args.pattern)

    if not run_dirs:
        raise RuntimeError(f"No run directories found under {root_dir} with pattern {args.pattern}")

    print("[INFO] Run directories:")
    for r in run_dirs:
        print(f"  {r.name}")

    manifest_rows = []
    summary_frames = []
    detection_frames = []

    for run_dir in run_dirs:
        mask_value = parse_mask_value(run_dir)
        manifest_rows.append(summarize_manifest(run_dir, mask_value))

        summary = load_summary_rows(run_dir, mask_value)
        if summary is not None:
            summary_frames.append(summary)

        detections = load_detections(run_dir, mask_value)
        if detections is not None:
            detection_frames.append(detections)

    manifest_summary = pd.DataFrame(manifest_rows).sort_values("mask_value")

    if not summary_frames:
        raise RuntimeError("No all_yolo_summary_rows.csv files could be loaded.")

    summary_all = pd.concat(summary_frames, ignore_index=True)

    mode_summary = build_mode_summary(summary_all)
    sample_recovery = build_sample_recovery(summary_all)
    recovery_summary = build_recovery_summary(sample_recovery)
    class_summary = build_class_summary(summary_all)

    detections_all = pd.concat(detection_frames, ignore_index=True) if detection_frames else pd.DataFrame()

    manifest_summary.to_csv(output_dir / "compiled_manifest_summary.csv", index=False)
    summary_all.to_csv(output_dir / "compiled_all_yolo_summary_rows.csv", index=False)
    mode_summary.to_csv(output_dir / "compiled_mode_summary.csv", index=False)
    sample_recovery.to_csv(output_dir / "compiled_sample_recovery.csv", index=False)
    recovery_summary.to_csv(output_dir / "compiled_recovery_summary.csv", index=False)
    class_summary.to_csv(output_dir / "compiled_class_summary.csv", index=False)

    if len(detections_all) > 0:
        detections_all.to_csv(output_dir / "compiled_all_yolo_detections.csv", index=False)

    if not args.no_plots:
        make_plots(mode_summary, recovery_summary, output_dir)

    print("\n[DONE] Wrote compiled results to:")
    print(f"  {output_dir}")
    print("\nMain files:")
    print(f"  {output_dir / 'compiled_manifest_summary.csv'}")
    print(f"  {output_dir / 'compiled_mode_summary.csv'}")
    print(f"  {output_dir / 'compiled_sample_recovery.csv'}")
    print(f"  {output_dir / 'compiled_recovery_summary.csv'}")
    print(f"  {output_dir / 'compiled_class_summary.csv'}")

    print("\n[MODE SUMMARY]")
    print(mode_summary.to_string(index=False))

    print("\n[RECOVERY SUMMARY]")
    print(recovery_summary.to_string(index=False))


if __name__ == "__main__":
    main()
