from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser("Generate PID->team/role template CSV for multitask training")
    parser.add_argument("--train_csv", type=str, default="data/processed/reid/splits/train.csv")
    parser.add_argument("--output_csv", type=str, default="data/processed/reid/pid_labels_template.csv")
    args = parser.parse_args()

    train_csv = Path(args.train_csv)
    if not train_csv.exists():
        raise FileNotFoundError(f"train.csv not found: {train_csv}")

    df = pd.read_csv(train_csv)
    if "pid" not in df.columns:
        raise ValueError(f"{train_csv} missing 'pid' column")

    # One row per PID, to be filled manually by user:
    # team in {left,right,other}, role in {player,goalkeeper,referee,ball,other}
    out = pd.DataFrame({"pid": sorted(df["pid"].astype(int).unique().tolist())})
    out["team"] = ""
    out["role"] = ""

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"[ok] template saved: {output_csv}")
    print(f"[ok] total pid rows: {len(out)}")


if __name__ == "__main__":
    main()
