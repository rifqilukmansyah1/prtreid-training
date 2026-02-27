from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Optional

import pandas as pd


FILENAME_RE = re.compile(r"^(?P<pid>\d+)_(?P<video_id>\d+)_(?P<image_id>\d+)\.jpg$")


def parse_image_path(path: Path) -> dict | None:
    m = FILENAME_RE.match(path.name)
    if m is None:
        return None
    pid = int(m.group("pid"))
    video_id = int(m.group("video_id"))
    image_id = int(m.group("image_id"))
    return {
        "img_path": str(path.resolve()),
        "pid": pid,
        "camid": video_id,
        "video_id": video_id,
        "image_id": image_id,
        "visibility": 1.0,
    }


def build_dataframe(split_dir: Path) -> pd.DataFrame:
    rows = []
    for p in split_dir.rglob("*.jpg"):
        row = parse_image_path(p)
        if row is not None:
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No valid images parsed from {split_dir}")
    df = pd.DataFrame(rows)
    df = df.sort_values(["pid", "video_id", "image_id"]).reset_index(drop=True)
    return df


def split_query_gallery(val_df: pd.DataFrame, query_ratio: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = random.Random(seed)
    query_parts = []
    gallery_parts = []

    for pid, g in val_df.groupby("pid", sort=True):
        idxs = g.index.tolist()
        if len(idxs) == 1:
            gallery_parts.append(g)
            continue
        rng.shuffle(idxs)
        q_count = max(1, int(round(len(idxs) * query_ratio)))
        q_count = min(q_count, len(idxs) - 1)
        q_set = set(idxs[:q_count])
        query_parts.append(g.loc[g.index.isin(q_set)])
        gallery_parts.append(g.loc[~g.index.isin(q_set)])

    query_df = pd.concat(query_parts, ignore_index=True) if query_parts else pd.DataFrame(columns=val_df.columns)
    gallery_df = pd.concat(gallery_parts, ignore_index=True) if gallery_parts else pd.DataFrame(columns=val_df.columns)
    query_df = query_df.sort_values(["pid", "video_id", "image_id"]).reset_index(drop=True)
    gallery_df = gallery_df.sort_values(["pid", "video_id", "image_id"]).reset_index(drop=True)
    return query_df, gallery_df


def load_pid_labels(pid_labels_csv: Optional[str]) -> Optional[pd.DataFrame]:
    if not pid_labels_csv:
        return None
    path = Path(pid_labels_csv)
    if not path.exists():
        raise FileNotFoundError(f"PID labels file not found: {path}")
    df = pd.read_csv(path)
    required = {"pid", "team", "role"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    out = df[["pid", "team", "role"]].copy()
    out["pid"] = out["pid"].astype(int)
    out["team"] = out["team"].astype(str).str.lower().str.strip()
    out["role"] = out["role"].astype(str).str.lower().str.strip()
    out = out.drop_duplicates(subset=["pid"], keep="last")
    return out


def attach_multitask_labels(
    df: pd.DataFrame,
    pid_labels_df: Optional[pd.DataFrame],
    default_team: str,
    default_role: str,
    require_complete: bool,
) -> pd.DataFrame:
    out = df.copy()
    if pid_labels_df is None:
        if require_complete:
            raise ValueError(
                "Multitask labels are required, but --pid_labels_csv is not provided."
            )
        out["team"] = default_team
        out["role"] = default_role
        return out

    out = out.merge(pid_labels_df, on="pid", how="left")
    missing_team = int(out["team"].isna().sum())
    missing_role = int(out["role"].isna().sum())
    if require_complete and (missing_team > 0 or missing_role > 0):
        missing_pids = sorted(out.loc[out["team"].isna() | out["role"].isna(), "pid"].unique().tolist())
        raise ValueError(
            "Missing team/role labels for some PID values. "
            f"missing_count={len(missing_pids)} sample_pids={missing_pids[:20]}"
        )

    out["team"] = out["team"].fillna(default_team).astype(str).str.lower().str.strip()
    out["role"] = out["role"].fillna(default_role).astype(str).str.lower().str.strip()
    return out


def cap_per_pid(df: pd.DataFrame, max_per_pid: int, seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    parts = []
    for _, g in df.groupby("pid", sort=True):
        if len(g) <= max_per_pid:
            parts.append(g)
            continue
        idxs = g.index.tolist()
        rng.shuffle(idxs)
        keep = set(idxs[:max_per_pid])
        parts.append(g.loc[g.index.isin(keep)])
    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["pid", "video_id", "image_id"]).reset_index(drop=True)
    return out


def save_triplet(root: Path, train_df: pd.DataFrame, query_df: pd.DataFrame, gallery_df: pd.DataFrame) -> None:
    split_dir = root / "splits"
    split_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(split_dir / "train.csv", index=False)
    query_df.to_csv(split_dir / "query.csv", index=False)
    gallery_df.to_csv(split_dir / "gallery.csv", index=False)


def print_stats(name: str, train_df: pd.DataFrame, query_df: pd.DataFrame, gallery_df: pd.DataFrame) -> None:
    print(f"\n[{name}] rows: train={len(train_df)}, query={len(query_df)}, gallery={len(gallery_df)}")
    print(
        f"[{name}] pids: train={train_df.pid.nunique()}, query={query_df.pid.nunique()}, gallery={gallery_df.pid.nunique()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser("Build CSV manifests from existing reid image crops")
    parser.add_argument("--source_dir", type=str, default="data/reid/images")
    parser.add_argument("--output_root", type=str, default="data/processed")
    parser.add_argument("--dataset_name", type=str, default="reid")
    parser.add_argument("--query_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--build_smoke", action="store_true")
    parser.add_argument("--smoke_dataset_name", type=str, default="reid_smoke")
    parser.add_argument("--smoke_max_train_per_pid", type=int, default=12)
    parser.add_argument("--smoke_max_eval_per_pid", type=int, default=6)
    parser.add_argument("--pid_labels_csv", type=str, default="")
    parser.add_argument("--default_team", type=str, default="other")
    parser.add_argument("--default_role", type=str, default="player")
    parser.add_argument("--require_multitask_labels", action="store_true")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    train_dir = source_dir / "train"
    val_dir = source_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Expected train/val folders under {source_dir}")

    print("[build] scanning train images ...")
    train_df = build_dataframe(train_dir)
    print("[build] scanning val images ...")
    val_df = build_dataframe(val_dir)

    query_df, gallery_df = split_query_gallery(val_df, query_ratio=float(args.query_ratio), seed=int(args.seed))
    pid_labels_df = load_pid_labels(args.pid_labels_csv)
    train_df = attach_multitask_labels(
        train_df,
        pid_labels_df=pid_labels_df,
        default_team=str(args.default_team).lower().strip(),
        default_role=str(args.default_role).lower().strip(),
        require_complete=bool(args.require_multitask_labels),
    )
    query_df = attach_multitask_labels(
        query_df,
        pid_labels_df=pid_labels_df,
        default_team=str(args.default_team).lower().strip(),
        default_role=str(args.default_role).lower().strip(),
        require_complete=bool(args.require_multitask_labels),
    )
    gallery_df = attach_multitask_labels(
        gallery_df,
        pid_labels_df=pid_labels_df,
        default_team=str(args.default_team).lower().strip(),
        default_role=str(args.default_role).lower().strip(),
        require_complete=bool(args.require_multitask_labels),
    )

    out_root = Path(args.output_root) / args.dataset_name
    save_triplet(out_root, train_df, query_df, gallery_df)
    print_stats(args.dataset_name, train_df, query_df, gallery_df)
    print(f"[build] saved: {out_root / 'splits'}")

    if args.build_smoke:
        s_train = cap_per_pid(train_df, int(args.smoke_max_train_per_pid), seed=int(args.seed))
        s_query = cap_per_pid(query_df, int(args.smoke_max_eval_per_pid), seed=int(args.seed))
        s_gallery = cap_per_pid(gallery_df, int(args.smoke_max_eval_per_pid), seed=int(args.seed))

        # Keep only pids present in both query and gallery for stable eval.
        valid_eval_pids = set(s_query.pid.unique()).intersection(set(s_gallery.pid.unique()))
        s_query = s_query[s_query.pid.isin(valid_eval_pids)].reset_index(drop=True)
        s_gallery = s_gallery[s_gallery.pid.isin(valid_eval_pids)].reset_index(drop=True)

        smoke_root = Path(args.output_root) / args.smoke_dataset_name
        save_triplet(smoke_root, s_train, s_query, s_gallery)
        print_stats(args.smoke_dataset_name, s_train, s_query, s_gallery)
        print(f"[build] saved: {smoke_root / 'splits'}")


if __name__ == "__main__":
    main()
