from __future__ import annotations

import argparse
import random
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Convert CVAT for video 1.1 XML annotations into ReID crop folders"
    )
    parser.add_argument("--cvat_xml", type=str, required=True, help="Path to CVAT XML export file")
    parser.add_argument(
        "--frames_dir",
        type=str,
        required=True,
        help="Directory of extracted video frames (jpg/png)",
    )
    parser.add_argument(
        "--output_images_dir",
        type=str,
        default="data/reid/images",
        help="Output root that will contain train/ and val/ folders",
    )
    parser.add_argument(
        "--pid_labels_csv",
        type=str,
        default="data/processed/reid/pid_labels_from_cvat.csv",
        help="Output CSV mapping pid -> team, role",
    )
    parser.add_argument("--video_id", type=int, default=1, help="Camera/video id used in filename")
    parser.add_argument(
        "--frame_pattern",
        type=str,
        default="{frame:06d}.jpg",
        help="Frame filename pattern; examples: '{frame:06d}.jpg', 'img_{frame:05d}.png'",
    )
    parser.add_argument(
        "--frame_index_offset",
        type=int,
        default=0,
        help="Add this offset before formatting frame file name (use 1 when frames start at 1)",
    )
    parser.add_argument(
        "--train_pid_ratio",
        type=float,
        default=0.8,
        help="Ratio of person IDs assigned to train split (remaining IDs go to val)",
    )
    parser.add_argument(
        "--min_track_boxes",
        type=int,
        default=5,
        help="Minimum visible boxes per track to keep that PID",
    )
    parser.add_argument(
        "--min_box_area",
        type=float,
        default=100.0,
        help="Ignore very small boxes (in pixel area)",
    )
    parser.add_argument(
        "--crop_padding",
        type=float,
        default=0.05,
        help="Relative padding added around bbox before crop (0.05 = 5%)",
    )
    parser.add_argument("--jpeg_quality", type=int, default=95, help="JPEG quality for saved crops")
    parser.add_argument("--default_team", type=str, default="other", help="Fallback team label")
    parser.add_argument("--default_role", type=str, default="player", help="Fallback role label")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for PID split")
    return parser.parse_args()


def _safe_int(text: Optional[str], default: int = 0) -> int:
    try:
        return int(text) if text is not None else default
    except Exception:
        return default


def _safe_float(text: Optional[str], default: float = 0.0) -> float:
    try:
        return float(text) if text is not None else default
    except Exception:
        return default


def _normalize_team(value: str, default_team: str) -> str:
    v = (value or "").strip().lower()
    if v in {"left", "right", "other"}:
        return v
    return default_team


def _normalize_role(value: str, default_role: str) -> str:
    v = (value or "").strip().lower()
    # Keep the common SoccerNet/TrackLab-like role vocabulary.
    if v in {"player", "goalkeeper", "referee", "ball", "other"}:
        return v
    return default_role


def parse_cvat_video_xml(
    xml_path: Path,
    default_team: str,
    default_role: str,
    min_box_area: float,
) -> Dict[int, List[dict]]:
    """
    Parse CVAT video XML and return per-pid visible boxes.

    Expected XML style:
      <track id="12" label="player"> <box frame="..." outside="0" ...> ... </box> ... </track>
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    pid_records: Dict[int, List[dict]] = defaultdict(list)

    for track in root.findall(".//track"):
        pid = _safe_int(track.attrib.get("id"), default=-1)
        if pid < 0:
            continue

        track_label = (track.attrib.get("label") or "").strip().lower()

        for box in track.findall("box"):
            outside = _safe_int(box.attrib.get("outside"), default=0)
            if outside == 1:
                continue

            frame_idx = _safe_int(box.attrib.get("frame"), default=-1)
            if frame_idx < 0:
                continue

            xtl = _safe_float(box.attrib.get("xtl"))
            ytl = _safe_float(box.attrib.get("ytl"))
            xbr = _safe_float(box.attrib.get("xbr"))
            ybr = _safe_float(box.attrib.get("ybr"))
            w = max(0.0, xbr - xtl)
            h = max(0.0, ybr - ytl)
            if w * h < min_box_area:
                continue

            # CVAT attributes may exist per-box; fallback to track label/defaults.
            attr_map = {}
            for attr in box.findall("attribute"):
                name = (attr.attrib.get("name") or "").strip().lower()
                val = (attr.text or "").strip()
                if name:
                    attr_map[name] = val

            raw_team = attr_map.get("team", default_team)
            raw_role = attr_map.get("role", track_label if track_label else default_role)
            team = _normalize_team(raw_team, default_team=default_team)
            role = _normalize_role(raw_role, default_role=default_role)

            pid_records[pid].append(
                {
                    "pid": pid,
                    "frame": frame_idx,
                    "xtl": xtl,
                    "ytl": ytl,
                    "xbr": xbr,
                    "ybr": ybr,
                    "team": team,
                    "role": role,
                }
            )

    return pid_records


def split_pids(pid_records: Dict[int, List[dict]], train_ratio: float, min_track_boxes: int, seed: int) -> tuple[set, set]:
    # Drop PIDs with too few visible samples.
    kept = [pid for pid, recs in pid_records.items() if len(recs) >= min_track_boxes]
    if len(kept) < 2:
        raise RuntimeError(
            f"Not enough valid PIDs after filtering: {len(kept)} (need >=2). "
            "Lower --min_track_boxes or check annotations."
        )

    rng = random.Random(seed)
    rng.shuffle(kept)
    n_train = max(1, int(round(len(kept) * train_ratio)))
    n_train = min(n_train, len(kept) - 1)

    train_pids = set(kept[:n_train])
    val_pids = set(kept[n_train:])
    return train_pids, val_pids


def frame_path(frames_dir: Path, pattern: str, frame_idx: int, frame_index_offset: int) -> Path:
    return frames_dir / pattern.format(frame=frame_idx + frame_index_offset)


def crop_with_padding(img, xtl: float, ytl: float, xbr: float, ybr: float, padding_ratio: float):
    h, w = img.shape[:2]
    bw = max(1.0, xbr - xtl)
    bh = max(1.0, ybr - ytl)
    padx = bw * padding_ratio
    pady = bh * padding_ratio

    x1 = int(max(0, round(xtl - padx)))
    y1 = int(max(0, round(ytl - pady)))
    x2 = int(min(w - 1, round(xbr + padx)))
    y2 = int(min(h - 1, round(ybr + pady)))

    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]


def mode_or_default(values: List[str], default_value: str) -> str:
    if not values:
        return default_value
    c = Counter(values)
    return c.most_common(1)[0][0]


def main() -> None:
    args = parse_args()

    cvat_xml = Path(args.cvat_xml)
    frames_dir = Path(args.frames_dir)
    output_images_dir = Path(args.output_images_dir)
    labels_csv = Path(args.pid_labels_csv)

    if not cvat_xml.exists():
        raise FileNotFoundError(f"CVAT XML not found: {cvat_xml}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    default_team = _normalize_team(args.default_team, default_team="other")
    default_role = _normalize_role(args.default_role, default_role="player")

    print("[cvat] parsing xml ...")
    pid_records = parse_cvat_video_xml(
        xml_path=cvat_xml,
        default_team=default_team,
        default_role=default_role,
        min_box_area=float(args.min_box_area),
    )

    print("[cvat] splitting pids train/val ...")
    train_pids, val_pids = split_pids(
        pid_records=pid_records,
        train_ratio=float(args.train_pid_ratio),
        min_track_boxes=int(args.min_track_boxes),
        seed=int(args.seed),
    )

    train_dir = output_images_dir / "train" / f"{int(args.video_id):03d}"
    val_dir = output_images_dir / "val" / f"{int(args.video_id):03d}"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    # Flatten all kept records and sort deterministically.
    all_records = []
    for pid, recs in pid_records.items():
        if pid in train_pids or pid in val_pids:
            all_records.extend(recs)
    all_records.sort(key=lambda x: (x["frame"], x["pid"]))

    # Read each frame once, then crop all boxes in that frame.
    by_frame: Dict[int, List[dict]] = defaultdict(list)
    for r in all_records:
        by_frame[r["frame"]].append(r)

    image_counter = 0
    save_count_train = 0
    save_count_val = 0
    pid_team: Dict[int, List[str]] = defaultdict(list)
    pid_role: Dict[int, List[str]] = defaultdict(list)

    print("[cvat] extracting crops ...")
    for frame_idx in sorted(by_frame.keys()):
        fpath = frame_path(
            frames_dir=frames_dir,
            pattern=args.frame_pattern,
            frame_idx=frame_idx,
            frame_index_offset=int(args.frame_index_offset),
        )
        if not fpath.exists():
            # Skip silently to tolerate missing frames; we still report at the end.
            continue

        img = cv2.imread(str(fpath))
        if img is None:
            continue

        for rec in by_frame[frame_idx]:
            pid = int(rec["pid"])
            crop = crop_with_padding(
                img=img,
                xtl=rec["xtl"],
                ytl=rec["ytl"],
                xbr=rec["xbr"],
                ybr=rec["ybr"],
                padding_ratio=float(args.crop_padding),
            )
            if crop is None or crop.size == 0:
                continue

            image_id = int(args.video_id) * 10_000_000 + frame_idx * 100 + (image_counter % 100)
            fname = f"{pid}_{int(args.video_id)}_{image_id}.jpg"
            split_dir = train_dir if pid in train_pids else val_dir
            out_path = split_dir / fname

            ok = cv2.imwrite(str(out_path), crop, [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)])
            if not ok:
                continue

            image_counter += 1
            if pid in train_pids:
                save_count_train += 1
            else:
                save_count_val += 1

            pid_team[pid].append(rec["team"])
            pid_role[pid].append(rec["role"])

    # Build pid labels mapping with majority vote.
    pid_rows = []
    for pid in sorted(list(train_pids.union(val_pids))):
        team = mode_or_default(pid_team.get(pid, []), default_team)
        role = mode_or_default(pid_role.get(pid, []), default_role)
        pid_rows.append({"pid": pid, "team": team, "role": role})

    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pid_rows).to_csv(labels_csv, index=False)

    print("\n[done] CVAT -> ReID crops completed")
    print(f"[done] output train dir: {train_dir}")
    print(f"[done] output val dir:   {val_dir}")
    print(f"[done] saved crops: train={save_count_train}, val={save_count_val}, total={save_count_train + save_count_val}")
    print(f"[done] pid labels csv: {labels_csv} (rows={len(pid_rows)})")
    print(
        "[next] run build_manifests_from_reid.py with --pid_labels_csv to generate PRTReid split CSV files"
    )


if __name__ == "__main__":
    main()
