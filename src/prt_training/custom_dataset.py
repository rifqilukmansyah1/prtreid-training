from __future__ import annotations

import os
import os.path as osp
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from prtreid.data import ImageDataset


class CSVReIDDataset(ImageDataset):
    """
    Generic dataset loader for PRTReid using CSV manifests.

    Expected files:
      <root>/<dataset_name>/splits/train.csv
      <root>/<dataset_name>/splits/query.csv
      <root>/<dataset_name>/splits/gallery.csv

    Required columns per CSV:
      img_path, pid, camid

    Optional columns:
      team, role, video_id, image_id, visibility, masks_path
    """

    dataset_dir = "custom_reid"
    # Optional registry for future mask folders:
    # {"masks_subdir_name": (parts_num, has_background, file_suffix)}
    masks_dirs: Dict[str, Tuple[int, bool, str]] = {}

    @staticmethod
    def get_masks_config(masks_dir: str):
        """PRTReid compatibility hook used by datamanager/masks utilities."""
        if not masks_dir:
            return None
        return CSVReIDDataset.masks_dirs.get(masks_dir)

    def __init__(
        self,
        root: str = "",
        dataset_name: str = "custom_reid",
        require_team_role: bool = False,
        default_team: str = "left",
        default_role: str = "player",
        **kwargs: Any,
    ) -> None:
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_name = dataset_name
        self.dataset_dir = osp.join(self.root, dataset_name)
        self.require_team_role = require_team_role
        self.default_team = default_team
        self.default_role = default_role

        split_dir = Path(self.dataset_dir) / "splits"
        train_df = self._load_split(split_dir / "train.csv", split_name="train")
        query_df = self._load_split(split_dir / "query.csv", split_name="query")
        gallery_df = self._load_split(split_dir / "gallery.csv", split_name="gallery")

        # Keep train pids contiguous as expected by classification heads.
        train_df = train_df.copy()
        train_df["pid"] = pd.factorize(train_df["pid"])[0].astype(int)

        # Encode categorical task columns and keep reverse map for PrtreidSampler.
        self.column_mapping: Dict[str, Dict[int, str]] = {}
        train_df, query_df, gallery_df = self._encode_multitask_columns(
            train_df, query_df, gallery_df
        )

        train = train_df.to_dict("records")
        query = query_df.to_dict("records")
        gallery = gallery_df.to_dict("records")

        super().__init__(train, query, gallery, **kwargs)

    def _load_split(self, csv_path: Path, split_name: str) -> pd.DataFrame:
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing split file: {csv_path}")

        df = pd.read_csv(csv_path)
        required = {"img_path", "pid", "camid"}
        missing = required.difference(df.columns)
        if missing:
            raise ValueError(f"{csv_path} missing required columns: {sorted(missing)}")

        if self.require_team_role:
            for col in ("team", "role"):
                if col not in df.columns:
                    raise ValueError(
                        f"{csv_path} missing '{col}' but multitask mode requires it"
                    )

        # Fill optional columns with safe defaults so PRTReid engines can read keys consistently.
        if "team" not in df.columns:
            df["team"] = self.default_team
        if "role" not in df.columns:
            df["role"] = self.default_role
        if "video_id" not in df.columns:
            df["video_id"] = df["camid"]
        if "image_id" not in df.columns:
            df["image_id"] = list(range(len(df)))
        if "visibility" not in df.columns:
            df["visibility"] = 1.0
        if "masks_path" not in df.columns:
            df["masks_path"] = ""

        # Normalize primitive dtypes.
        df["pid"] = df["pid"].astype(int)
        df["camid"] = df["camid"].astype(int)
        df["video_id"] = df["video_id"].astype(int)
        df["image_id"] = df["image_id"].astype(int)
        df["visibility"] = df["visibility"].astype(float)

        # Resolve relative image paths against dataset root.
        abs_paths: List[str] = []
        for raw in df["img_path"].astype(str):
            p = Path(raw)
            if not p.is_absolute():
                p = Path(self.dataset_dir) / p
            abs_paths.append(str(p.resolve()))
        df["img_path"] = abs_paths

        # Keep deterministic ordering for reproducibility.
        df = df.sort_values(["pid", "camid", "image_id"]).reset_index(drop=True)

        keep_cols = [
            "img_path",
            "pid",
            "camid",
            "team",
            "role",
            "video_id",
            "image_id",
            "visibility",
            "masks_path",
        ]
        return df[keep_cols]

    def _encode_multitask_columns(
        self, train_df: pd.DataFrame, query_df: pd.DataFrame, gallery_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        dfs = [train_df.copy(), query_df.copy(), gallery_df.copy()]

        # Ordered values keep stable integer encoding across runs.
        role_priority = ["ball", "goalkeeper", "other", "player", "referee"]
        team_priority = ["left", "right", "other"]

        for col, priority in (("role", role_priority), ("team", team_priority)):
            all_values = [str(v) for df in dfs for v in df[col].tolist() if pd.notna(v)]
            seen = set(all_values)
            ordered = [v for v in priority if v in seen]
            ordered += sorted(v for v in seen if v not in set(ordered))
            mapping = {name: idx for idx, name in enumerate(ordered)}
            mapping[None] = -1

            for i in range(len(dfs)):
                dfs[i][col] = [mapping.get(str(v), -1) if pd.notna(v) else -1 for v in dfs[i][col]]
                dfs[i][col] = dfs[i][col].astype(int)

            # PrtreidSampler expects reverse mapping int -> string.
            reverse = {idx: name for name, idx in mapping.items()}
            self.column_mapping[col] = reverse

        return dfs[0], dfs[1], dfs[2]


class ConfiguredCSVReIDDataset(CSVReIDDataset):
    """
    Pickle-safe runtime wrapper for Windows dataloader workers.
    The configuration is stored on class attributes (top-level class).
    """

    configured_dataset_name: str = "custom_reid"
    configured_require_team_role: bool = False
    configured_default_team: str = "left"
    configured_default_role: str = "player"

    def __init__(self, root: str = "", **kwargs: Any) -> None:
        super().__init__(
            root=root,
            dataset_name=self.configured_dataset_name,
            require_team_role=self.configured_require_team_role,
            default_team=self.configured_default_team,
            default_role=self.configured_default_role,
            **kwargs,
        )


def configure_runtime_dataset(
    dataset_name: str,
    require_team_role: bool,
    default_team: str = "left",
    default_role: str = "player",
) -> None:
    ConfiguredCSVReIDDataset.configured_dataset_name = dataset_name
    ConfiguredCSVReIDDataset.configured_require_team_role = require_team_role
    ConfiguredCSVReIDDataset.configured_default_team = default_team
    ConfiguredCSVReIDDataset.configured_default_role = default_role
