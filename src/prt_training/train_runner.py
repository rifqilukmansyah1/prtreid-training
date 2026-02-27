from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config_utils import ensure_runtime_paths, load_config_bundle
from .prtreid_patch import (
    apply_ascii_writer_patch,
    apply_generic_prtreid_sampler_patch,
    apply_generic_random_identity_sampler_patch,
    apply_reid_only_patch,
    apply_triplet_none_guard_patch,
    register_csv_dataset,
)


@dataclass
class TrainArgs:
    base_config: Path
    profile_config: Optional[Path]
    mode: str
    dataset_name: str
    dataset_nickname: str
    data_root: Path
    output_dir: Path
    weights: str
    job_id: int
    workers: int
    max_epoch: int
    train_batch_size: int
    test_batch_size: int
    test_only: bool


def run_train(args: TrainArgs) -> None:
    from yacs.config import CfgNode as CN
    from prtreid.scripts.default_config import engine_run_kwargs
    from prtreid.scripts.main import build_config, build_torchreid_model_engine

    cfg_dict = load_config_bundle(args.base_config, args.profile_config)
    cfg_dict = ensure_runtime_paths(cfg_dict, args.data_root, args.output_dir)

    # Keep sources/targets aligned with the registered custom dataset name.
    cfg_dict.setdefault("data", {})
    cfg_dict["data"]["sources"] = [args.dataset_name]
    cfg_dict["data"]["targets"] = [args.dataset_name]
    cfg_dict["data"]["workers"] = int(args.workers)

    cfg_dict.setdefault("project", {})
    cfg_dict["project"]["job_id"] = int(args.job_id)

    cfg_dict.setdefault("train", {})
    cfg_dict["train"]["max_epoch"] = int(args.max_epoch)
    cfg_dict["train"]["batch_size"] = int(args.train_batch_size)

    cfg_dict.setdefault("test", {})
    cfg_dict["test"]["batch_size"] = int(args.test_batch_size)

    if args.weights:
        cfg_dict.setdefault("model", {})
        cfg_dict["model"]["load_weights"] = args.weights

    mode = args.mode.lower()
    if mode not in {"multitask", "reid_only"}:
        raise ValueError(f"Unsupported mode: {args.mode}")

    apply_ascii_writer_patch()
    apply_triplet_none_guard_patch()

    require_team_role = mode == "multitask"
    register_csv_dataset(
        dataset_name=args.dataset_name,
        nickname=args.dataset_nickname,
        require_team_role=require_team_role,
    )

    if mode == "reid_only":
        # PID-only training: ignore team/role losses while keeping the same model family.
        apply_reid_only_patch()
        apply_generic_random_identity_sampler_patch()
        cfg_dict.setdefault("sampler", {})
        cfg_dict["sampler"]["train_sampler"] = "RandomIdentitySampler"
        cfg_dict["sampler"]["train_sampler_t"] = "RandomIdentitySampler"
        cfg_dict["sampler"]["num_instances"] = max(int(cfg_dict["sampler"].get("num_instances", 4)), 2)
    else:
        # Keep multitask objective but remove SoccerNet-specific batch composition assumption.
        apply_generic_prtreid_sampler_patch()
        cfg_dict.setdefault("sampler", {})
        if cfg_dict["sampler"].get("train_sampler") == "PrtreidSampler":
            cfg_dict["sampler"]["num_instances"] = max(int(cfg_dict["sampler"].get("num_instances", 4)), 2)

    cfg = CN(cfg_dict)
    cfg = build_config(config=cfg)

    engine, _ = build_torchreid_model_engine(cfg)
    run_kwargs = engine_run_kwargs(cfg)
    run_kwargs["test_only"] = bool(args.test_only)
    engine.run(**run_kwargs)
