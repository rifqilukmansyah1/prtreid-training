from __future__ import annotations

import argparse
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
THIRD_PARTY_PRTREID = ROOT / "third_party" / "prtreid"
if THIRD_PARTY_PRTREID.exists() and str(THIRD_PARTY_PRTREID) not in sys.path:
    # Allow running from local source checkout when editable install fails on Windows Cython build.
    sys.path.insert(0, str(THIRD_PARTY_PRTREID))

from prt_training.train_runner import TrainArgs, run_train


def _sanitize_cuda_alloc_conf() -> None:
    """
    Torch 1.x does not support 'expandable_segments' allocator option.
    Remove it early to avoid RuntimeError during CUDA init/checkpoint load.
    """
    conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    if "expandable_segments" not in conf:
        return
    try:
        torch_major = int(version("torch").split(".", 1)[0])
    except (PackageNotFoundError, ValueError):
        torch_major = None
    if torch_major is None or torch_major < 2:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        print("[env] removed unsupported PYTORCH_CUDA_ALLOC_CONF=expandable_segments for torch 1.x")


def _prepare_runtime_env() -> None:
    _sanitize_cuda_alloc_conf()
    os.environ.setdefault("WANDB_MODE", "disabled")
    os.environ.setdefault("WANDB_DISABLED", "true")
    # Make stdout/stderr safe for unicode logs on Windows terminal.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Standalone PRTReid trainer")
    parser.add_argument("--base_config", type=str, default="configs/prtreid/base.yaml")
    parser.add_argument("--profile_config", type=str, default="configs/prtreid/profiles/multitask_soccernet_like.yaml")
    parser.add_argument("--mode", type=str, choices=["multitask", "reid_only"], default="multitask")
    parser.add_argument("--dataset_name", type=str, default="custom_reid")
    parser.add_argument("--dataset_nickname", type=str, default="crid")
    parser.add_argument("--data_root", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="outputs/runs")
    parser.add_argument("--weights", type=str, default="model/prtreid-soccernet-baseline.pth.tar")
    parser.add_argument("--job_id", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_epoch", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--test_batch_size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    _prepare_runtime_env()
    args = parse_args()
    run_train(
        TrainArgs(
            base_config=Path(args.base_config),
            profile_config=Path(args.profile_config) if args.profile_config else None,
            mode=args.mode,
            dataset_name=args.dataset_name,
            dataset_nickname=args.dataset_nickname,
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
            weights=args.weights,
            job_id=args.job_id,
            workers=args.workers,
            max_epoch=args.max_epoch,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            test_only=False,
        )
    )


if __name__ == "__main__":
    main()
