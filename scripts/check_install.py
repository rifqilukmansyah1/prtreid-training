from __future__ import annotations

import importlib
import platform
import sys


def _safe_version(module_name: str) -> str:
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return f"missing ({exc})"


def main() -> None:
    print("[check] Python:", sys.version.replace("\n", " "))
    print("[check] Platform:", platform.platform())
    print("[check] torch:", _safe_version("torch"))
    print("[check] torchvision:", _safe_version("torchvision"))
    print("[check] numpy:", _safe_version("numpy"))
    print("[check] prtreid:", _safe_version("prtreid"))


if __name__ == "__main__":
    main()
