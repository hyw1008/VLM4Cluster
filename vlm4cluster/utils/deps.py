from __future__ import annotations

import importlib


def require_module(module_name: str, install_hint: str) -> object:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing optional dependency '{module_name}'. Install it with: {install_hint}"
        ) from exc
