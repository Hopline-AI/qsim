"""Loader for `constants.toml`, the single source of every tunable value.

`simulator.py` and `contract.py` define their module constants from `C`, and
`DeviceParams` takes its defaults from those. No physical value is written
anywhere else in the package, so the TOML is the only place to look or edit.

Set `TRANSMON_SIM_CONSTANTS` to a path to load a different file, which is how a
caller models a device with a different set of published constants without
editing the package. Per-device overrides should still go through
`DeviceParams`; this is for swapping the whole baseline at once.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path(__file__).with_name("constants.toml")
HOUR_S = 3600.0


def _scale(value: Any, factor: float) -> Any:
    return [x * factor for x in value] if isinstance(value, list) else value * factor


def load(path: str | os.PathLike[str] | None = None) -> dict[str, Any]:
    """Read a constants file. Keys ending in `_h` are also exposed in seconds.

    A duration `x_h` becomes `x_s` in seconds; a rate `x_per_h` becomes `x_per_s` per second.
    """
    p = Path(path or os.environ.get("TRANSMON_SIM_CONSTANTS") or DEFAULT_PATH)
    with p.open("rb") as fh:
        data = tomllib.load(fh)
    for section in data.values():
        if isinstance(section, dict):
            for key in [k for k in section if k.endswith("_h")]:
                factor = 1.0 / HOUR_S if key.endswith("_per_h") else HOUR_S
                section[f"{key[:-2]}_s"] = _scale(section[key], factor)
    return data


C = load()
