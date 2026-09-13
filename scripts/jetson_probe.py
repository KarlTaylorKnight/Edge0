#!/usr/bin/env python3
"""Collect reproducible Jetson/CUDA facts before running Edge0 benchmarks.

The probe is read-only and uses the Python standard library unless PyTorch is
installed. It prints JSON so benchmark reports can retain the exact platform
and storage context that produced them.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_AUTO_TORCH = object()


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(errors="replace").rstrip("\x00\n")
    except (OSError, UnicodeError):
        return None


def parse_meminfo(text: str) -> dict[str, int | None]:
    """Return total and available memory in bytes from Linux meminfo text."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        fields = raw.split()
        if not fields:
            continue
        try:
            value = int(fields[0])
        except ValueError:
            continue
        unit = fields[1].lower() if len(fields) > 1 else "bytes"
        multiplier = 1024 if unit == "kb" else 1
        values[name] = value * multiplier
    return {
        "total_bytes": values.get("MemTotal"),
        "available_bytes": values.get("MemAvailable"),
    }


def _unescape_mount_field(value: str) -> str:
    for escaped, plain in (
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ):
        value = value.replace(escaped, plain)
    return value


def _mounts(mountinfo_text: str) -> list[dict[str, str]]:
    mounts = []
    for line in mountinfo_text.splitlines():
        fields = line.split()
        try:
            separator = fields.index("-")
            mounts.append(
                {
                    "device_id": fields[2],
                    "mount_point": _unescape_mount_field(fields[4]),
                    "filesystem": fields[separator + 1],
                    "device": _unescape_mount_field(fields[separator + 2]),
                }
            )
        except (ValueError, IndexError):
            continue
    return mounts


def _sysfs_tree_is_nvme(path: Path, seen: set[Path] | None = None) -> bool:
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return False
    seen = seen or set()
    if resolved in seen:
        return False
    seen.add(resolved)
    try:
        slaves = list((resolved / "slaves").iterdir())
    except OSError:
        slaves = []
    if slaves:
        return all(_sysfs_tree_is_nvme(slave, seen) for slave in slaves)
    return any(part.startswith("nvme") for part in resolved.parts)


def _device_is_nvme(
    device: str,
    device_id: str | None,
    sys_dev_block: Path,
) -> bool | None:
    if device_id:
        sysfs_device = sys_dev_block / device_id
        if sysfs_device.exists():
            return _sysfs_tree_is_nvme(sysfs_device)
    if device.startswith(("/dev/mmc", "/dev/sd", "/dev/vd")):
        return False
    return None


def storage_device_for_path(
    path: Path,
    mountinfo_text: str,
    *,
    sys_dev_block: Path = Path("/sys/dev/block"),
    disk_usage_fn: Any = shutil.disk_usage,
    path_device_id: str | None = None,
) -> dict[str, Any]:
    """Describe the longest-prefix Linux mount containing *path*."""
    target = path.resolve()
    if path_device_id is None:
        try:
            stat = target.stat()
            path_device_id = f"{os.major(stat.st_dev)}:{os.minor(stat.st_dev)}"
        except OSError:
            path_device_id = None
    candidates = []
    for position, mount in enumerate(_mounts(mountinfo_text)):
        mount_path = Path(mount["mount_point"])
        try:
            target.relative_to(mount_path)
        except ValueError:
            continue
        if path_device_id is not None and mount["device_id"] != path_device_id:
            continue
        candidates.append((len(mount_path.parts), position, mount))

    selected = (
        max(candidates, key=lambda item: (item[0], item[1]))[2]
        if candidates
        else {
            "device_id": None,
            "mount_point": None,
            "filesystem": None,
            "device": None,
        }
    )
    device = selected.get("device") or ""
    is_nvme = _device_is_nvme(
        device,
        selected.get("device_id"),
        sys_dev_block,
    )

    try:
        usage = disk_usage_fn(target)
        total_bytes, free_bytes = usage.total, usage.free
        disk_usage_available = True
    except OSError:
        total_bytes = free_bytes = None
        disk_usage_available = False

    return {
        **selected,
        "is_nvme": is_nvme,
        "path": str(target),
        "path_exists": target.exists(),
        "path_is_dir": target.is_dir(),
        "disk_usage_available": disk_usage_available,
        "total_bytes": total_bytes,
        "free_bytes": free_bytes,
    }


def collect_torch_info(torch_module: Any = _AUTO_TORCH) -> dict[str, Any]:
    """Collect CUDA facts without making torch a probe dependency."""
    info: dict[str, Any] = {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_name": None,
        "compute_capability": None,
        "device_memory_bytes": None,
        "errors": [],
    }
    if torch_module is _AUTO_TORCH:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except Exception as exc:  # noqa: BLE001 - vendor imports can fail broadly
            info["errors"].append(f"PyTorch import failed: {type(exc).__name__}: {exc}")
            return info
    if torch_module is None:
        info["errors"].append("PyTorch import was skipped or unavailable")
        return info

    info["installed"] = True
    info["version"] = str(getattr(torch_module, "__version__", "unknown"))
    cuda = getattr(torch_module, "cuda", None)
    info["cuda_version"] = getattr(getattr(torch_module, "version", None), "cuda", None)
    try:
        cuda_available = bool(cuda is not None and cuda.is_available())
    except Exception as exc:  # noqa: BLE001 - CUDA driver failures vary
        info["errors"].append(
            f"CUDA availability check failed: {type(exc).__name__}: {exc}"
        )
        return info
    info["cuda_available"] = cuda_available
    if cuda_available:
        assert cuda is not None
        try:
            props = cuda.get_device_properties(0)
            info.update(
                {
                    "device_name": props.name,
                    "compute_capability": [props.major, props.minor],
                    "device_memory_bytes": props.total_memory,
                }
            )
        except Exception as exc:  # noqa: BLE001 - CUDA driver failures vary
            info["errors"].append(
                f"CUDA device query failed: {type(exc).__name__}: {exc}"
            )
    return info


def collect_probe(
    model_dir: Path,
    *,
    fs_root: Path = Path("/"),
    mountinfo_text: str | None = None,
    torch_module: Any = _AUTO_TORCH,
    disk_usage_fn: Any = shutil.disk_usage,
    path_device_id: str | None = None,
) -> dict[str, Any]:
    """Collect a JSON-serializable Edge0/Jetson readiness report."""
    model = _read_text(fs_root / "proc/device-tree/model")
    l4t_release = _read_text(fs_root / "etc/nv_tegra_release")
    is_jetson_orin = bool(
        model and "jetson" in model.lower() and "orin" in model.lower()
    )

    meminfo_text = _read_text(fs_root / "proc/meminfo") or ""
    memory = parse_meminfo(meminfo_text)
    if mountinfo_text is None:
        mountinfo_text = _read_text(fs_root / "proc/self/mountinfo") or ""
    storage = storage_device_for_path(
        model_dir,
        mountinfo_text,
        sys_dev_block=fs_root / "sys/dev/block",
        disk_usage_fn=disk_usage_fn,
        path_device_id=path_device_id,
    )
    torch_info = collect_torch_info(torch_module)

    blockers = []
    if not storage["path_exists"]:
        blockers.append("model path does not exist")
    elif not storage["path_is_dir"]:
        blockers.append("model path is not a directory")
    if not storage["disk_usage_available"]:
        blockers.append("model path disk usage could not be determined")
    if not is_jetson_orin:
        blockers.append("hardware is not a Jetson Orin")
    if not torch_info["installed"]:
        blockers.append("PyTorch is not installed")
    if not torch_info["cuda_available"]:
        blockers.append("CUDA is not available through PyTorch")
    blockers.extend(
        f"PyTorch/CUDA probe error: {error}" for error in torch_info["errors"]
    )
    if storage["is_nvme"] is None:
        blockers.append("model path NVMe backing could not be confirmed")
    elif not storage["is_nvme"]:
        blockers.append("model path is not on NVMe storage")

    return {
        "schema_version": 1,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "jetson": {
            "is_jetson_orin": is_jetson_orin,
            "model": model,
            "l4t_release": l4t_release,
        },
        "memory": memory,
        "storage": storage,
        "torch": torch_info,
        "edge0_backend": os.environ.get("EDGE0_BACKEND", "mlx"),
        "ready": not blockers,
        "blockers": blockers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path.cwd(),
        help="checkpoint directory whose backing storage should be inspected",
    )
    parser.add_argument("--output", type=Path, help="also write JSON to this path")
    parser.add_argument(
        "--skip-torch",
        action="store_true",
        help="do not import PyTorch (reported as an explicit blocker)",
    )
    args = parser.parse_args(argv)

    torch_module = None if args.skip_torch else _AUTO_TORCH
    report = collect_probe(args.model_dir, torch_module=torch_module)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    sys.stdout.write(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
