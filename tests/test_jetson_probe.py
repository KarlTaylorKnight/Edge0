from __future__ import annotations

import json
from pathlib import Path

from scripts import jetson_probe
from scripts.jetson_probe import (
    collect_probe,
    collect_torch_info,
    parse_meminfo,
    storage_device_for_path,
)


def _make_nvme_sysfs(root: Path, device_id: str = "259:1") -> Path:
    sys_dev_block = root / "sys/dev/block"
    backing = root / "sys/devices/platform/nvme/nvme0/nvme0n1"
    backing.mkdir(parents=True, exist_ok=True)
    sys_dev_block.mkdir(parents=True, exist_ok=True)
    (sys_dev_block / device_id).symlink_to(backing)
    return sys_dev_block


def test_parse_meminfo_converts_kib_to_bytes():
    parsed = parse_meminfo("MemTotal: 8192000 kB\nMemAvailable: 4096000 kB\n")

    assert parsed == {
        "total_bytes": 8_388_608_000,
        "available_bytes": 4_194_304_000,
    }


def test_parse_meminfo_allows_missing_available_memory():
    parsed = parse_meminfo("MemTotal: 1024 kB\n")

    assert parsed == {"total_bytes": 1_048_576, "available_bytes": None}


def test_storage_device_uses_longest_mount_prefix(tmp_path: Path):
    model_dir = tmp_path / "models" / "edge0-8b"
    model_dir.mkdir(parents=True)
    mountinfo = "\n".join(
        [
            "30 20 8:1 / / rw,relatime - ext4 /dev/mmcblk0p1 rw",
            f"31 30 259:1 / {tmp_path / 'models'} rw,relatime - ext4 /dev/nvme0n1p1 rw",
        ]
    )

    storage = storage_device_for_path(
        model_dir,
        mountinfo,
        path_device_id="259:1",
        sys_dev_block=_make_nvme_sysfs(tmp_path),
    )

    assert storage["device"] == "/dev/nvme0n1p1"
    assert storage["mount_point"] == str(tmp_path / "models")
    assert storage["filesystem"] == "ext4"
    assert storage["is_nvme"] is True


def test_storage_device_marks_non_nvme_media(tmp_path: Path):
    mountinfo = "30 20 179:1 / / rw,relatime - ext4 /dev/mmcblk0p1 rw"

    storage = storage_device_for_path(
        tmp_path,
        mountinfo,
        path_device_id="179:1",
    )

    assert storage["is_nvme"] is False


def test_storage_device_resolves_device_mapper_backing_nvme(tmp_path: Path):
    sys_dev_block = tmp_path / "sys/dev/block"
    mapped = tmp_path / "sys/devices/virtual/block/dm-0"
    physical = tmp_path / "sys/devices/platform/nvme/nvme0/nvme0n1"
    (mapped / "slaves").mkdir(parents=True)
    physical.mkdir(parents=True)
    sys_dev_block.mkdir(parents=True)
    (sys_dev_block / "253:0").symlink_to(mapped)
    (mapped / "slaves/nvme0n1").symlink_to(physical)
    mountinfo = "30 20 253:0 / / rw,relatime - ext4 /dev/mapper/cryptroot rw"

    storage = storage_device_for_path(
        tmp_path,
        mountinfo,
        path_device_id="253:0",
        sys_dev_block=sys_dev_block,
    )

    assert storage["device_id"] == "253:0"
    assert storage["is_nvme"] is True


def test_storage_device_rejects_mixed_device_mapper_backing(tmp_path: Path):
    sys_dev_block = tmp_path / "sys/dev/block"
    mapped = tmp_path / "sys/devices/virtual/block/dm-0"
    nvme = tmp_path / "sys/devices/platform/nvme/nvme0/nvme0n1"
    mmc = tmp_path / "sys/devices/platform/mmc/mmcblk0"
    (mapped / "slaves").mkdir(parents=True)
    nvme.mkdir(parents=True)
    mmc.mkdir(parents=True)
    sys_dev_block.mkdir(parents=True)
    (sys_dev_block / "253:0").symlink_to(mapped)
    (mapped / "slaves/nvme0n1").symlink_to(nvme)
    (mapped / "slaves/mmcblk0").symlink_to(mmc)

    storage = storage_device_for_path(
        tmp_path,
        "30 20 253:0 / / rw - ext4 /dev/mapper/mixed rw",
        path_device_id="253:0",
        sys_dev_block=sys_dev_block,
    )

    assert storage["is_nvme"] is False


def test_storage_device_handles_escaped_mounts_and_malformed_rows(tmp_path: Path):
    model_dir = tmp_path / "model files"
    model_dir.mkdir()
    escaped = str(model_dir).replace(" ", "\\040")
    mountinfo = "\n".join(
        [
            "malformed",
            f"31 30 259:1 / {escaped} rw - ext4 /dev/nvme0n1p1 rw",
        ]
    )

    storage = storage_device_for_path(
        model_dir,
        mountinfo,
        path_device_id="259:1",
        sys_dev_block=_make_nvme_sysfs(tmp_path),
    )

    assert storage["mount_point"] == str(model_dir)
    assert storage["is_nvme"] is True


def test_storage_device_preserves_unknown_nvme_state(tmp_path: Path):
    mountinfo = "30 20 0:42 / / rw - overlay overlay rw"

    storage = storage_device_for_path(
        tmp_path,
        mountinfo,
        path_device_id="0:42",
        sys_dev_block=tmp_path / "missing-sysfs",
    )

    assert storage["is_nvme"] is None


def test_storage_device_does_not_trust_network_or_pseudo_source_names(tmp_path: Path):
    for mountinfo, device_id in (
        ("30 20 0:42 / / rw - nfs server:/nvme0n1 rw", "0:42"),
        ("31 20 0:43 / / rw - tmpfs /dev/nvme0n1 rw", "0:43"),
    ):
        storage = storage_device_for_path(
            tmp_path,
            mountinfo,
            path_device_id=device_id,
            sys_dev_block=tmp_path / "missing-sysfs",
        )

        assert storage["is_nvme"] is not True


def test_storage_device_matches_active_device_id_for_stacked_mounts(tmp_path: Path):
    mountinfo = (
        "30 20 259:1 / / rw - ext4 /dev/nvme0n1p1 rw\n"
        "31 20 179:1 / / rw - ext4 /dev/mmcblk0p1 rw"
    )

    storage = storage_device_for_path(
        tmp_path,
        mountinfo,
        path_device_id="179:1",
    )

    assert storage["device_id"] == "179:1"
    assert storage["is_nvme"] is False


def test_storage_device_reports_disk_usage_failure(tmp_path: Path):
    def fail_disk_usage(_path):
        raise OSError("unavailable")

    storage = storage_device_for_path(
        tmp_path,
        "30 20 259:1 / / rw - ext4 /dev/nvme0n1p1 rw",
        disk_usage_fn=fail_disk_usage,
    )

    assert storage["disk_usage_available"] is False
    assert storage["total_bytes"] is None
    assert storage["free_bytes"] is None


class _FakeProps:
    name = "Orin"
    total_memory = 8_000_000_000
    major = 8
    minor = 7


class _FakeCuda:
    @staticmethod
    def is_available():
        return True

    @staticmethod
    def get_device_properties(_index):
        return _FakeProps()


class _FakeVersion:
    cuda = "12.6"


class _FakeTorch:
    __version__ = "2.8.0"
    cuda = _FakeCuda()
    version = _FakeVersion()


def test_collect_torch_info_reports_cuda_device():
    info = collect_torch_info(_FakeTorch())

    assert info == {
        "installed": True,
        "version": "2.8.0",
        "cuda_available": True,
        "cuda_version": "12.6",
        "device_name": "Orin",
        "compute_capability": [8, 7],
        "device_memory_bytes": 8_000_000_000,
        "errors": [],
    }


def test_collect_torch_info_reports_missing_torch():
    info = collect_torch_info(None)

    assert info == {
        "installed": False,
        "version": None,
        "cuda_available": False,
        "cuda_version": None,
        "device_name": None,
        "compute_capability": None,
        "device_memory_bytes": None,
        "errors": ["PyTorch import was skipped or unavailable"],
    }


class _BrokenCuda:
    @staticmethod
    def is_available():
        raise RuntimeError("CUDA driver initialization failed")


class _BrokenTorch:
    __version__ = "2.8.0"
    cuda = _BrokenCuda()
    version = _FakeVersion()


def test_collect_torch_info_turns_cuda_driver_failure_into_probe_data():
    info = collect_torch_info(_BrokenTorch())

    assert info["installed"] is True
    assert info["cuda_available"] is False
    assert info["errors"] == [
        (
            "CUDA availability check failed: "
            "RuntimeError: CUDA driver initialization failed"
        )
    ]


class _BrokenPropertiesCuda(_FakeCuda):
    @staticmethod
    def get_device_properties(_index):
        raise RuntimeError("device query failed")


class _BrokenPropertiesTorch:
    __version__ = "2.8.0"
    cuda = _BrokenPropertiesCuda()
    version = _FakeVersion()


def test_collect_torch_info_turns_device_query_failure_into_probe_data():
    info = collect_torch_info(_BrokenPropertiesTorch())

    assert info["cuda_available"] is True
    assert info["errors"] == [
        "CUDA device query failed: RuntimeError: device query failed"
    ]


def test_collect_probe_reports_actionable_blockers(tmp_path: Path):
    root = tmp_path / "root"
    (root / "proc/device-tree").mkdir(parents=True)
    (root / "etc").mkdir()
    (root / "proc/device-tree/model").write_bytes(
        b"NVIDIA Jetson Orin Nano Developer Kit\x00"
    )
    (root / "etc/nv_tegra_release").write_text("# R36 (release), REVISION: 4.3\n")
    (root / "proc/meminfo").write_text(
        "MemTotal: 8192000 kB\nMemAvailable: 4096000 kB\n"
    )
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    mountinfo = "30 20 179:1 / / rw,relatime - ext4 /dev/mmcblk0p1 rw"

    report = collect_probe(
        model_dir=model_dir,
        fs_root=root,
        mountinfo_text=mountinfo,
        torch_module=None,
        path_device_id="179:1",
    )

    assert report["schema_version"] == 1
    assert report["jetson"]["is_jetson_orin"] is True
    assert report["jetson"]["l4t_release"].startswith("# R36")
    assert report["storage"]["is_nvme"] is False
    assert report["ready"] is False
    assert "PyTorch is not installed" in report["blockers"]
    assert "CUDA is not available through PyTorch" in report["blockers"]
    assert "model path is not on NVMe storage" in report["blockers"]


def test_collect_probe_rejects_a_missing_model_path(tmp_path: Path):
    root = tmp_path / "root"
    (root / "proc/device-tree").mkdir(parents=True)
    (root / "etc").mkdir()
    (root / "proc/device-tree/model").write_text("NVIDIA Jetson Orin Nano")
    (root / "etc/nv_tegra_release").write_text("# R36")
    (root / "proc/meminfo").write_text("MemTotal: 8192000 kB\n")
    _make_nvme_sysfs(root)

    report = collect_probe(
        model_dir=tmp_path / "missing-model",
        fs_root=root,
        mountinfo_text="30 20 259:1 / / rw - ext4 /dev/nvme0n1p1 rw",
        torch_module=_FakeTorch(),
        path_device_id="259:1",
    )

    assert report["storage"]["is_nvme"] is True
    assert report["ready"] is False
    assert "model path does not exist" in report["blockers"]


def test_collect_probe_rejects_a_model_path_that_is_not_a_directory(tmp_path: Path):
    model_file = tmp_path / "model.safetensors"
    model_file.write_bytes(b"")
    root = tmp_path / "root"
    (root / "proc/device-tree").mkdir(parents=True)
    (root / "etc").mkdir()
    (root / "proc/device-tree/model").write_text("NVIDIA Jetson Orin Nano")
    (root / "proc/meminfo").write_text("MemTotal: 8192000 kB\n")
    _make_nvme_sysfs(root)

    report = collect_probe(
        model_dir=model_file,
        fs_root=root,
        mountinfo_text="30 20 259:1 / / rw - ext4 /dev/nvme0n1p1 rw",
        torch_module=_FakeTorch(),
        path_device_id="259:1",
    )

    assert report["ready"] is False
    assert "model path is not a directory" in report["blockers"]


def test_collect_probe_blocks_incomplete_cuda_device_evidence(tmp_path: Path):
    root = tmp_path / "root"
    (root / "proc/device-tree").mkdir(parents=True)
    (root / "etc").mkdir()
    (root / "proc/device-tree/model").write_text("NVIDIA Jetson Orin Nano")
    (root / "proc/meminfo").write_text("MemTotal: 8192000 kB\n")
    _make_nvme_sysfs(root)

    report = collect_probe(
        model_dir=tmp_path,
        fs_root=root,
        mountinfo_text="30 20 259:1 / / rw - ext4 /dev/nvme0n1p1 rw",
        torch_module=_BrokenPropertiesTorch(),
        path_device_id="259:1",
    )

    assert report["ready"] is False
    assert any("CUDA device query failed" in item for item in report["blockers"])


def test_collect_probe_can_report_fully_ready(tmp_path: Path):
    root = tmp_path / "root"
    (root / "proc/device-tree").mkdir(parents=True)
    (root / "etc").mkdir()
    (root / "proc/device-tree/model").write_text("NVIDIA Jetson Orin Nano")
    (root / "etc/nv_tegra_release").write_text("# R36")
    (root / "proc/meminfo").write_text("MemTotal: 8192000 kB\n")
    _make_nvme_sysfs(root)

    report = collect_probe(
        model_dir=tmp_path,
        fs_root=root,
        mountinfo_text="30 20 259:1 / / rw - ext4 /dev/nvme0n1p1 rw",
        torch_module=_FakeTorch(),
        path_device_id="259:1",
    )

    assert report["ready"] is True
    assert report["blockers"] == []


def test_collect_probe_blocks_when_disk_usage_is_unavailable(tmp_path: Path):
    root = tmp_path / "root"
    (root / "proc/device-tree").mkdir(parents=True)
    (root / "etc").mkdir()
    (root / "proc/device-tree/model").write_text("NVIDIA Jetson Orin Nano")
    (root / "proc/meminfo").write_text("MemTotal: 8192000 kB\n")
    _make_nvme_sysfs(root)

    def fail_disk_usage(_path):
        raise OSError("unavailable")

    report = collect_probe(
        model_dir=tmp_path,
        fs_root=root,
        mountinfo_text="30 20 259:1 / / rw - ext4 /dev/nvme0n1p1 rw",
        torch_module=_FakeTorch(),
        disk_usage_fn=fail_disk_usage,
        path_device_id="259:1",
    )

    assert report["ready"] is False
    assert "model path disk usage could not be determined" in report["blockers"]


def test_probe_report_is_json_serializable(tmp_path: Path):
    root = tmp_path / "root"
    (root / "proc/device-tree").mkdir(parents=True)
    (root / "etc").mkdir()
    (root / "proc/device-tree/model").write_text("generic arm board")
    (root / "proc/meminfo").write_text("MemTotal: 1024 kB\n")

    report = collect_probe(
        model_dir=tmp_path,
        fs_root=root,
        mountinfo_text="30 20 8:1 / / rw - ext4 /dev/sda1 rw",
        torch_module=None,
    )

    json.dumps(report)
    assert "hardware is not a Jetson Orin" in report["blockers"]


def test_cli_writes_json_and_returns_nonzero_for_blockers(
    tmp_path: Path,
    monkeypatch,
    capsys,
):
    expected = {"schema_version": 1, "ready": False, "blockers": ["test"]}
    monkeypatch.setattr(jetson_probe, "collect_probe", lambda *_args, **_kw: expected)
    output = tmp_path / "probe.json"

    result = jetson_probe.main(
        [
            "--model-dir",
            str(tmp_path),
            "--output",
            str(output),
            "--skip-torch",
        ]
    )

    assert result == 1
    assert json.loads(output.read_text()) == expected
    assert json.loads(capsys.readouterr().out) == expected


def test_cli_returns_zero_for_a_ready_report(tmp_path: Path, monkeypatch, capsys):
    expected = {"schema_version": 1, "ready": True, "blockers": []}
    monkeypatch.setattr(jetson_probe, "collect_probe", lambda *_args, **_kw: expected)

    result = jetson_probe.main(["--model-dir", str(tmp_path)])

    assert result == 0
    assert json.loads(capsys.readouterr().out) == expected
