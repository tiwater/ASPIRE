"""Deployment tests use temporary directories only; never connect to a robot."""

import fcntl
import importlib.util
import io
import json
import tarfile

import pytest

spec = importlib.util.spec_from_file_location("realman_release", "scripts/realman/release.py")
release = importlib.util.module_from_spec(spec)
spec.loader.exec_module(release)


@pytest.fixture(scope="module")
def bundle():
    return release.build_archive()


def test_bundle_reproducible_and_contains_calibration_evidence(bundle):
    data, manifest = bundle
    assert release.build_archive()[0] == data
    files, checked = release.verify_archive(data)
    assert checked == manifest
    assert "configs/realman/calibration/right_wrist_20261028.json" in files
    assert "assets/realman/calibration/manifest.json" in files
    assert all("ref/Tmp" not in p and ".codex" not in p for p in files)
    evidence = json.loads(files["assets/realman/calibration/manifest.json"])
    for item in evidence["profiles"]:
        assert release.digest(files[item["path"]]) == item["sha256"]
    for item in evidence["evidence"]:
        assert release.digest(files[item["stored"]]) == item["sha256"]


def test_install_policy_preservation_and_locked_service(bundle, tmp_path):
    data, manifest = bundle
    root = tmp_path / "robot with spaces"
    result = release.install_archive(data, root)
    assert not result["service_started"] and not result["motion_commanded"]
    assert (root / "current").resolve().name == manifest["release"]
    policy = root / "config/control_policy.json"
    assert json.loads(policy.read_text())["motion_enabled"] is False
    custom = b'{"motion_enabled":false,"allow_stop":true,"operator_note":"keep"}\n'
    policy.write_bytes(custom)
    assert release.install_archive(data, root)["policy_preserved"]
    assert policy.read_bytes() == custom
    with (root / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            release.install_archive(data, root)
        with pytest.raises(BlockingIOError):
            release.activate_release(root, manifest["release"])


def test_bad_archive_and_tampered_release_never_activate(bundle, tmp_path):
    data, manifest = bundle
    bad = io.BytesIO()
    with tarfile.open(fileobj=bad, mode="w:gz") as tar:
        info = tarfile.TarInfo("../escape")
        info.size = 1
        tar.addfile(info, io.BytesIO(b"x"))
    with pytest.raises(ValueError, match="member"):
        release.install_archive(bad.getvalue(), tmp_path / "bad")
    assert not (tmp_path / "bad").exists()
    root = tmp_path / "robot"
    release.install_archive(data, root)
    (root / "current/src/cap_harness/realman/control.py").write_text("tampered")
    with pytest.raises(ValueError, match="checksum"):
        release.activate_release(root, manifest["release"])


def test_actual_upgrade_and_rollback_keep_policy(bundle, tmp_path):
    data, first = bundle
    root = tmp_path / "robot"
    release.install_archive(data, root)
    files, _ = release.verify_archive(data)
    files["src/cap_harness/realman/__init__.py"] += b"\n# upgrade test\n"
    second = dict(first)
    second["sha256"] = {n: release.digest(v) for n, v in files.items()}
    second["release"] = release.release_id(second["sha256"])
    files["manifest.json"] = json.dumps(second).encode()
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w:gz") as tar:
        for name, content in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    old_policy = (root / "config/control_policy.json").read_bytes()
    release.install_archive(archive.getvalue(), root)
    assert (root / "current").resolve().name == second["release"]
    release.activate_release(root, first["release"])
    assert (root / "current").resolve().name == first["release"]
    assert (root / "config/control_policy.json").read_bytes() == old_policy
