"""Offline, checksummed control/calibration bundles and locked robot-local installation.

Installing or activating a release never starts services or sends robot commands.
Only the Python standard library is required; ROS is needed at service runtime.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[2]
MODULES = [
    "__init__",
    "control",
    "ros_transport",
    "handeye_capture",
    "rgbd_capture",
    "live_server",
    "mcp_proxy",
    "calibration",
    "handeye_solver",
    "capture",
    "telemetry",
    "camera_geometry",
    "rgbd_geometry",
    "shared_board_calibration",
    "moving_base_tcp",
    "tcp_calibration",
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def release_id(hashes):
    return digest(json.dumps(hashes, sort_keys=True).encode())[:16]


def build_archive(repo=ROOT):
    repo = Path(repo)
    paths = ["src/cap_harness/__init__.py"]
    paths += [f"src/cap_harness/realman/{name}.py" for name in MODULES]
    paths += [
        "configs/realman/robot.json",
        "configs/realman/control_policy.json",
        "configs/realman/controller_urdf_candidate.json",
        "configs/realman/calibration_board.json",
    ]
    for pattern in [
        "configs/realman/calibration/**/*.json",
        "assets/realman/calibration/**/*",
        "scripts/realman/*.sh",
        "scripts/realman/*requirements.txt",
        "scripts/realman/solve_*.py",
        "scripts/realman/inspect_*.py",
        "docs/realman/*.md",
    ]:
        paths += [str(p.relative_to(repo)) for p in sorted(repo.glob(pattern)) if p.is_file()]
    paths += ["scripts/realman/release.py", "scripts/realman/aspire-realman-mcp.service"]
    files = {name: (repo / name).read_bytes() for name in sorted(set(paths))}
    manifest = {
        "schema_version": 1,
        "kind": "realman_control_and_calibration",
        "sha256": {name: digest(data) for name, data in files.items()},
        "starts_service": False,
        "activates_motion": False,
        "scope": "Control/capture runtime and offline calibration utilities; no trained coffee policy or simulation assets.",
    }
    manifest["release"] = release_id(manifest["sha256"])
    files["manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    buffer = io.BytesIO()
    with (
        gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as tar,
    ):
        for name, data in sorted(files.items()):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if name.endswith(".sh") else 0o644
            tar.addfile(info, io.BytesIO(data))
    data = buffer.getvalue()
    verify_archive(data)
    return data, manifest


def verify_archive(data):
    files = {}
    total = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for item in tar:
            path = PurePosixPath(item.name)
            if (
                not item.isfile()
                or path.is_absolute()
                or ".." in path.parts
                or str(path) != item.name
                or item.name in files
                or item.size > 20_000_000
            ):
                raise ValueError("invalid archive member")
            total += item.size
            if total > 100_000_000:
                raise ValueError("release exceeds size budget")
            files[item.name] = tar.extractfile(item).read()
    manifest = json.loads(files.pop("manifest.json"))
    hashes = {name: digest(value) for name, value in files.items()}
    if hashes != manifest["sha256"] or release_id(hashes) != manifest["release"]:
        raise ValueError("release checksum mismatch")
    policy = json.loads(files["configs/realman/control_policy.json"])
    if policy.get("motion_enabled") is not False or policy.get("allow_stop") is not False:
        raise ValueError("packaged default policy must be observation-only")
    return files, manifest


def verify_installed(folder):
    folder = Path(folder)
    manifest = json.loads((folder / "manifest.json").read_text())
    if folder.name != manifest["release"] or release_id(manifest["sha256"]) != folder.name:
        raise ValueError("installed release identity mismatch")
    for name, expected in manifest["sha256"].items():
        relative = PurePosixPath(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not (folder / name).resolve().is_relative_to(folder.resolve())
        ):
            raise ValueError("invalid installed path")
        if digest((folder / name).read_bytes()) != expected:
            raise ValueError("installed checksum mismatch: " + name)
    return manifest


def _activate(root, release):
    verify_installed(root / "releases" / release)
    current = root / "current"
    previous = os.readlink(current) if current.is_symlink() else None
    if current.exists() and not current.is_symlink():
        raise ValueError("current must be a symlink")
    temporary = root / ".current-next"
    temporary.unlink(missing_ok=True)
    temporary.symlink_to("releases/" + release)
    os.replace(temporary, current)
    return previous


def install_archive(data, root):
    files, manifest = verify_archive(data)
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name in ["releases", "config"]:
            if (root / name).is_symlink():
                raise ValueError("installation directory must not be a symlink")
            (root / name).mkdir(exist_ok=True)
        destination = root / "releases" / manifest["release"]
        if not destination.exists():
            temporary = Path(tempfile.mkdtemp(prefix=".incoming-", dir=root / "releases"))
            try:
                for name, content in files.items():
                    target = temporary / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    if name.endswith(".sh"):
                        target.chmod(0o755)
                (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                os.rename(temporary, destination)
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        verify_installed(destination)
        policy = root / "config/control_policy.json"
        existed = policy.exists()
        if not existed:
            with policy.open("xb") as output:
                output.write(files["configs/realman/control_policy.json"])
        policy_hash = digest(policy.read_bytes())
        # Stable service entrypoints follow current, also on rollback.
        for name, script in [
            ("start_mcp.sh", "robot_start_mcp.sh"),
            ("start_backend.sh", "robot_start_backend.sh"),
            ("connect_mcp.sh", "robot_connect_mcp.sh"),
        ]:
            launcher = root / name
            launcher.write_text(
                "#!/usr/bin/env bash\nset -euo pipefail\nexport ASPIRE_REALMAN_ROOT="
                + shlex.quote(str(root))
                + '\nexec bash "$ASPIRE_REALMAN_ROOT/current/scripts/realman/'
                + script
                + '" "$@"\n'
            )
            launcher.chmod(0o755)
        previous = _activate(root, manifest["release"])
        return {
            "release": manifest["release"],
            "previous": previous,
            "policy_preserved": existed,
            "policy_sha256": policy_hash,
            "service_started": False,
            "motion_commanded": False,
        }


def activate_release(root, release):
    if not re.fullmatch("[0-9a-f]{16}", release):
        raise ValueError("invalid release identifier")
    root = Path(root).resolve()
    with (root / "control.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return {"release": release, "previous": _activate(root, release), "service_started": False}


def main():
    parser = argparse.ArgumentParser(__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--output", required=True, type=Path)
    verify = commands.add_parser("verify")
    verify.add_argument("archive", type=Path)
    install = commands.add_parser("install")
    install.add_argument("archive", type=Path)
    install.add_argument("--root", required=True, type=Path)
    activate = commands.add_parser("activate")
    activate.add_argument("release")
    activate.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "build":
        data, result = build_archive()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as output:
            output.write(data)
        result = {
            "archive": str(args.output),
            "release": result["release"],
            "archive_sha256": digest(data),
        }
    elif args.command == "verify":
        _, result = verify_archive(args.archive.read_bytes())
        result = {"release": result["release"], "files": len(result["sha256"]), "verified": True}
    elif args.command == "install":
        result = install_archive(args.archive.read_bytes(), args.root)
    else:
        result = activate_release(args.root, args.release)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
