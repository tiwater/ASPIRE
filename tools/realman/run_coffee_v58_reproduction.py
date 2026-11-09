#!/usr/bin/env python3
"""Materialize and run the recorded v58 coffee-service scenario.

The original v58 run did not save a complete runtime module snapshot. This
launcher recreates the run from the preserved runner, inputs, and arguments,
and captures the exact RealMan Python sources used by each new run so that the
new run can itself be replayed with replay_coffee_trial.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from types import SimpleNamespace


REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "assets/realman/coffee/v58_reproduction"
ARTIFACTS = REPO / "artifacts/realman/coffee_service_20261102"
CASE = "v58_reproduction"
RUNNER_NAME = "run_service_lifecycle_staff_collection_clearance_v31_cycle_speed2_v58_reproduction.py"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy_verified(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha(source) != sha(target):
            raise FileExistsError(f"refusing to replace different input: {target}")
        return
    shutil.copy2(source, target)


def validate_package() -> None:
    manifest_path = PACKAGE / "input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest.items():
        path = PACKAGE / relative
        if not path.is_file() or path.stat().st_size != expected["bytes"] or sha(path) != expected["sha256"]:
            raise ValueError(f"reproduction input is missing or changed: {relative}")


def materialize_base_housing_parts() -> None:
    manifest = json.loads((PACKAGE / "scene/base_housing_parts_manifest.json").read_text())
    expected = {entry["name"]: entry for entry in manifest["files"]}
    expected[manifest["metadata"]["name"]] = manifest["metadata"]
    if len(manifest["files"]) != 1320:
        raise ValueError("v58 base-housing mesh manifest must contain all 1320 OBJ files")
    target = ARTIFACTS / "base_housing_parts"
    missing = set(expected)
    if target.is_dir():
        for name, entry in expected.items():
            path = target / name
            if not path.exists():
                continue
            if not path.is_file() or path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
                raise ValueError(f"existing base-housing part is missing or differs: {path}")
            missing.remove(name)
        if not missing:
            return
    if target.exists() and not target.is_dir():
        raise FileExistsError(f"base-housing destination is not a directory: {target}")

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    archive = PACKAGE / "scene/base_housing_parts.tar.gz"
    with tempfile.TemporaryDirectory(prefix="v58_base_housing_", dir=ARTIFACTS) as temp_name:
        temporary = Path(temp_name) / "base_housing_parts"
        temporary.mkdir()
        with tarfile.open(archive, "r:gz") as bundle:
            members = bundle.getmembers()
            if len(members) != len(expected):
                raise ValueError("base-housing archive member count differs from its manifest")
            seen: set[str] = set()
            for member in members:
                parts = Path(member.name).parts
                if len(parts) != 2 or parts[0] != "base_housing_parts" or parts[1] not in expected or not member.isfile():
                    raise ValueError(f"unexpected base-housing archive member: {member.name}")
                name = parts[1]
                if name in seen:
                    raise ValueError(f"duplicate base-housing archive member: {name}")
                seen.add(name)
                source = bundle.extractfile(member)
                if source is None:
                    raise ValueError(f"cannot read base-housing archive member: {name}")
                output = temporary / name
                output.write_bytes(source.read())
                entry = expected[name]
                if output.stat().st_size != entry["bytes"] or sha(output) != entry["sha256"]:
                    raise ValueError(f"base-housing part checksum mismatch: {name}")
        if seen != set(expected):
            raise ValueError("base-housing archive is incomplete")
        if target.is_dir():
            for name in missing:
                shutil.move(str(temporary / name), str(target / name))
        else:
            temporary.replace(target)


def runtime_snapshot(snapshot_root: Path, trial: Path, runner: Path) -> None:
    destination = trial / "implementation_snapshots"
    destination.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, str] = {}
    package_copy = destination / "cap_harness"
    shutil.copytree(snapshot_root / "cap_harness", package_copy)
    for path in sorted(package_copy.rglob("*")):
        if path.is_file():
            manifest["src/" + path.relative_to(package_copy.parent).as_posix()] = sha(path)
    runner_copy = destination / "experiment.py"
    shutil.copy2(runner, runner_copy)
    manifest["experiment.py"] = sha(runner_copy)
    (destination / "captured_during_run.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="new output name; existing runs are never overwritten")
    parser.add_argument("--prepare-only", action="store_true", help="materialize inputs and print the command without running")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_]+", args.name):
        parser.error("--name must contain only letters, numbers, and underscores")

    validate_package()
    materialize_base_housing_parts()
    try:
        import mujoco
        import numpy
        import scipy
    except ImportError as exc:
        raise SystemExit(f"simulation dependencies are missing; install with `pip install -e '.[realman]'`: {exc}")
    if mujoco.__version__ != "3.3.0":
        raise SystemExit(f"v58 requires MuJoCo 3.3.0; found {mujoco.__version__}")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit(f"v58 is pinned to Python 3.12; found {sys.version.split()[0]}")
    if numpy.__version__ != "1.26.4" or scipy.__version__ != "1.17.1":
        raise SystemExit(
            "v58 is pinned to NumPy 1.26.4 and SciPy 1.17.1; found "
            f"NumPy {numpy.__version__}, SciPy {scipy.__version__}"
        )

    case_root = ARTIFACTS / CASE
    trial = case_root / args.name
    if trial.exists():
        raise FileExistsError(f"preserve existing trial; choose another --name: {trial}")

    runner = ARTIFACTS / RUNNER_NAME
    runner_text = (PACKAGE / "historical/original_runner.py").read_text()
    old_plan_path = 'ServiceLifecycle(e,Path(__file__).parent/"lifecycle/plan.json")'
    new_plan_path = 'ServiceLifecycle(e,Path(__file__).parent/args.case/"lifecycle/plan.json")'
    if runner_text.count(old_plan_path) != 1:
        raise ValueError("historical runner does not contain the expected lifecycle-plan path")
    portable_runner = runner_text.replace(old_plan_path, new_plan_path).encode()
    runner.parent.mkdir(parents=True, exist_ok=True)
    if runner.exists():
        if runner.read_bytes() != portable_runner:
            raise FileExistsError(f"preserve different existing reproduction runner: {runner}")
    else:
        runner.write_bytes(portable_runner)
    for name in [
        "local_ik.py", "local_ik_constrained.py", "local_ik_return.py",
        "local_ik_clearance_strict.py", "local_ik_return_elbow_v2.py",
    ]:
        copy_verified(PACKAGE / "ik" / name, ARTIFACTS / name)
    for path in sorted((PACKAGE / "validation").glob("*.json")):
        copy_verified(path, ARTIFACTS / path.name)
    copy_verified(PACKAGE / "scene/scene.xml", case_root / "review/model/scene.xml")
    copy_verified(PACKAGE / "scene/scene_config.json", case_root / "review/scene_config.json")
    copy_verified(PACKAGE / "scene/review_result.json", case_root / "review/result.json")
    copy_verified(PACKAGE / "scene/initial_standing.npz", case_root / "initial_standing.npz")
    copy_verified(PACKAGE / "scene/initial_standing.json", case_root / "initial_standing.json")
    copy_verified(PACKAGE / "lifecycle/plan.json", case_root / "lifecycle/plan.json")
    scene_path = case_root / "review/model/scene.xml"
    scene_hash = sha(scene_path)
    review_meta = json.loads((case_root / "review/result.json").read_text())
    initial_meta = json.loads((case_root / "initial_standing.json").read_text())
    lifecycle_plan = json.loads((case_root / "lifecycle/plan.json").read_text())
    if review_meta.get("model_sha256") != scene_hash or initial_meta.get("model_sha256") != scene_hash:
        raise ValueError("scene model hash differs from the saved initial-state metadata")
    if lifecycle_plan.get("model_sha256") != scene_hash or lifecycle_plan.get("sampled_prepare_passed") is not True:
        raise ValueError("lifecycle plan hash or sampled-prepare evidence differs from the scene model")
    if initial_meta.get("npz_sha256") != sha(case_root / "initial_standing.npz"):
        raise ValueError("initial-state checkpoint checksum mismatch")
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    with numpy.load(case_root / "initial_standing.npz") as state:
        expected_state_size = mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
        if "state" not in state or state["state"].shape != (expected_state_size,):
            raise ValueError("initial standing checkpoint does not match the scene model")
    from cap_harness.realman.service_lifecycle import ServiceLifecycle
    ServiceLifecycle(
        SimpleNamespace(model_hash=scene_hash, adapter=SimpleNamespace(model=model)),
        case_root / "lifecycle/plan.json",
    )

    baseline_files = json.loads((PACKAGE / "ik/baseline_manifest.json").read_text())["files"]
    for relative, expected in baseline_files.items():
        package_source = PACKAGE / "ik" / Path(relative).name
        if sha(package_source) != expected:
            raise ValueError(f"IK baseline digest mismatch: {relative}")
        copy_verified(package_source, REPO / relative)

    runner_command = json.loads((PACKAGE / "historical/command.json").read_text())
    command = list(runner_command)
    command[0] = sys.executable
    command[1] = str(runner)
    command[command.index("--case") + 1] = CASE
    command[command.index("--name") + 1] = args.name
    command_path = ARTIFACTS / f"{args.name}_command.json"
    if command_path.exists():
        if json.loads(command_path.read_text()) != command:
            raise FileExistsError(f"preserve different existing command: {command_path}")
    else:
        command_path.write_text(json.dumps(command, indent=2) + "\n")

    # Use an immutable copy of the installed Python package for this run. The
    # same source bytes are copied into the trial after it reaches a terminal
    # result and can be selected by replay_coffee_trial.py.
    snapshot_root = ARTIFACTS / "runtime_sources" / args.name
    source_package = REPO / "src/cap_harness"
    if snapshot_root.exists():
        for source in source_package.rglob("*"):
            if source.is_file():
                target = snapshot_root / "cap_harness" / source.relative_to(source_package)
                if not target.is_file() or sha(source) != sha(target):
                    raise FileExistsError(f"existing runtime snapshot differs from source: {snapshot_root}")
    else:
        snapshot_root.mkdir(parents=True)
        shutil.copytree(
            source_package,
            snapshot_root / "cap_harness",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )

    try:
        source_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        source_revision = "unavailable"

    details = {
        "trial": str(trial),
        "command": command,
        "python": sys.version.split()[0],
        "mujoco": mujoco.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "source_git_revision": source_revision,
        "portable_scene_sha256": scene_hash,
        "expected_historical_scene_sha256": json.loads((PACKAGE / "historical/summary.json").read_text())["historical_model_sha256"],
        "source_snapshot_directory": str(snapshot_root),
        "simulation_only": True,
        "physical_motion_allowed": False,
    }
    plan_path = trial.parent / f"{args.name}_reproduction_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != details:
        raise FileExistsError(f"preserve different reproduction plan: {plan_path}")
    plan_path.write_text(json.dumps(details, indent=2) + "\n")
    if args.prepare_only:
        print(json.dumps({"prepared": True, **details}, indent=2))
        return 0

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(snapshot_root) + os.pathsep + environment.get("PYTHONPATH", "")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MUJOCO_GL", "egl")
    log_path = trial.parent / f"{args.name}.log"
    with log_path.open("wb") as log:
        result = subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)

    if trial.is_dir() and (trial / "experiment.py").is_file():
        runtime_snapshot(snapshot_root, trial, runner)
    result_path = trial / "result.json"
    if not result_path.is_file():
        print(json.dumps({"run_exit_code": result.returncode, "log": str(log_path), "trial": str(trial)}, indent=2))
        return result.returncode or 1
    recorded = json.loads(result_path.read_text())
    if recorded.get("experiment_success") is not True or recorded.get("error") is not None:
        print(json.dumps({"experiment_success": recorded.get("experiment_success"), "error": recorded.get("error"), "log": str(log_path), "trial": str(trial)}, indent=2))
        return 1

    with log_path.open("ab") as audit_log:
        audit = subprocess.run(
            [sys.executable, str(REPO / "tools/realman/audit_coffee_delivery_session.py"), str(trial), "--expected-cups", "3"],
            cwd=REPO, env=environment, stdout=audit_log, stderr=subprocess.STDOUT,
        )
        speed_audit = subprocess.run(
            [sys.executable, str(REPO / "tools/realman/audit_coffee_speed_policy.py"), str(trial), "--output", str(trial / "speed_policy_audit.json")],
            cwd=REPO, env=environment, stdout=audit_log, stderr=subprocess.STDOUT,
        )
    status = {
        "experiment_success": True,
        "runner_exit_code": result.returncode,
        "delivery_audit_exit_code": audit.returncode,
        "speed_policy_audit_exit_code": speed_audit.returncode,
        "trial": str(trial),
        "log": str(log_path),
        "simulation_seconds": recorded.get("simulation_seconds"),
        "model_sha256": recorded.get("model_sha256"),
        "historical_v58_result_replayed": False,
    }
    print(json.dumps(status, indent=2))
    return 0 if audit.returncode == 0 and speed_audit.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
