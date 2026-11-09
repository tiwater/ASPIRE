"""Prepare or run a recorded simulation with its captured RealMan modules.

Uses a new output name and the recorded controller bytes. A local resumed trial
remains local; replaying it never grants continuous-workflow qualification.
Models retain their recorded workspace paths and must already be restored.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import sys

LAUNCHER_SOURCE = Path(__file__).read_bytes()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_path(snapshots: Path, source_path: str) -> Path:
    """Resolve both portable package-tree snapshots and legacy flat snapshots."""
    relative = Path(source_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe snapshot path: " + source_path)
    if relative.parts[:2] == ("src", "cap_harness"):
        portable = snapshots.joinpath(*relative.parts[1:])
        if portable.is_file():
            return portable
    if source_path == "experiment.py" and (snapshots / "experiment.py").is_file():
        return snapshots / "experiment.py"
    return snapshots / relative.name


def prepare(trial, name):
    trial = Path(trial).resolve()
    repo = Path(__file__).resolve().parents[2]
    root = repo / "artifacts/realman/coffee_service_20261102"
    if root not in trial.parents or not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("require a coffee service experiment and a simple new name")
    result = json.loads((trial / "result.json").read_text())
    if result.get("simulation_only") is not True or result.get("physical_motion_allowed") is not False:
        raise ValueError("only recorded simulation trials are supported")
    arguments = result["parameters"]["arguments"]
    command_candidates = {
        directory / (trial.name + "_command.json")
        for directory in [trial.parent.parent, root]
    }
    command_paths = [path for path in command_candidates if path.is_file()]
    if len(command_paths) != 1:
        raise ValueError("require one unambiguous recorded command")
    command_path = command_paths[0]
    command = json.loads(command_path.read_text())
    runner_root = Path(command[1]).parent.resolve()
    if (
        arguments["name"] != trial.name
        or command[command.index("--name") + 1] != trial.name
        or arguments["case"] != trial.parent.name
        or command[command.index("--case") + 1] != trial.parent.name
        or runner_root not in {root, root / "speed_probe"}
        or trial.parent not in {
            runner_root / arguments["case"],
            runner_root / "speed_probe" / arguments["case"],
        }
    ):
        raise ValueError("recorded command does not identify this trial layout")
    snapshots = trial / "implementation_snapshots"
    manifest = json.loads((snapshots / "captured_during_run.json").read_text())
    for path, expected in manifest.items():
        source = snapshot_path(snapshots, path)
        if not source.is_file() or digest(source) != expected:
            raise ValueError("captured source hash mismatch: " + path)
    source = trial / "experiment.py"
    if digest(source) not in manifest.values():
        raise ValueError("controller was not captured with these modules")
    model = trial.parent / "review/model/scene.xml"
    if digest(model) != result["model_sha256"]:
        raise ValueError("recorded model XML changed")
    baseline = repo / "assets/realman/coffee/v58_reproduction/ik/baseline_manifest.json"
    if not baseline.is_file():
        baseline = repo / "assets/realman/coffee/continuous_coffee_20261101/manifest.json"
    baseline_files = json.loads(baseline.read_text())["files"]
    local_ik = {}
    for module_name in ["local_ik", "local_ik_constrained", "local_ik_return", "local_ik_clearance_strict"]:
        relative = "artifacts/realman/continuous_coffee_20261101/" + module_name + ".py"
        path = repo / relative
        if digest(path) != baseline_files[relative]:
            raise ValueError("restore the unchanged baseline IK dependency: " + module_name)
        # These coffee candidates retained the baseline IK helpers. Refuse a
        # silently changed working copy, including one from another experiment.
        if digest(runner_root / (module_name + ".py")) != baseline_files[relative]:
            raise ValueError("working IK helper differs from the recorded baseline: " + module_name)
        local_ik[module_name] = {"path": str(path), "sha256": baseline_files[relative]}
    for filename, module_name in [
        ("pour_ik_impl.py", "local_ik_constrained"),
        ("return_ik_impl.py", "local_ik_return"),
    ]:
        if digest(trial / filename) != local_ik[module_name]["sha256"]:
            raise ValueError("recorded trial used a different IK implementation: " + filename)
    # New diagnostic helpers must have their own captured provenance. They do
    # not replace the four original baseline dependencies above.
    for helper in ["local_ik_return_elbow", "local_ik_return_elbow_v2"]:
        captured = [path for path in manifest if Path(path).name == helper + ".py"]
        if captured:
            path = snapshots / (helper + ".py")
            expected = manifest[captured[0]]
            if (
                digest(trial / "return_elbow_impl.py") != expected
                or digest(runner_root / (helper + ".py")) != expected
            ):
                raise ValueError("return elbow helper differs from its captured implementation")
            local_ik[helper] = {"path": str(path), "sha256": expected}
    output = trial.parent / name
    replay = runner_root / ("replay_" + name + ".py")
    plan_path = runner_root / (name + "_replay_plan.json")
    replay_command_path = runner_root / (name + "_command.json")
    if any(path.exists() for path in [output, replay, plan_path, replay_command_path]):
        raise FileExistsError("choose a new replay name; existing evidence is never replaced")
    if not Path(command[0]).is_file():
        raise FileNotFoundError("recorded Python environment is unavailable")
    command[1] = str(replay)
    command[command.index("--name") + 1] = name
    plan = {
        "source_trial": str(trial),
        "source_result_sha256": digest(trial / "result.json"),
        "launcher_sha256": hashlib.sha256(LAUNCHER_SOURCE).hexdigest(),
        "controller_sha256": digest(source),
        "captured_manifest_sha256": digest(snapshots / "captured_during_run.json"),
        "model_sha256": digest(model),
        "command": command,
        "output": str(output),
        "snapshot_directory": str(snapshots),
        "local_ik_modules": local_ik,
        "simulation_only": True,
        "physical_motion_allowed": False,
        "source_phase": arguments["phase"],
        "source_checkpoint": arguments.get("checkpoint"),
        "qualification_inherited": False,
    }
    replay.write_bytes(source.read_bytes())
    replay_command_path.write_text(json.dumps(command, indent=2) + "\n")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan, plan_path


def execute(plan):
    command = plan["command"]
    if Path(sys.executable).resolve() != Path(command[0]).resolve():
        raise ValueError("invoke this tool with the recorded Python interpreter")
    repo = Path(__file__).resolve().parents[2]
    snapshots = Path(plan["snapshot_directory"])
    package_snapshot = snapshots / "cap_harness"
    if package_snapshot.is_dir():
        sys.path.insert(0, str(snapshots))
    else:
        sys.path.insert(0, str(repo / "src"))
    import cap_harness

    if any(name.startswith("cap_harness.realman") for name in sys.modules):
        raise RuntimeError("RealMan modules must not be loaded before snapshot selection")
    realman_snapshot = package_snapshot / "realman" if (package_snapshot / "realman").is_dir() else snapshots
    spec = importlib.util.spec_from_file_location(
        "cap_harness.realman",
        realman_snapshot / "__init__.py",
        submodule_search_locations=[str(realman_snapshot)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    cap_harness.realman = module
    spec.loader.exec_module(module)
    for name, entry in plan["local_ik_modules"].items():
        dependency = Path(entry["path"])
        if digest(dependency) != entry["sha256"] or name in sys.modules:
            raise ValueError("IK dependency changed or was loaded before selection: " + name)
        spec = importlib.util.spec_from_file_location(name, dependency)
        dependency_module = importlib.util.module_from_spec(spec)
        sys.modules[name] = dependency_module
        spec.loader.exec_module(dependency_module)
    output = Path(plan["output"])
    sys.argv = command[1:]
    try:
        runpy.run_path(command[1], run_name="__main__")
    finally:
        if output.is_dir():
            shutil.copytree(snapshots, output / "implementation_snapshots")
            (output / "replay_launcher.py").write_bytes(LAUNCHER_SOURCE)
            (output / "reproduction.json").write_text(json.dumps(plan, indent=2) + "\n")
    result = json.loads((output / "result.json").read_text())
    return 0 if result["experiment_success"] and result["error"] is None else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trial", type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--execute", action="store_true", help="run; otherwise only prepare a replay")
    args = parser.parse_args()
    # Set before the captured controller imports numerical/rendering libraries.
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MUJOCO_GL", "egl")
    plan, path = prepare(args.trial, args.name)
    print(json.dumps({"replay_plan": str(path), "executing": args.execute}), flush=True)
    if args.execute:
        raise SystemExit(execute(plan))


if __name__ == "__main__":
    main()
