# v58 reproduction inputs

This directory contains the small source and input set needed to launch a fresh v58-parameter simulation. It deliberately excludes the original run's rendered video and multi-gigabyte traces.

- `source/`: preserved simulation runner from the original v58 attempt.
- `historical/`: original launch arguments, success summary, and acceptance/speed audits. `original_result.json` is the recorded comparison reference.
- `scene/`: scene model, scene configuration, initial integrated state, matching metadata, and a verified archive of all 1,320 `base_housing_*.obj` meshes plus their source manifest. The launcher extracts these meshes to the ignored artifacts directory and resolves all STL/OBJ paths relative to the repository.
- `lifecycle/` and `validation/`: lifecycle motion plan and three small validation records whose hashes are recorded by the runner.
- `ik/`: local solver implementations and the minimal hash manifest required for future trial replay.
- `input_manifest.json`: SHA-256 and byte count for each reproduction input.

The archived v58 did not contain a complete runtime module snapshot. A new run therefore cannot be claimed as a byte-identical replay of that historical run. `tools/realman/run_coffee_v58_reproduction.py` captures a complete source snapshot for each new run, making that new run replayable.
