"""Cup-arm rejoin planning with the other arm's verified pot grasp preserved.

The RRT sampler follows joint_planning.py. Only existing stationary-arm contacts
within the original ContactRollout pad neighbourhood may be exempted. Moving-arm
contacts and robot/environment contacts remain collision failures.
"""

import numpy as np

from .workspace import robot_contacts


def plan_service_joint_path(
    adapter,
    arm,
    target,
    *,
    allowed_contacts=(),
    seed=51,
    iterations=500,
    step_rad=0.3,
    resolution_rad=0.025,
):
    """Other arm and objects remain fixed. No execution or dynamics qualification."""
    if arm not in adapter.arms or not 1 <= iterations <= 10000:
        raise ValueError("invalid arm or iteration budget")
    if not np.isfinite([step_rad, resolution_rad]).all() or not 0 < resolution_rad <= step_rad <= 1:
        raise ValueError("invalid planner step")
    m, mj = adapter.model, adapter.mj
    stationary = "left" if arm == "right" else "right"
    allowed = set(map(tuple, allowed_contacts))
    if any(not geom.startswith(stationary + "_") or body != "pot" for geom, body in allowed):
        raise ValueError("only stationary-arm verified pot contacts may be permitted")
    if allowed:
        evidence = adapter.task_evidence["evidence"]
        if stationary != "left" or evidence["pot_handle_in_left_gripper"].value is not True:
            raise ValueError("stationary pot grasp not independently verified")
    indices = adapter.qadr[arm]
    limits = m.jnt_range[adapter.joint_ids[arm]]
    target = np.asarray(target, dtype=float)
    if (
        target.shape != (7,)
        or not np.isfinite(target).all()
        or np.any(target < limits[:, 0])
        or np.any(target > limits[:, 1])
    ):
        raise ValueError("target outside joint limits")
    data = mj.MjData(m)
    data.qpos[:] = adapter.data.qpos
    start = data.qpos[indices].copy()
    checks = 0

    def free(q):
        nonlocal checks
        checks += 1
        data.qpos[indices] = q
        mj.mj_forward(m, data)
        for contact in robot_contacts(m, data):
            names = [contact["geom1"], contact["geom2"]]
            permitted = False
            for index in (0, 1):
                tool, obj = names[index], names[1 - index]
                if not obj.startswith("handle_collision_") or contact["penetration_m"] > 0.002:
                    continue
                if (tool, "pot") in allowed:
                    permitted = True
                for side in ["Left", "Right"]:
                    if tool != f"{stationary[0]}tool_{side}_Support_Link_collision_0":
                        continue
                    pad = f"{stationary}_{side.lower()}_pad_contact"
                    if (pad, "pot") in allowed:
                        site = data.site(f"{stationary}_{side.lower()}_pad_surface")
                        local = site.xmat.reshape(3, 3).T @ (
                            np.asarray(contact["position_world_m"]) - site.xpos
                        )
                        if np.all(np.abs(local) <= [0.006, 0.020, 0.016]):
                            permitted = True
            if not permitted:
                return False
        return True

    def edge(a, b):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / resolution_rad)))
        return all(free(a + (b - a) * (i / n)) for i in range(1, n + 1))

    def report(path, reason, used):
        return {
            "arm": arm,
            "stationary_arm_permitted_contacts": sorted(allowed),
            "found": path is not None,
            "reason": reason,
            "waypoints_urdf_rad": None if path is None else [q.tolist() for q in path],
            "seed": seed,
            "iterations": used,
            "collision_checks": checks,
            "max_joint_step_rad": resolution_rad,
            "continuous_collision_checked": False,
            "dynamics_checked": False,
            "physical_motion_allowed": False,
        }

    if not free(start) or not free(target):
        return report(None, "start_or_goal_collision", 0)
    if edge(start, target):
        return report([start, target], "direct_sampled_segment", 0)
    rng = np.random.default_rng(seed)
    trees = [([start], [-1]), ([target], [-1])]

    def extend(tree, q):
        nodes, parents = tree
        nearest = int(np.argmin([np.linalg.norm(node - q) for node in nodes]))
        delta = q - nodes[nearest]
        distance = float(np.linalg.norm(delta))
        if distance < 1e-10:
            return nearest, True
        new = nodes[nearest] + delta * min(1.0, step_rad / distance)
        if not edge(nodes[nearest], new):
            return None, False
        nodes.append(new)
        parents.append(nearest)
        return len(nodes) - 1, distance <= step_rad

    def trace(tree, index):
        chain = []
        while index != -1:
            chain.append(tree[0][index])
            index = tree[1][index]
        return chain[::-1]

    for iteration in range(iterations):
        active = iteration % 2
        q = rng.uniform(limits[:, 0], limits[:, 1])
        index, _ = extend(trees[active], q)
        if index is None:
            continue
        q = trees[active][0][index]
        while True:
            other_index, reached = extend(trees[1 - active], q)
            if other_index is None:
                break
            if reached:
                indices_found = [None, None]
                indices_found[active], indices_found[1 - active] = index, other_index
                path = trace(trees[0], indices_found[0]) + trace(trees[1], indices_found[1])[-2::-1]
                # Deterministic shortcutting retains collision sampling on every edge.
                simplified = [path[0]]
                i = 0
                while i < len(path) - 1:
                    j = len(path) - 1
                    while j > i + 1 and not edge(path[i], path[j]):
                        j -= 1
                    simplified.append(path[j])
                    i = j
                return report(simplified, "rrt_connect_sampled_path", iteration + 1)
    return report(None, "search_budget_exhausted", iterations)
