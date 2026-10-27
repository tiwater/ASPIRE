# Run only with: python -m cap_harness.realman run THIS_FILE
# A small joint move verifies ASPIRE's seven-joint adapter plumbing.
# It is not a coffee policy and must never be replayed on the physical robot.
state = get_robot_state()
target = state.joint_positions["right"].copy()
target[6] += 0.03
motion = move_to_joints(target, arm="right", tolerance=0.01, max_steps=120)
result = {"motion_ok": motion.ok, "arms": list(state.arms),
          "joints_per_arm": len(target), "coffee_task_success": None}
