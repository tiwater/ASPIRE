from pathlib import Path

import numpy as np
import pytest

from cap_harness.contracts import ArmCommand, RobotAction, Trajectory
from cap_harness.realman.ik import solve_mount_ik
from cap_harness.realman.model import build_model
from cap_harness.realman.sim import CoffeeSimAdapter

mujoco = pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    urdf = Path("ref/realman_robot/LSRX_V0.0.13.SLDASM/urdf/LSRX_V0.0.13.SLDASM.urdf")
    if not urdf.exists():
        pytest.skip("requires user-supplied RealBot URDF/meshes")
    out = tmp_path_factory.mktemp("realman_model")
    return build_model(urdf, Path("configs/realman/coffee_scene.json"), out)


@pytest.fixture
def adapter(built):
    a = CoffeeSimAdapter(built["scene"], width=160, height=120)
    yield a
    a.close()


def test_import_preserves_seven_axes_and_gripper_mimics(built, adapter):
    assert built["actuators"] == 16
    assert built["mimic_constraints"] == 10
    assert "l_camera_rojoint" in built["frozen_urdf_joints"]
    state = adapter.get_robot_state()
    assert state.embodiment == "realman"
    assert all(len(q) == 7 for q in state.joint_positions.values())
    assert state.gripper_positions["left"] == pytest.approx(1)
    assert adapter.model.body("pot").mass[0] == pytest.approx(0.396, abs=1e-8)


def test_gripper_linkage_does_not_shear_under_unilateral_handle_load(adapter):
    """A mimic constraint represents linkage geometry, not soft material flex."""
    m, d = adapter.model, adapter.data
    body = m.body("ltool_Right_Pad_Link").id
    maximum = 0.0
    for _ in range(50):
        d.xfrc_applied[body, :3] = 15 * d.body(body).xmat.reshape(3, 3)[:, 0]
        adapter._advance(0.01)
        for i in range(m.neq):
            j1, j2 = m.eq_obj1id[i], m.eq_obj2id[i]
            if m.joint(j1).name.startswith("ltool_"):
                error = (
                    d.qpos[m.jnt_qposadr[j1]]
                    - m.eq_data[i, 0]
                    - m.eq_data[i, 1] * d.qpos[m.jnt_qposadr[j2]]
                )
                maximum = max(maximum, abs(float(error)))
    assert maximum < 0.003  # less than 0.18 deg, versus multi-degree soft linkage drift
    assert not np.any(d.warning.number)


def test_grasp_frame_tracks_opening_without_redefining_mount(adapter):
    from cap_harness.realman.grasp_geometry import grasp_geometry, mount_pose_for_grasp

    m = adapter.model
    d = mujoco.MjData(m)
    d.qpos[:] = adapter.data.qpos
    mujoco.mj_forward(m, d)
    initial = grasp_geometry(m, d, "right")
    assert initial["signed_aperture_m"] == pytest.approx(0.088, abs=1e-5)
    assert initial["T_mount_grasp"][2][3] == pytest.approx(0.183515, abs=1e-5)
    mount_before = d.site("right_tool_mount").xpos.copy()
    driver = m.joint("rtool_Left_1_Joint").id
    d.qpos[m.jnt_qposadr[driver]] = -0.455
    for eid in range(m.neq):
        if m.eq_type[eid] == mujoco.mjtEq.mjEQ_JOINT and m.eq_obj2id[eid] == driver:
            d.qpos[m.jnt_qposadr[m.eq_obj1id[eid]]] = m.eq_data[eid, 0] + m.eq_data[eid, 1] * -0.455
    mujoco.mj_forward(m, d)
    closing = grasp_geometry(m, d, "right")
    assert 0 < closing["signed_aperture_m"] < initial["signed_aperture_m"]
    assert closing["T_mount_grasp"][2][3] > initial["T_mount_grasp"][2][3] + 0.01
    np.testing.assert_allclose(d.site("right_tool_mount").xpos, mount_before)
    target = np.eye(4)
    target[:3, :3] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]
    target[:3, 3] = [0.4, -0.3, 0.9]
    mounting = mount_pose_for_grasp(target, closing["T_mount_grasp"])
    np.testing.assert_allclose(mounting @ closing["T_mount_grasp"], target, atol=1e-12)
    assert not closing["hardware_tcp_qualified"] and not closing["grasp_verified"]


def test_pad_proxy_collides_at_surface_and_leaves_open_gap(built):
    import xml.etree.ElementTree as ET

    from cap_harness.realman.grasp_geometry import grasp_geometry

    root = ET.parse(built["scene"]).getroot()
    probe = ET.SubElement(root.find("worldbody"), "body", name="pad_contact_probe")
    ET.SubElement(probe, "freejoint", name="pad_contact_probe_free")
    ET.SubElement(probe, "geom", name="pad_probe", type="sphere", size=".006", mass=".01")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    adr = m.joint("pad_contact_probe_free").qposadr[0]
    probe_id = m.geom("pad_probe").id
    pad_ids = {m.geom("right_" + side + "_pad_contact").id for side in ["left", "right"]}
    geometry = grasp_geometry(m, d, "right")

    def touches_pad(position):
        d.qpos[adr : adr + 3] = position
        mujoco.mj_forward(m, d)
        return any(
            probe_id in (c.geom1, c.geom2)
            and pad_ids.intersection((c.geom1, c.geom2))
            and c.dist < 0
            for c in d.contact
        )

    assert not touches_pad(np.array(geometry["T_world_grasp"])[:3, 3])
    for side in ["left", "right"]:
        site = d.site("right_" + side + "_pad_surface")
        # Contact from the inward (-local-X) side, with 0.5 mm overlap.
        position = site.xpos.copy() - site.xmat.reshape(3, 3)[:, 0] * 0.0055
        assert touches_pad(position)


def test_cup_is_hollow(adapter):
    m, d = adapter.model, adapter.data
    pos = d.body("cup_3").xpos + [0, 0, 0.09]
    gid = np.array([-1], dtype=np.int32)
    distance = mujoco.mj_ray(m, d, pos, np.array([0.0, 0.0, -1.0]), None, 1, -1, gid)
    assert 0.08 < distance < 0.095
    assert m.geom(int(gid[0])).name == "cup_3_bottom"


def test_side_camera_frames_coffee_machine(adapter):
    m, d = adapter.model, adapter.data
    machine = m.body("coffee_machine").id
    machine_geoms = np.flatnonzero(m.geom_bodyid == machine)
    with mujoco.Renderer(m, height=120, width=160) as renderer:
        renderer.update_scene(d, camera="side")
        rgb = renderer.render().copy()
        renderer.enable_segmentation_rendering()
        renderer.update_scene(d, camera="side")
        segmentation = renderer.render()
    # Bright pixels alone could be a wrongly framed table or robot. Require
    # visible coffee-machine geometry as well as a nonblank RGB image.
    visible = np.isin(segmentation[:, :, 0], machine_geoms)
    visible &= segmentation[:, :, 1] == mujoco.mjtObj.mjOBJ_GEOM
    assert visible.mean() > 0.08
    assert rgb.std() > 10


def test_machine_controls_follow_cutaway_head_slope(adapter):
    m, d = adapter.model, adapter.data
    origin = d.body("coffee_machine").xpos

    def surface(x, y):
        gid = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            m, d, origin + [x, y, 0.60], np.array([0.0, 0.0, -1.0]), None, 1, -1, gid
        )
        assert distance >= 0
        return 0.60 - distance, m.geom(int(gid[0])).name

    # Bare side margin: the solid head itself must slope, not just a rotated
    # panel layered on an unchanged rectangular head.
    for x, expected in [(-0.18, 0.395), (-0.13, 0.420), (0.0, 0.430)]:
        z, name = surface(x, 0.157)
        assert name == "machine_head"
        assert z == pytest.approx(expected, abs=0.0001)
    z, name = surface(-0.155, 0)
    assert name == "machine_controls"
    assert z == pytest.approx(0.4075 + 0.002 / np.cos(np.arctan(0.5)), abs=0.0001)
    normal = d.body("machine_controls_mount").xmat.reshape(3, 3)[:, 2]
    np.testing.assert_allclose(normal, np.array([-1.0, 0, 2.0]) / np.sqrt(5), atol=1e-8)


def test_pot_bay_has_five_mm_radial_clearance_and_chamfered_open_ends(adapter):
    m, d = adapter.model, adapter.data
    center = d.body("pot").xpos
    groups = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)

    def gap(angle, z=0.09):
        angle = np.radians(angle)
        direction = np.array([np.cos(angle), np.sin(angle), 0.0])
        gid = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            m, d, center + 0.069 * direction + [0, 0, z], direction, groups, 1, -1, gid
        )
        name = m.geom(int(gid[0])).name if gid[0] >= 0 else None
        return distance + 0.001 if distance >= 0 else None, name

    # Rear is machine +X. Measure clearance against the belly envelope at
    # several angles/heights, rather than just checking the configured radius.
    for z in (0.02, 0.09, 0.13):
        for angle in (-60, -35, 0.5, 35, 60):
            distance, name = gap(angle, z)
            assert name.startswith("pot_bay_wall_collision_")
            assert distance == pytest.approx(0.005, abs=0.0001)
    for angle in (-69, 69):
        distance, name = gap(angle)
        assert name.startswith("pot_bay_wall_collision_")
        assert 0.008 < distance < 0.012  # edge chamfer opens away from the pot
    for angle in (135, 180, 225):
        assert gap(angle)[0] is None  # no enclosing wall across the front opening


def test_tray_rear_wall_is_flush_with_bay_chamfer_end(adapter):
    m, d = adapter.model, adapter.data
    origin = d.body("coffee_machine").xpos
    groups = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)
    end_x = -0.075 + 0.079 * np.cos(np.radians(70))
    for y in (-0.14, -0.077, -0.03, 0.15):
        for z in (0.06, 0.13, 0.19):
            gid = np.array([-1], dtype=np.int32)
            # The accepted handle can occlude the opposite chamfer end. This
            # measurement targets the housing plane, so exclude the pot body.
            distance = mujoco.mj_ray(
                m,
                d,
                origin + [-0.18, y, z],
                np.array([1.0, 0, 0]),
                groups,
                1,
                m.body("pot").id,
                gid,
            )
            assert m.geom(int(gid[0])).name.startswith("pot_bay_wall_collision_")
            assert -0.18 + distance == pytest.approx(end_x, abs=0.0001)
    tray = m.geom("drip_tray")
    assert tray.pos[0] + tray.size[0] == pytest.approx(end_x)
    for i in range(7):
        slot = m.geom(f"tray_slot{i}")
        assert slot.pos[0] + slot.size[0] < end_x


def test_pot_bay_collision_preserves_gap_instead_of_filling_cavity(built):
    import xml.etree.ElementTree as ET

    root = ET.parse(built["scene"]).getroot()
    probe = ET.SubElement(root.find("worldbody"), "body", name="bay_probe")
    ET.SubElement(probe, "freejoint", name="bay_probe_free")
    ET.SubElement(probe, "geom", name="bay_probe_geom", type="sphere", size="0.002", mass="0.01")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    center = d.body("pot").xpos.copy()
    adr = m.jnt_qposadr[m.joint("bay_probe_free").id]
    gid = m.geom("bay_probe_geom").id

    def contacts(radius, angle):
        angle = np.radians(angle)
        d.qpos[adr : adr + 3] = center + [radius * np.cos(angle), radius * np.sin(angle), 0.09]
        mujoco.mj_forward(m, d)
        return [c for c in d.contact if gid in (c.geom1, c.geom2) and c.dist < 0]

    for angle in (-55, 0, 55):
        assert not contacts(0.0705, angle)  # 4 mm sphere fits the 5 mm gap
        touching = contacts(0.073, angle)
        assert any(
            m.geom(c.geom1 if c.geom2 == gid else c.geom2).name.startswith(
                "pot_bay_wall_collision_"
            )
            for c in touching
        )


def test_pot_handle_aperture_and_grip_are_real_collision_geometry(adapter):
    """A 30 mm hole must survive convex compilation, with a 14 mm outer grip."""
    m, d = adapter.model, adapter.data
    pose = d.body("pot")
    rotation = pose.xmat.reshape(3, 3)
    group = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)

    def ray(point, direction):
        gid = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            m, d, pose.xpos + rotation @ point, rotation @ direction, group, 1, -1, gid
        )
        return distance

    # Photo-derived nominal aperture centre, including the lower inset.
    center = np.array([0.094, 0, 0.081])
    assert ray(center, np.array([1.0, 0, 0])) == pytest.approx(0.015, abs=0.0001)
    assert ray(center, np.array([-1.0, 0, 0])) == pytest.approx(0.015, abs=0.0001)
    assert ray(center, np.array([0.0, 1, 0])) == -1
    assert ray(center, np.array([0.0, -1, 0])) == -1
    # Grasp site follows the solid outer grip, midway through its radial thickness.
    grasp = rotation.T @ (d.site("pot_handle_grasp").xpos - pose.xpos)
    assert ray(grasp, np.array([1.0, 0, 0])) == pytest.approx(0.007, abs=0.0001)
    assert ray(grasp, np.array([-1.0, 0, 0])) == pytest.approx(0.007, abs=0.0001)
    # The former square outer corners must not block a finger beside the rounding.
    assert ray(np.array([0.123, -0.05, 0.130]), np.array([0.0, 1, 0])) == -1


def test_pot_handle_aperture_and_front_access_with_rear_wall(built):
    """Separate intrinsic aperture clearance from the seated pot's approach space."""
    import xml.etree.ElementTree as ET

    root = ET.parse(built["scene"]).getroot()
    probe = ET.SubElement(root.find("worldbody"), "body", name="finger_gauge")
    ET.SubElement(probe, "freejoint", name="finger_gauge_free")
    ET.SubElement(probe, "geom", name="finger_gauge_geom", type="sphere", size="0.014", mass="0.01")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    origin, rotation = d.body("pot").xpos.copy(), d.body("pot").xmat.reshape(3, 3).copy()
    adr = m.jnt_qposadr[m.joint("finger_gauge_free").id]
    probe_geom = m.geom("finger_gauge_geom").id

    def contacts_at(point):
        d.qpos[adr : adr + 3] = origin + rotation @ np.array(point)
        mujoco.mj_forward(m, d)
        return [c for c in d.contact if probe_geom in (c.geom1, c.geom2) and c.dist < 0]

    def other_geom(contact):
        return contact.geom1 if contact.geom2 == probe_geom else contact.geom2

    # The 28 mm gauge still clears the pot itself through the full handle depth.
    # The close-fitting machine housing now legitimately obstructs the rear
    # approach; it must not be mistaken for a filled-in handle aperture.
    for y in np.linspace(-0.05, 0.05, 11):
        assert all(
            m.geom_bodyid[other_geom(c)] != m.body("pot").id for c in contacts_at([0.094, y, 0.081])
        )
    for y in np.linspace(0.05, 0, 6):
        assert not contacts_at([0.094, y, 0.081])  # machine front -> aperture centre
    assert any(
        m.geom(other_geom(c)).name.startswith("pot_bay_wall_collision_")
        for c in contacts_at([0.094, -0.05, 0.081])
    )
    assert contacts_at([0.116, 0, 0.081])  # outer grip
    assert contacts_at([0.094, 0, 0.125])  # top rail
    assert contacts_at([0.092, 0, 0.037])  # bottom rail


def test_pot_spout_is_recessed_over_neck_and_channel_is_open(adapter):
    """Check the compiled collision silhouette and the lid, not just the site."""
    m, d = adapter.model, adapter.data
    origin = d.body("pot").xpos
    rotation = d.body("pot").xmat.reshape(3, 3)
    glass_group = np.array([0, 0, 0, 1, 0, 0], dtype=np.uint8)

    def ray(point, direction, group=None):
        gid = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            m,
            d,
            origin + rotation @ np.array(point),
            rotation @ np.array(direction),
            group,
            1,
            -1,
            gid,
        )
        return distance, m.geom(int(gid[0])).name if gid[0] >= 0 else None

    # Offset 0.1 mm from the angular seam: a ray exactly on the shared edge is
    # sensitive to float32 mesh recentering, unlike a finite-size contact probe.
    radii = []
    for z in (0.090, 0.126, 0.1399):
        distance, name = ray([-0.10, 0.0001, z], [1.0, 0, 0], glass_group)
        assert name.startswith("pot_glass_collision_")
        radii.append(0.10 - distance)
    belly, neck, lip = radii
    assert belly == pytest.approx(0.068, abs=0.0001)
    assert neck == pytest.approx(0.055, abs=0.0001)
    assert lip == pytest.approx(0.066, abs=0.0002)
    assert neck + 0.008 < lip <= belly
    np.testing.assert_allclose(m.site("pot_spout").pos, [-0.066, 0, 0.140])

    # The lid must not seal the pouring channel; its front opening sees glass
    # below the lid's underside, while the neighbouring covered area sees lid.
    # Start below the coffee machine head, which is 153 mm above the pot bottom.
    distance, name = ray([-0.058, 0.0001, 0.15], [0.0, 0, -1])
    assert name.startswith("pot_glass_") and 0.15 - distance < 0.134
    _, name = ray([-0.040, 0.0001, 0.15], [0.0, 0, -1])
    assert name.startswith("pot_lid_segment_")
    # The new vessel meshes must remain hollow below the lid.
    distance, name = ray([0, 0, 0.12], [0.0, 0, -1])
    assert name == "pot_bottom" and distance == pytest.approx(0.117, abs=0.0001)


def test_pot_neck_recess_is_not_filled_by_collision_hull(built):
    import xml.etree.ElementTree as ET

    root = ET.parse(built["scene"]).getroot()
    probe = ET.SubElement(root.find("worldbody"), "body", name="neck_probe")
    ET.SubElement(probe, "freejoint", name="neck_probe_free")
    ET.SubElement(probe, "geom", name="neck_probe_geom", type="sphere", size="0.003", mass="0.01")
    m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    origin, rotation = d.body("pot").xpos.copy(), d.body("pot").xmat.reshape(3, 3).copy()
    adr = m.jnt_qposadr[m.joint("neck_probe_free").id]
    gid = m.geom("neck_probe_geom").id

    def touches(point):
        d.qpos[adr : adr + 3] = origin + rotation @ np.array(point)
        mujoco.mj_forward(m, d)
        return any(gid in (c.geom1, c.geom2) and c.dist < 0 for c in d.contact)

    assert not touches([-0.062, 0, 0.126])  # indentation below spout, on angular seam
    assert touches([-0.068, 0, 0.09])  # glass belly
    assert touches([-0.066, 0, 0.139])  # integral lip
    assert not touches([0, 0, 0.09])  # interior remains hollow


def test_nominal_stack_does_not_fall_off_dispenser(adapter):
    before = {
        n: adapter.data.body(n).xpos.copy() for n in ["pot", "cup_0", "cup_1", "cup_2", "cup_3"]
    }
    adapter._advance(10.0)
    assert not adapter.data.warning.number.any()
    for n, old in before.items():
        assert np.linalg.norm(adapter.data.body(n).xpos - old) < 0.01
    assert adapter.data.body("cup_0").xpos[2] > 0.82


def test_invalid_batch_is_rejected_before_any_control_write(adapter):
    old = adapter.data.ctrl.copy()
    action = RobotAction(
        {
            "left": ArmCommand("joint_position", np.zeros(7), embodiment="realman"),
            "right": ArmCommand("joint_position", np.ones(7) * 99, embodiment="realman"),
        }
    )
    assert not adapter.step(action).ok
    np.testing.assert_array_equal(old, adapter.data.ctrl)


def test_stale_trajectory_does_not_run(adapter):
    state = adapter.get_robot_state()
    t = Trajectory(
        np.zeros((1, 7)),
        0.02,
        state.joint_names["right"],
        "test",
        False,
        state,
        arm="right",
        embodiment="realman",
    )
    adapter.data.qpos[adapter.qadr["left"][0]] += 0.1
    adapter.mj.mj_forward(adapter.model, adapter.data)
    result = adapter.execute_trajectory(t)
    assert not result.ok and result.error.code.value == "stale_plan"
    assert adapter.data.time == 0


def test_ik_is_offline_and_does_not_move_sim_state(adapter):
    old = adapter.data.qpos.copy()
    pose = adapter.get_robot_state().end_effector_poses["right"]
    result = solve_mount_ik(adapter, "right", pose.position, pose.quaternion_wxyz, restarts=1)
    assert result["pose_solved"]
    assert not result["path_checked"]
    assert not result["physical_motion_allowed"]
    np.testing.assert_array_equal(adapter.data.qpos, old)


def test_cap_program_runs_through_public_tools(adapter):
    from cap_harness.api import CapApi
    from cap_harness.run import run_program_with_api

    result = run_program_with_api(
        api=CapApi(adapter, default_camera="front"),
        program_path=Path("examples/realman/observe_and_move_sim.py"),
    )
    assert result.ok
    assert result.result["motion_ok"]
    assert result.result["joints_per_arm"] == 7
    assert any(call.tool == "move_to_joints" for call in result.calls)
    assert adapter.get_robot_state().joint_positions["right"][6] == pytest.approx(0.03, abs=0.01)


def test_frozen_body_pose_changes_fk_without_adding_actuators(tmp_path, built):
    import json

    config = json.loads(Path("configs/realman/coffee_scene.json").read_text())
    config["robot"]["frozen_joints_rad"] = {"xt_joint_1": -0.15}
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps(config))
    result = build_model(
        Path("ref/realman_robot/LSRX_V0.0.13.SLDASM/urdf/LSRX_V0.0.13.SLDASM.urdf"),
        scene,
        tmp_path / "model",
    )
    a = CoffeeSimAdapter(result["scene"])
    baseline = CoffeeSimAdapter(built["scene"])
    try:
        assert result["actuators"] == 16
        assert result["frozen_urdf_positions_rad"]["xt_joint_1"] == -0.15
        assert result["upper_body_posture"]["tilt_deg"] < 10
        assert (
            np.linalg.norm(a.data.body("xb_link").xpos - baseline.data.body("xb_link").xpos) > 0.1
        )
    finally:
        a.close()
        baseline.close()


def test_contact_solver_candidate_is_explicit_and_preserves_physical_model(tmp_path, built):
    import json

    config = json.loads(Path("configs/realman/coffee_scene.json").read_text())
    config["contact_solver"].update(timestep_s=0.000125, multiccd=True, ccd_tolerance=1e-8)
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps(config))
    result = build_model(
        Path("ref/realman_robot/LSRX_V0.0.13.SLDASM/urdf/LSRX_V0.0.13.SLDASM.urdf"),
        scene,
        tmp_path / "model",
    )
    candidate = mujoco.MjModel.from_xml_path(str(result["scene"]))
    baseline = mujoco.MjModel.from_xml_path(str(built["scene"]))
    assert candidate.opt.timestep == pytest.approx(0.000125)
    assert candidate.opt.ccd_tolerance == pytest.approx(1e-8)
    assert candidate.opt.enableflags & mujoco.mjtEnableBit.mjENBL_MULTICCD
    assert not baseline.opt.enableflags & mujoco.mjtEnableBit.mjENBL_MULTICCD
    for name in [
        "body_mass",
        "body_inertia",
        "geom_size",
        "geom_friction",
        "jnt_range",
        "actuator_forcerange",
    ]:
        np.testing.assert_allclose(
            getattr(candidate, name), getattr(baseline, name), rtol=0, atol=0
        )


def test_scene_builder_rejects_upper_body_tilt_above_configured_limit(tmp_path):
    import json

    config = json.loads(Path("configs/realman/coffee_scene.json").read_text())
    config["robot"]["frozen_joints_rad"] = {"xt_joint_1": -0.2}
    scene = tmp_path / "scene.json"
    scene.write_text(json.dumps(config))
    with pytest.raises(ValueError, match=r"upper-body tilt.*exceeds"):
        build_model(
            Path("ref/realman_robot/LSRX_V0.0.13.SLDASM/urdf/LSRX_V0.0.13.SLDASM.urdf"),
            scene,
            tmp_path / "model",
        )


def test_four_passive_tabs_clear_walls_and_support_rolled_lip(adapter):
    m, d = adapter.model, adapter.data
    before = d.body("cup_0").xpos.copy()
    adapter._advance(0.5)
    contacts = []
    for c in d.contact:
        names = [m.geom(g).name for g in (c.geom1, c.geom2)]
        if any(n.startswith("retention_pad") for n in names) and any(
            n.startswith("cup_0_rolled_rim") for n in names
        ):
            contacts.append(c)
    assert contacts
    assert np.linalg.norm(d.body("cup_0").xpos - before) < 0.003
    for i in range(4):
        joint = m.joint(f"finger_deflection{i}")
        assert joint.type[0] == mujoco.mjtJoint.mjJNT_HINGE
        assert -0.15 < d.joint(f"finger_deflection{i}").qpos[0] < 0
        assert m.qpos_spring[joint.qposadr[0]] == 0
        assert joint.id not in m.actuator_trnid[:, 0]


def test_all_cup_surfaces_have_lower_friction_than_rack(adapter):
    m, d = adapter.model, adapter.data
    for i in range(m.ngeom):
        if m.body(m.geom_bodyid[i]).name in ["cup_0", "cup_1", "cup_2", "cup_3"]:
            assert m.geom_friction[i, 0] == pytest.approx(0.05)
    adapter._advance(0.5)  # gravity establishes contacts from a nonpenetrating initial pose
    pairs = {}
    for c in d.contact:
        names = [m.body(m.geom_bodyid[g]).name for g in [c.geom1, c.geom2]]
        if names[0].startswith("cup_") and names[1].startswith("cup_"):
            pairs["cup_cup"] = c.friction[0]
        if names[0].startswith("retention_finger") and names[1].startswith("cup_"):
            pairs["cup_rack"] = c.friction[0]
    assert pairs["cup_cup"] == pytest.approx(0.05)
    assert pairs["cup_rack"] == pytest.approx(0.8)


def test_outlet_nose_can_cam_up_over_lid_in_diagonal_withdrawal(adapter):
    """A vertical cylinder inside the lid hole creates a spurious lateral lock."""
    m, d = adapter.model, adapter.data
    pot = m.joint("pot_free").qposadr[0]
    d.qpos[pot : pot + 3] += np.array([-1, 1, 0]) * 0.003 / np.sqrt(2)
    mujoco.mj_forward(m, d)
    contacts = [
        c
        for c in d.contact[: d.ncon]
        if m.geom(c.geom1).name == "outlet_tip" and m.geom(c.geom2).name.startswith("pot_lid_")
    ]
    assert contacts
    # Normal points from nose down to the lid: the equal opposite force can
    # raise the spring slider. Prior flat-lid/cylinder normals had Z ~= 0.
    assert all(c.frame[2] < -0.2 for c in contacts)
    assert m.body("pot").mass[0] == pytest.approx(0.396, abs=1e-8)


def test_settled_pot_floor_does_not_sink_through_hotplate(adapter):
    from cap_harness.realman.platform import is_machine_support_geom

    adapter._advance(2.0)
    d, m = adapter.data, adapter.model
    contacts = [
        c
        for c in d.contact[: d.ncon]
        if any(
            m.geom(a).name == "pot_bottom" and is_machine_support_geom(m.geom(b).name)
            for a, b in [(c.geom1, c.geom2), (c.geom2, c.geom1)]
        )
    ]
    assert contacts
    assert any(
        "hotplate" in {m.geom(c.geom1).name, m.geom(c.geom2).name} for c in contacts
    ), "the corrected disk must actually support the seated pot floor"
    assert max(-c.dist for c in contacts) < 0.0001
    assert d.body("pot").xmat[8] > np.cos(np.deg2rad(1))


def test_handle_target_follows_pot_yaw_instead_of_nominal_wrist(adapter):
    from scipy.spatial.transform import Rotation

    from cap_harness.realman.grasp_geometry import pot_handle_grasp_pose

    m, d = adapter.model, adapter.data
    q = m.joint("pot_free").qposadr[0]
    quat = Rotation.from_euler("z", 70, degrees=True).as_quat()
    d.qpos[q + 3 : q + 7] = quat[[3, 0, 1, 2]]
    mujoco.mj_forward(m, d)
    target = pot_handle_grasp_pose(d)
    np.testing.assert_allclose(
        target[:3, 0], [np.cos(np.deg2rad(70)), np.sin(np.deg2rad(70)), 0], atol=1e-8
    )
    np.testing.assert_allclose(target[:3, 3], d.site("pot_handle_grasp").xpos)
    np.testing.assert_allclose(target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-8)
    assert np.linalg.det(target[:3, :3]) == pytest.approx(1)
