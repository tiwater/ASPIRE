import itertools

import mujoco as mj
import numpy as np

from cap_harness.realman.clearance_sampling import DistancePairs, body_pair_lever_bound


def test_broadphase_matches_exhaustive_random_near_and_far_mesh_and_primitives():
    model = mj.MjModel.from_xml_string("""<mujoco>
      <asset><mesh name="tetra" vertex="0 0 0 .2 0 0 0 .2 0 0 0 .2"/></asset>
      <worldbody>
        <geom type="mesh" mesh="tetra"/>
        <geom type="box" size=".1 .2 .05"/>
        <geom type="sphere" size=".08"/>
        <geom type="cylinder" size=".06 .15"/>
        <geom type="capsule" size=".04 .12"/>
      </worldbody></mujoco>""")
    data = mj.MjData(model)
    pairs = list(itertools.combinations(range(model.ngeom), 2))
    query = DistancePairs(model, pairs)
    rng = np.random.default_rng(42)
    for scale in [0.05, 0.2, 0.5, 5]:
        for _ in range(20):
            model.geom_pos[:] = rng.uniform(-scale, scale, model.geom_pos.shape)
            quat = rng.normal(size=model.geom_quat.shape)
            model.geom_quat[:] = quat / np.linalg.norm(quat, axis=1, keepdims=True)
            mj.mj_forward(model, data)
            exact = min(float(mj.mj_geomDistance(model, data, a, b, 0.15, None)) for a, b in pairs)
            actual, _ = query.minimum(model, data)
            np.testing.assert_allclose(actual, exact, atol=1e-12)


def test_infinite_plane_is_never_culled_by_finite_sphere_rule():
    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom type="plane" size="1 1 .1"/>
      <geom type="sphere" pos="100 0 .02" size=".05"/>
    </worldbody></mujoco>""")
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    value, pair = DistancePairs(model, [(0, 1)]).minimum(model, data)
    assert pair == (0, 1)
    np.testing.assert_allclose(value, -0.03, atol=1e-12)


def test_body_bound_cancels_common_rotations_and_bounds_finite_changes():
    model = mj.MjModel.from_xml_string("""<mujoco><worldbody>
      <geom name="ground_part" type="sphere" size=".1" pos=".3 .4 .2"/>
      <body pos="0 0 .3"><joint name="body0" axis="0 0 1"/>
        <geom name="trunk" type="box" size=".1 .2 .1"/>
        <body pos="0 0 .5"><joint name="body1" axis="0 1 0"/>
          <geom name="upper" type="sphere" size=".1" pos=".2 0 0"/>
          <body pos=".5 0 0"><joint name="arm" axis="1 0 0"/>
            <geom name="tip" type="box" size=".1 .05 .03" pos=".2 0 0"/>
          </body>
        </body>
      </body></worldbody></mujoco>""")
    data = mj.MjData(model)
    names = ["body0", "body1"]
    tip = model.geom("tip").id
    upper = model.geom("upper").id
    assert body_pair_lever_bound(model, [(tip, upper)], names)["maximum_pair_lever_m"] == 0
    pairs = [(tip, model.geom(n).id) for n in ["ground_part", "trunk", "upper"]]
    bound = body_pair_lever_bound(model, pairs, names)["maximum_pair_lever_m"]
    assert 0 < bound < 2
    rng = np.random.default_rng(3)
    for _ in range(100):
        data.qpos[:] = [0, 0, rng.uniform(-3, 3)]
        mj.mj_forward(model, data)
        old = [mj.mj_geomDistance(model, data, a, b, 10, None) for a, b in pairs]
        angles = rng.uniform(-0.2, 0.2, 2)
        data.qpos[:2] = angles
        mj.mj_forward(model, data)
        new = [mj.mj_geomDistance(model, data, a, b, 10, None) for a, b in pairs]
        assert np.max(np.abs(np.asarray(new) - old)) <= bound * np.abs(angles).sum() + 1e-10
        # Both descendants of both body rotations: exact cancellation.
        np.testing.assert_allclose(new[-1], old[-1], atol=1e-9)
