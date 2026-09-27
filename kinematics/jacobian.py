import mujoco
import numpy as np


def jacobian_2R(q, L1, L2):
  q1, q2 = q
  s1 = np.sin(q1)
  s12 = np.sin(q1 + q2)
  c1 = np.cos(q1)
  c12 = np.cos(q1 + q2)

  return np.array(
      [[-L1 * s1 - L2 * s12, -L2 * s12], [L1 * c1 + L2 * c12, L2 * c12]]
  )


def test_jacobian():
  L1, L2 = 0.5, 0.5
  model = mujoco.MjModel.from_xml_path("envs/arm2.xml")
  data = mujoco.MjData(model)

  data.qpos[0] = np.deg2rad(45)
  data.qpos[1] = np.deg2rad(45)
  mujoco.mj_forward(model, data)

  site_id = data.site("ee").id
  jacp = np.zeros((3, model.nv))
  mujoco.mj_jacSite(model, data, jacp, None, site_id)

  J_mj = jacp[:2, :2]
  J_ana = jacobian_2R(data.qpos[:2], L1, L2)

  assert np.allclose(
      J_mj, J_ana, atol=1e-5
  ), f"Jacobian mismatch!\nMuJoCo:\n{J_mj}\nAnalytic:\n{J_ana}"
  print("Jacobian test passed successfully!")


if __name__ == "__main__":
  test_jacobian()