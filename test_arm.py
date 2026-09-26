import mujoco
import numpy as np


model = mujoco.MjModel.from_xml_path("envs/arm2.xml")
data = mujoco.MjData(model)

data.qpos[0] = np.deg2rad(45)
data.qpos[1] = np.deg2rad(45)

mujoco.mj_forward(model, data)

ee_pos = data.site('ee').xpos

print("End-effector position:", ee_pos)