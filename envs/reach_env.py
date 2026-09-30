import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco

class PlanarReachEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 100}

    def __init__(
            self,
            target_mode: str = "random",
            frame_skip: int = 5,
            max_steps: int = 200,
            link_lengths: tuple = (0.5, 0.5),
            model_path: str = "envs/arm2.xml",
        ):
        super().__init__()
        self.target_mode = target_mode
        self.frame_skip = frame_skip
        self.max_steps = max_steps
        self.l1, self.l2 = link_lengths
        self.current_step = 0
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(
            low = -1.0,
            high = 1.0,
            shape = (2,),
            dtype = np.float32
        )

        high = np.array([
            1.0, 1.0, 1.0, 1.0,      # cos(q1), cos(q2), sin(q1), sin(q2) in [-1, 1]
            50.0, 50.0,              # Joint velocity bounds (rad/s)
            1.0, 1.0,                # Target position bounds (m)
            2.0, 2.0                 # Relative distance dx, dy bounds (m)
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-high,
            high=high,
            dtype=np.float32
        )

    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)

        qpos = self.data.qpos[:2]
        qvel = self.data.qvel[:2]

        x_current = self.data.site('ee').xpos[:2]
        x_target = self.data.site('target').xpos[:2]

        dx_dy = x_current - x_target

        return np.array([
            np.cos(qpos[0]), 
            np.cos(qpos[1]), 
            np.sin(qpos[0]),  
            np.sin(qpos[1]),  
            qvel[0],
            qvel[1], 
            x_target[0],  
            x_target[1],  
            dx_dy[0],  
            dx_dy[1],  
        ], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data) # resets the simulator's state between episodes

        self.current_step = 0

        qpos = self.np_random.uniform(low = -np.pi, high = np.pi, size = (2,))
        qvel = np.zeros(2)

        self.data.qpos[:2] = qpos
        self.data.qvel[:2] = qvel

        if self.target_mode == "fixed":
            target_xy = np.array([0.4, 0.4])
        else:
            r = self.np_random.uniform(low = 0.2, high = self.l1 + self.l2)
            theta = self.np_random.uniform(low=-np.pi, high=np.pi)
            target_xy = np.array([
                r * np.cos(theta),
                r * np.sin(theta)
            ])
        
        self.data.mocap_pos[0][:2] = target_xy
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.data.ctrl[:2] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.current_step += 1

        obs = self._get_obs()
        ee_pos = self.data.site('ee').xpos[:2]
        target_pos = self.data.site('target').xpos[:2]

        dist = np.linalg.norm(ee_pos - target_pos)
        reward_dist = -dist
        reward_ctrl = -0.01 * (np.linalg.norm(action) ** 2)
        reward = float(reward_dist + reward_ctrl)

        is_success = bool(dist < 0.02)

        terminated = False
        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "is_success": is_success,
            "distance": dist
        }

        return obs, reward, terminated, truncated, info

