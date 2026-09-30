import numpy as np
from envs.reach_env import PlanarReachEnv
from gymnasium.utils.env_checker import check_env

# 1. Verify environment compliance
env = PlanarReachEnv(target_mode="random")
check_env(env)
print("Gymnasium env_checker passed!")

# 2. Run 100-episode random baseline
returns = []
successes = []

for ep in range(100):
    obs, info = env.reset(seed=ep)
    ep_return = 0.0
    ep_success = False

    for _ in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        ep_return += reward
        
        if info["is_success"]:
            ep_success = True

    returns.append(ep_return)
    successes.append(ep_success)

print(f"Random Baseline (100 episodes):")
print(f"Success Rate : {np.mean(successes) * 100:.1f}%")
print(f"Mean Return  : {np.mean(returns):.2f} +/- {np.std(returns):.2f}")