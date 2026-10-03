import math
import os
import torch
import numpy as np

from rl.policy import GaussianPolicy

# Absolute path to the directory where test_policy.py lives
script_dir = os.path.dirname(os.path.abspath(__file__))

# Go up one level to project root, then down into envs/arm2.xml
xml_path = os.path.abspath(os.path.join(script_dir, "..", "envs", "arm2.xml"))

def test_gaussian_log_prob():
    seed = 42
    torch.manual_seed(seed)

    obs_dim, act_dim = 10, 2

    # Instantiate policy & environment
    policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim)
    obs = torch.randn(5, obs_dim)

    # 2. Get action and log_prob from PyTorch distribution
    action, pytorch_log_prob = policy.get_action(obs=obs)

    # 3. Compute log_prob manually using analytical formula
    mean = policy.net(obs)
    std = torch.exp(policy.log_std)

    normalized_sq_err = torch.pow((action - mean) / std, 2)
    log_variance = 2 * policy.log_std  # 2 * log(std)
    log_2pi = math.log(2 * math.pi)

    manual_log_prob = -0.5 * torch.sum(
        normalized_sq_err + log_variance + log_2pi, 
        dim=-1
    )

    # 4. Check if PyTorch matches analytical formula
    assert torch.allclose(pytorch_log_prob, manual_log_prob, atol=1e-6), \
        f"Mismatch!\nPyTorch: {pytorch_log_prob}\nManual:  {manual_log_prob}"

    print("✅ Success! PyTorch dist.log_prob matches analytical Gaussian formula exactly.")

if __name__ == "__main__":
    test_gaussian_log_prob()