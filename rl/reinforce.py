import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Import your environment and policy classes
from envs.reach_env import PlanarReachEnv
from rl.policy import GaussianPolicy


def compute_reward_to_go(rewards: list[float], gamma: float = 0.99) -> torch.Tensor:
    discounted_rewards = []
    running_g = 0.0
    for r in reversed(rewards):
        running_g = r + gamma * running_g
        discounted_rewards.append(running_g)

    discounted_rewards = discounted_rewards[::-1]
    return torch.tensor(discounted_rewards, dtype=torch.float32)


def compute_loss(log_prob: torch.Tensor, return_to_go: torch.Tensor):
    return -(log_prob * return_to_go).mean()


def collect_batch(env, policy, num_episodes: int = 16, gamma: float = 0.99):
    batch_log_probs = []
    batch_returns = []
    episode_returns = []
    episode_successes = []

    for _ in range(num_episodes):
        obs_np, _ = env.reset()
        done = False
        ep_log_probs = []
        ep_rewards = []
        ep_return = 0.0
        ep_success = False

        while not done:
            obs_tensor = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            action_tensor, log_prob = policy.get_action(obs_tensor)

            action_np = action_tensor.squeeze(0).detach().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_np)
            done = terminated or truncated

            if info.get("is_success", False):
                ep_success = True

            ep_log_probs.append(log_prob.squeeze(0))
            ep_rewards.append(reward)
            ep_return += reward
            obs_np = next_obs

        g_t = compute_reward_to_go(ep_rewards, gamma=gamma)

        batch_log_probs.extend(ep_log_probs)
        batch_returns.append(g_t)
        episode_returns.append(ep_return)
        episode_successes.append(1.0 if ep_success else 0.0)

    batch_log_probs = torch.stack(batch_log_probs)
    batch_returns = torch.cat(batch_returns)

    # Return normalization for batch gradient stability
    batch_returns = (batch_returns - batch_returns.mean()) / (batch_returns.std() + 1e-8)

    avg_return = np.mean(episode_returns)
    success_rate = np.mean(episode_successes)

    return batch_log_probs, batch_returns, avg_return, success_rate


def train_seed(seed: int, num_iterations: int = 300, num_episodes: int = 16, gamma: float = 0.99):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = PlanarReachEnv(target_mode="fixed")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = GaussianPolicy(obs_dim=obs_dim, act_dim=act_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)

    writer = SummaryWriter(log_dir=f"runs/reinforce_seed_{seed}")

    print(f"\n=== Starting Training for Seed {seed} ===")

    for iteration in range(1, num_iterations + 1):
        batch_log_probs, batch_returns, avg_return, success_rate = collect_batch(
            env, policy, num_episodes=num_episodes, gamma=gamma
        )

        loss = compute_loss(batch_log_probs, batch_returns)

        optimizer.zero_grad()
        loss.backward()

        # Compute gradient norm
        grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

        optimizer.step()

        # Log metrics to TensorBoard
        writer.add_scalar("Train/AverageReturn", avg_return, iteration)
        writer.add_scalar("Train/SuccessRate", success_rate, iteration)
        writer.add_scalar("Train/GradNorm", grad_norm.item(), iteration)
        writer.add_scalar("Train/Loss", loss.item(), iteration)

        if iteration % 20 == 0 or iteration == 1:
            print(
                f"Seed {seed:2d} | Iter {iteration:3d}/{num_iterations} | "
                f"Avg Return: {avg_return:7.2f} | Success Rate: {success_rate * 100:5.1f}% | "
                f"Grad Norm: {grad_norm.item():.4f}"
            )

    writer.close()


def main():
    seeds = [42, 43, 44]
    for seed in seeds:
        train_seed(seed)


if __name__ == "__main__":
    main()