import torch


def compute_gae(
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: torch.Tensor, 
        last_value: torch.Tensor, 
        gamma: float = 0.99, 
        lam: float = 0.95
    ) -> tuple[torch.Tensor, torch.Tensor]:
    T, N = rewards.shape

    advantages = torch.zeros(T, N, dtype=torch.float32, device=rewards.device)
    gae = torch.zeros(N, dtype=torch.float32, device=rewards.device)


    for t in reversed(range(T)):
        if t == T-1:
            next_value = last_value
        else:
            next_value = values[t+1]

        non_terminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]

        gae = delta + gamma * lam * non_terminal * gae

        advantages[t] = gae

    returns = advantages + values
    return advantages, returns





