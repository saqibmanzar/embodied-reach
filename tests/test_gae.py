import torch
from rl.gae import compute_gae

def test_compute_gae_hand_calculation():
    gamma = 0.99
    lam = 0.95

    rewards = torch.tensor([[1.0], [2.0], [10.0]], dtype=torch.float32)
    values = torch.tensor([[0.5], [1.5], [8.0]], dtype=torch.float32)
    dones = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float32)
    last_value = torch.tensor([5.0], dtype=torch.float32)

    # Hand-calculated expected values
    expected_adv = torch.tensor([
        [2.45525],
        [0.50000],
        [6.95000]
    ])
    expected_returns = expected_adv + values

    adv, returns = compute_gae(rewards, values, dones, last_value, gamma=gamma, lam=lam)

    print("\nComputed Advantages:\n", adv)
    print("Expected Advantages:\n", expected_adv)

    assert torch.allclose(adv, expected_adv, atol=1e-4)
    assert torch.allclose(returns, expected_returns, atol=1e-4)


test_compute_gae_hand_calculation()