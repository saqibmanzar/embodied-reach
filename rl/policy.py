import torch
import torch.nn as nn
from torch.distributions import Normal

"""
Since we are dealing with Gaussian Policy. so policy needs to be updated for us get the maximum objective. 
Being Gaussian we need to update mean and std. and since NN will output mean in this case so parameter which 
can be updated are: $\theta = \{W, b, \text{log\_std}\}$
"""

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, self.act_dim)
        )

        self.log_std = nn.Parameter(torch.full((self.act_dim,), -0.5))

    def get_distribution(self, obs: torch.Tensor):
        mean = self.net(obs)
        std = torch.exp(self.log_std)
        return Normal(loc=mean, scale=std)

    def get_action(self, obs: torch.Tensor):
        dist = self.get_distribution(obs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob
    