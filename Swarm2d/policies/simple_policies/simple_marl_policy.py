import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from typing import List, Dict

class SimpleMARLPolicy(nn.Module):
    """
    A simple shared policy for a team of agents.
    Each agent in the team uses this same policy network to make decisions.
    The policy is updated based on the collective experience of the team.
    """
    def __init__(self, obs_dim, movement_dim, pickup_dim):
        super(SimpleMARLPolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        self.movement_head = nn.Linear(128, movement_dim)
        self.pickup_head = nn.Linear(128, pickup_dim)
        
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

        # Buffers for a single episode's experience for the whole team
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, obs):
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float32)
        
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        features = self.net(obs)
        movement = torch.tanh(self.movement_head(features))
        pickup_logits = self.pickup_head(features)
        return movement, pickup_logits

    def act(self, team_obs: List[np.ndarray]) -> List[Dict]:
        """
        Determines actions for a whole team of agents.
        
        Args:
            team_obs: A list of observations, one for each agent in the team.

        Returns:
            A list of action dictionaries, one for each agent.
        """
        actions = []
        # Store log probs for this step for all agents
        step_log_probs = []

        for obs in team_obs:
            movement, pickup_logits = self.forward(obs)
            
            pickup_prob = torch.softmax(pickup_logits, dim=-1)
            pickup_dist = Categorical(pickup_prob)
            pickup_action = pickup_dist.sample()

            # Save log probability for this agent's action
            step_log_probs.append(pickup_dist.log_prob(pickup_action))

            actions.append({
                "movement": movement.cpu().detach().numpy().squeeze(),
                "pickup": pickup_action.cpu().detach().numpy().squeeze()
            })
        
        # Append the list of log_probs for this step to the episode buffer
        self.saved_log_probs.append(step_log_probs)

        return actions

    def update(self):
        """
        Updates the policy using the REINFORCE algorithm based on the
        collected experience of the entire team over an episode.
        """
        if not self.saved_log_probs:
            return 0.0

        # The rewards buffer should contain lists of rewards, one per step.
        # We'll use a team-based reward: sum of rewards for all agents at each step.
        team_rewards_per_step = [sum(step_rewards) for step_rewards in self.rewards]

        R = 0
        policy_loss = []
        returns = []
        
        # Calculate discounted team rewards
        for r in reversed(team_rewards_per_step):
            R = r + 0.99 * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        if len(returns) > 1:
             returns = (returns - returns.mean()) / (returns.std() + 1e-6)

        # Calculate policy loss
        for step_log_probs, R in zip(self.saved_log_probs, returns):
            for log_prob in step_log_probs:
                policy_loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        if not policy_loss:
            return 0.0

        policy_loss_tensor = torch.cat(policy_loss).sum()
        policy_loss_tensor.backward()
        self.optimizer.step()

        loss_val = policy_loss_tensor.item()

        # Clear buffers for the next episode
        self.rewards = []
        self.saved_log_probs = []

        return loss_val
