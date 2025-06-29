import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class ActorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(ActorNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.fc(x)

class CriticNetwork(nn.Module):
    def __init__(self, state_size):
        super(CriticNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.fc(x)

class ActorCriticAgent:
    def __init__(self, state_size, action_size, gamma=0.9975, actor_lr=1e-4, critic_lr=1e-4,
                actor_path = None, critic_path = None):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        if actor_path is None:
            self.actor = ActorNetwork(state_size, action_size).to(self.device)
            self.critic = CriticNetwork(state_size).to(self.device)
        else :
            self.actor = ActorNetwork(state_size, action_size).to(self.device)
            self.critic = CriticNetwork(state_size).to(self.device)
            self.actor.load_state_dict(torch.load(actor_path))
            self.critic.load_state_dict(torch.load(critic_path))

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Gradient accumulation
        self.saved_log_probs = []
        self.saved_values = []
        self.saved_rewards = []

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        probs = self.actor(state)
        value = self.critic(state)

        dist = torch.distributions.Categorical(probs)
        action = dist.sample()

        self.saved_log_probs.append(dist.log_prob(action))
        self.saved_values.append(value)
        return action.item()

    def remember(self, reward):
        self.saved_rewards.append(reward)

    def replay(self):

        # Compute returns and advantages
        returns = []
        discounted_sum = 0
        for reward in reversed(self.saved_rewards):
            discounted_sum = reward + self.gamma * discounted_sum
            returns.insert(0, discounted_sum)
        
        returns = torch.FloatTensor(returns).to(self.device)

        # Compute losses
        actor_loss = []
        critic_loss = []
        for log_prob, value, ret in zip(self.saved_log_probs, self.saved_values, returns):
            value = value.squeeze()
            advantage = ret - value
            actor_loss.append(-log_prob * advantage.detach())
            critic_loss.append(nn.MSELoss()(value, ret))

        actor_loss = torch.stack(actor_loss).sum()
        critic_loss = torch.stack(critic_loss).sum()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Clear saved data
        self.saved_log_probs = []
        self.saved_values = []
        self.saved_rewards = []
    
    def save_model(self, actor_path, critic_path):
        """
        Save the actor and critic model weights to the specified file paths.
        
        Args:
            actor_path (str): File path to save the actor model weights.
            critic_path (str): File path to save the critic model weights.
        """
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)