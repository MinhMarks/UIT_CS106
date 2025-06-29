import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.fc(x)


class DQNAgentPlus:
    def __init__(self, state_size, action_size, gamma=0.9975, lr=1e-4, batch_size=64,
                 memory_size=10000, path=None, use_cuda=None, tau=0.01, size_update= 1000):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.tau = tau  # Soft update factor
        self.size_update = size_update  # Số lần cập nhật target model
        # Device
        if use_cuda is True:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise RuntimeError("GPU requested but not available.")
        elif use_cuda is False:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"✅ Using device: {self.device}")

        # Main model and target model
        self.model = QNetwork(state_size, action_size).to(self.device)
        self.target_model = QNetwork(state_size, action_size).to(self.device)
        self.target_model.load_state_dict(self.model.state_dict())  # Init as copy

        if path is not None:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            self.target_model.load_state_dict(self.model.state_dict())
            print(f"📂 Loaded model from: {path}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.memory = deque(maxlen=memory_size)

        # Epsilon-greedy
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self):
        if len(self.memory) < self.size_update:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze()

        # ❌ Không dùng model ở đây nữa
        # ✅ Dùng target_model để tính next Q-values
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0]

        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.update_target_model()  # <== cập nhật target model

        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target_model(self):
        # Soft update (tau=1.0 -> hard update)
        for target_param, local_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"💾 Saved model to {path}")
