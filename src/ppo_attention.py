import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict

class MultiHeadAttention(nn.Module):
    def __init__(self, state_dim: int, n_heads: int = 4):
        super().__init__()
        self.state_dim = state_dim
        self.n_heads = n_heads
        self.head_dim = state_dim // n_heads
        
        self.q_linear = nn.Linear(state_dim, state_dim)
        self.k_linear = nn.Linear(state_dim, state_dim)
        self.v_linear = nn.Linear(state_dim, state_dim)
        self.projection = nn.Linear(state_dim, state_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        
        q = self.q_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attention_weights = F.softmax(scores, dim=-1)
        
        
        attended = torch.matmul(attention_weights, v)
        
        
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.state_dim)
        return self.projection(attended)

class AttentionBlock(nn.Module):
    def __init__(self, state_dim: int, n_heads: int = 4):
        super().__init__()
        self.attention = MultiHeadAttention(state_dim, n_heads)
        self.norm1 = nn.LayerNorm(state_dim)
        self.ff = nn.Sequential(
            nn.Linear(state_dim, state_dim * 4),
            nn.ReLU(),
            nn.Linear(state_dim * 4, state_dim)
        )
        self.norm2 = nn.LayerNorm(state_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended = self.attention(x)
        x = self.norm1(x + attended)
        ff_output = self.ff(x)
        return self.norm2(x + ff_output)

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, n_attention_blocks: int = 3):
        super().__init__()
        
        
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        
        
        self.attention_blocks = nn.ModuleList([
            AttentionBlock(128) for _ in range(n_attention_blocks)
        ])
        
        
        self.actor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
        
        
        self.critic = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(state)
        
        
        for block in self.attention_blocks:
            x = block(x)
            
        
        action_probs = F.softmax(self.actor(x), dim=-1)
        value = self.critic(x)
        
        return action_probs, value

class PPOAgent_attention:
    def __init__(
        self,
        state_dim: int = 6,
        action_dim: int =4,
        lr: float = 3e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        c1: float = 1.0,
        c2: float = 0.01
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=lr)
        
        self.gamma = gamma
        self.epsilon = epsilon
        self.c1 = c1
        self.c2 = c2
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_probs, value = self.actor_critic(state)
            
        action_dist = torch.distributions.Categorical(action_probs)
        action = action_dist.sample()
        log_prob = action_dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def train(
        self,
        env,
        max_episodes: int = 1000,
        max_steps: int = 200,
        batch_size: int = 64
    ):
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join("models", f"ppo_agent_{timestamp}")
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            
        
        episode_rewards_all = []
        episode_lengths = []
        
        for episode in range(max_episodes):
            state, _ = env.reset()
            episode_reward = 0
            
            
            states, actions, rewards = [], [], []
            values, log_probs = [], []
            
            for step in range(max_steps):
                action, log_prob, value = self.select_action(state)
                
                next_state, reward, done, _, _ = env.step(action)
                
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                values.append(value)
                log_probs.append(log_prob)
                
                state = next_state
                episode_reward += reward
                
                if done:
                    break
            
            
            advantages = self._compute_advantages(rewards, values)
            
            
            self._update_network(
                states, actions, log_probs,
                advantages, values, batch_size
            )
            
            
            episode_rewards_all.append(episode_reward)
            episode_lengths.append(step + 1)
            
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(episode_rewards_all[-10:])
                print(f"Episode {episode+1}, Avg Reward: {avg_reward:.2f}")
                save_name = f"model_ep{episode+1}_reward{int(episode_reward)}.pth"
                save_path = os.path.join(base_path, "models")
                self.save(save_path,save_name)
                
                
                plt.figure(figsize=(10, 6))
                plt.plot(episode_rewards_all, label='Episode Return')
                plt.axhline(y=3432807, color='r', linestyle='--', label='Seuil 3432807')
                plt.axhline(y=1e8, color='g', linestyle='--', label='Seuil 1e8')
                plt.yscale('log')
                plt.xlabel('Episode')
                plt.ylabel('Cumulative Reward')
                plt.title(f'Training Progress - Episode {episode+1}')
                plt.legend()
                plt.grid(True)
                
                plot_path = os.path.join(base_path, f"progress_{timestamp}.png")
                plt.savefig(plot_path)
                plt.close()
    
    def _compute_advantages(
        self,
        rewards: List[float],
        values: List[float]
    ) -> torch.Tensor:
        advantages = []
        returns = []
        next_return = 0
        
        for r, v in zip(reversed(rewards), reversed(values)):
            next_return = r + self.gamma * next_return
            returns.insert(0, next_return)
            advantages.insert(0, next_return - v)
            
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages
    
    def _update_network(
        self,
        states: List[np.ndarray],
        actions: List[int],
        old_log_probs: List[float],
        advantages: torch.Tensor,
        values: List[float],
        batch_size: int
    ):
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        
        for _ in range(10): 
            for idx in range(0, len(states), batch_size):
                batch_states = states[idx:idx + batch_size]
                batch_actions = actions[idx:idx + batch_size]
                batch_log_probs = old_log_probs[idx:idx + batch_size]
                batch_advantages = advantages[idx:idx + batch_size]
                batch_values = values[idx:idx + batch_size]
                
                action_probs, current_values = self.actor_critic(batch_states)
                print(current_values.shape)
                dist = torch.distributions.Categorical(action_probs)
                current_log_probs = dist.log_prob(batch_actions)
                
                
                ratios = torch.exp(current_log_probs - batch_log_probs)
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                
                value_loss = F.mse_loss(current_values.squeeze(-1), batch_values)
                
                
                entropy_loss = -dist.entropy().mean()
                
                
                loss = policy_loss + self.c1 * value_loss + self.c2 * entropy_loss
                
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
    
    def save(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, os.path.join(path, "best_model"))
    
    def load(self):
        path = '/Users/laura/Documents/MVA/RL/mva-rl-assignment-lauraachoquett-main/models/ppo_agent_20250114_120223/models/best_model.pth'
        checkpoint = torch.load(path)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])