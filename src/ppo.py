import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
from datetime import datetime
import matplotlib.pyplot as plt
from utils import PrioritizedBuffer

class HybridActorCritic(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super().__init__()
        self.state_dim = state_dim
        
        # Encoder commun
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),  # Augmenter la taille
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )
        
        # Calcul des ratios (toutes les combinaisons possibles)
        self.n_ratios = (state_dim * (state_dim - 1)) // 2
        
        # Actor network avec attention sur les ratios
        self.actor = nn.Sequential(
            nn.Linear(64 + self.n_ratios, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def compute_ratios(self, state):
        ratios = []
        for i in range(self.state_dim):
            for j in range(i+1, self.state_dim):
                ratio = state[:, i] / (state[:, j] + 1e-6)
                ratios.append(torch.log1p(ratio))
        return torch.stack(ratios, dim=1)
    
    def forward(self, state):
        encoded = self.encoder(state)
        ratios = self.compute_ratios(state)
        
        actor_input = torch.cat([encoded, ratios], dim=1)
        action_probs = torch.softmax(self.actor(actor_input), dim=-1)
        value = self.critic(encoded)
        
        return action_probs, value

class PPOAgent:
    def __init__(self, config, state_dim=6, action_dim=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_critic = HybridActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config.get('learning_rate', 3e-4))
        
        # Configuration
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.batch_size = config.get('batch_size', 64)
        self.gamma = config.get('gamma', 0.99)
        
        # Buffer pour stocker les transitions
        self.buffer = []

        self.prioritized_buffer = PrioritizedBuffer(
            capacity=10000,
            alpha=0.6,  # Contrôle l'importance de la priorité
            beta=0.4    # Commence bas et augmente vers 1
        )
        
    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, value = self.actor_critic(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        
        log_prob = dist.log_prob(action)
        
        return action.item(), value.item(), log_prob.item()
    
    def act(self, observation):
        action, _, _ = self.select_action(observation)
        return action

            
    def store_transition(self, state, action, reward, next_state, done, value, log_prob):
        # Calcul de l'erreur TD
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            _, next_value = self.actor_critic(next_state_tensor)
            next_value = next_value.item()
            td_error = reward + self.gamma * next_value * (1-done) - value
            
        transition = {
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'value': value,
            'log_prob': log_prob
        }
        
        self.prioritized_buffer.add(transition, td_error)
    
    def update(self):
        if self.prioritized_buffer.size < self.batch_size:
            return
            
        # Échantillonnage prioritaire
        batch, indices, weights = self.prioritized_buffer.sample(self.batch_size)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Préparation des données
        states = torch.FloatTensor([t['state'] for t in batch]).to(self.device)
        actions = torch.LongTensor([t['action'] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t['next_state'] for t in batch]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in batch]).to(self.device)
        old_values = torch.FloatTensor([t['value'] for t in batch]).to(self.device)
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in batch]).to(self.device)
        
        # Calcul des returns et advantage avec les poids d'importance
        with torch.no_grad():
            _, next_values = self.actor_critic(next_states)
            next_values = next_values.squeeze()
            returns = rewards + self.gamma * next_values * (1 - dones)
            advantages = (returns - old_values) * weights
            
            # Normalisation des avantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Optimisation PPO
        for _ in range(10):
            action_probs, values = self.actor_critic(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            values = values.squeeze()
            
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            
            # Calcul des nouvelles erreurs TD pour mise à jour des priorités
            td_errors = (returns - values).detach().cpu().numpy()
            self.prioritized_buffer.update_priorities(indices, td_errors)
            
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values, returns)
            entropy_loss = -dist.entropy().mean()
            
            loss = actor_loss + self.value_coef * critic_loss + self.entropy_coef * entropy_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
    
    def train(self, env, max_episode):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join("models", f"ppo_agent_{timestamp}")
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            
        print("Beginning of the training with curriculum")
        episode_rewards = []
        episode_lengths = []
        V0_list = []
        discounted_rewards = []
        action_list = np.zeros(4)
        
        # Paramètres du curriculum
        initial_max_steps = 50
        final_max_steps = 200
        curriculum_length = max_episode // 4  # Durée du curriculum
        
        def get_max_steps(episode):
            if episode < curriculum_length:
                # Augmentation progressive linéaire
                progress = episode / curriculum_length
                return int(initial_max_steps + (final_max_steps - initial_max_steps) * progress)
            return final_max_steps
        
        for episode in range(max_episode):
            state, _ = env.reset()
            episode_reward = 0
            step = 0
            V0 = None
            
            max_steps_current = get_max_steps(episode)
            while step < max_steps_current:
                action, value, log_prob = self.select_action(state)
                if V0 is None:
                    V0 = value
                
                next_state, reward, done, trunc, _ = env.step(action)
                
                self.store_transition(state, action, reward, next_state, done, value, log_prob)
                action_list[action] += 1
                episode_reward += reward
                step += 1
                
                if len(self.buffer) >= self.batch_size:
                    self.update()
                
                if done or trunc:
                    break
                    
                state = next_state
            
            # Calcul de la récompense actualisée
            discounted_reward = sum([r['reward'] * (self.gamma ** i) for i, r in enumerate(self.buffer[-step:])])
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            V0_list.append(V0)
            discounted_rewards.append(discounted_reward)
            
            print(f"Episode {episode:3d}, "
                  f"episode return {episode_reward:4.1f}, "
                  f"length {step:3d}/{max_steps_current}, "
                  f"V0 {V0:4.1f}, "
                  f"action distribution {action_list}")
            
            # Sauvegarde périodique tous les 10 épisodes
            if (episode + 1) % 10 == 0:
                save_name = f"model_ep{episode+1}_reward{int(episode_reward)}.pth"
                save_path = os.path.join(base_path, save_name)
                self.save(save_path)
                
                # Création et sauvegarde du graphique
                plt.figure(figsize=(10, 6))
                plt.plot(episode_rewards, label='Episode Return')
                plt.axhline(y=3432807, color='r', linestyle='--', label='Seuil 3432807')
                plt.axhline(y=1e8, color='g', linestyle='--', label='Seuil 1e8')
                plt.yscale('log')
                plt.xlabel('Episode')
                plt.ylabel('Cumulative Reward')
                plt.title(f'Training Progress - Episode {episode+1}')
                plt.legend()
                plt.grid(True)
                
                plot_path = os.path.join(base_path, "progress.png")
                plt.savefig(plot_path)
                plt.close()
                
            # Réinitialisation du compteur d'actions
            action_list = np.zeros(4)
        
        return [episode_lengths, discounted_rewards, episode_rewards, V0_list]
    
    def save(self, path):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, extension = os.path.splitext(path)
        path_with_timestamp = f"{filename}_{timestamp}{extension}"
        
        state_dict = {
            'config': {
                'clip_epsilon': self.clip_epsilon,
                'value_coef': self.value_coef,
                'entropy_coef': self.entropy_coef,
                'max_grad_norm': self.max_grad_norm,
                'batch_size': self.batch_size,
                'gamma': self.gamma
            },
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        
        torch.save(state_dict, path_with_timestamp)
        
    def load(self):
        path = '/Users/laura/Documents/MVA/RL/mva-rl-assignment-lauraachoquett-main/models/ppo_agent_20250114_120223/models/best_model.pth'
        try:
            checkpoint = torch.load(path)
            
            # Restauration des paramètres de configuration
            config = checkpoint['config']
            self.clip_epsilon = config['clip_epsilon']
            self.value_coef = config['value_coef']
            self.entropy_coef = config['entropy_coef']
            self.max_grad_norm = config['max_grad_norm']
            self.batch_size = config['batch_size']
            self.gamma = config['gamma']
            
            # Restauration des poids et de l'optimizer
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"Model successfully loaded from {path}")
            
        except FileNotFoundError:
            print(f"File {path} not found")
        except Exception as e:
            print(f"Error loading model: {str(e)}")