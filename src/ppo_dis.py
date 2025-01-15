import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import os
from datetime import datetime
import matplotlib.pyplot as plt
from utils import PrioritizedBuffer
from utils import GreedyHeuristic
import pandas as pd
from statistics import mean
from utils import SimpleBuffer
class HybridActorCritic(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super().__init__()
        self.state_dim = state_dim

        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),  
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU()
        )

        
        self.n_ratios = (state_dim * (state_dim - 1)) // 2

        
        self.actor = nn.Sequential(
            nn.Linear(64 + self.n_ratios, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )

        
        self.critic = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
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

class PPOAgent_test:
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
        
        
        self.buffer = []

        self.prioritized_buffer = SimpleBuffer(
            capacity=10000,
        )
        self.epsilon = 0.1

        self.epsilon_min = 0.01
        self.epsilon_max = 1
        self.epsilon_decay_period = 41000
        self.epsilon_delay_decay =5000
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_decay_period
        self.step_count=1

        self.heuristic = GreedyHeuristic()
        self.heuristic_prob_beginning = 0.30
        self.heuristic_prob_end= 0.01
        self.heuristic_prob =  self.heuristic_prob_beginning


    def initialize_buffer_with_heuristic(self, env, n_episodes=60,save_dir="output"):
        """Initialise le buffer avec un mélange de trajectoires heuristiques et aléatoires"""
        print("Initializing buffer with heuristic and random trajectories...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        heuristic_rewards = []
        random_rewards = []
        heuristic_actions = []
        heuristic_states = []
        
        for episode in range(n_episodes):
            state, _ = env.reset()

            done = False
            episode_reward = 0
            episode_actions = []  
            episode_states = []   
            is_heuristic_episode = np.random.random() < 0.4
            
            while not done:
                if is_heuristic_episode:
                    action = self.heuristic.select_action(state)
                    episode_actions.append(action)
                    episode_states.append(state)
                else:
                    action = np.random.randint(4)
                
                next_state, reward, done, trunc, _ = env.step(action)
                episode_reward += reward
                
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, value = self.actor_critic(state_tensor)
                self.prioritized_buffer.add(
                    {
                        'state': state,
                        'action': action,
                        'reward': reward,
                        'next_state': next_state,
                        'done': done or trunc,
                        'value': value.item(),
                        'log_prob': 0
                    }
                )
                
                state = next_state
                if trunc:
                    break
            
            if is_heuristic_episode:
                heuristic_rewards.append(episode_reward)
                heuristic_actions.append(episode_actions)
                heuristic_states.append(episode_states)
            else:
                random_rewards.append(episode_reward)
        
        # Afficher les statistiques de récompense
        if heuristic_rewards:
            avg_heuristic = np.mean(heuristic_rewards)
            std_heuristic = np.std(heuristic_rewards)
            print(f"Heuristic episodes ({len(heuristic_rewards)}): Average reward = {avg_heuristic:.2f} ± {std_heuristic:.2f}")
        
        if random_rewards:
            avg_random = np.mean(random_rewards)
            std_random = np.std(random_rewards)
            print(f"Random episodes ({len(random_rewards)}): Average reward = {avg_random:.2f} ± {std_random:.2f}")
        
 

        if heuristic_actions:
            all_actions = np.concatenate(heuristic_actions)
            action_counts = np.bincount(all_actions, minlength=4)

            plt.subplot(1, 2, 2)
            plt.bar(range(4), action_counts, color=['grey', 'blue', 'green', 'red'])
            plt.title('Heuristic Action Distribution')
            plt.xlabel('Action')
            plt.ylabel('Count')
            plt.xticks(range(4), ['No Drug', 'RTI', 'PI', 'Both'])

        plt.tight_layout()
        plt.show()

        
        if heuristic_states:
            for episode_idx, states in enumerate(heuristic_states[:1]):  # Limité au premier épisode pour cet exemple
                df = pd.DataFrame(states, columns=['T1', 'T1star', 'T2', 'T2star', 'V', 'E'])
                df.to_csv(f"{save_dir}/heuristic_episode_{episode_idx}_states.csv", index=False)
                print(f"Saved states of heuristic episode {episode_idx} to {save_dir}/heuristic_episode_{episode_idx}_states.csv")
                
                
                for column in df.columns:
                    plt.figure()
                    plt.plot(df[column])
                    plt.title(f"Evolution of {column} in episode {episode_idx}")
                    plt.xlabel("Time step")
                    plt.ylabel(column)
                    plt.savefig(f"{save_dir}/heuristic_episode_{episode_idx}_{column}.png")
                    print(f"Saved plot for {column} of heuristic episode {episode_idx} to {save_dir}/heuristic_episode_{episode_idx}_{column}.png")
                    plt.close()
        
        print(f"\nBuffer initialized with {self.prioritized_buffer.size} transitions")
        
        return {
            'heuristic_rewards': heuristic_rewards,
            'random_rewards': random_rewards,
            'heuristic_actions': heuristic_actions,
            'heuristic_states': heuristic_states}

    def act(self,state) : 
        state = np.sign(state)*np.log(1+np.abs(state))
        if np.random.random() < self.heuristic_prob:
            action = self.heuristic.select_action(state)
            
        if np.random.random() < self.epsilon:
            action = np.random.randint(4)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, value = self.actor_critic(state_tensor)
        action = action_probs.argmax()
            
        return action.item()
        
    def select_action(self, state):
    
        state = np.sign(state)*np.log(1+np.abs(state))
        if self.step_count > self.epsilon_delay_decay:
                self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)

        if np.random.random() < self.heuristic_prob:
            
            action = self.heuristic.select_action(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, value = self.actor_critic(state_tensor)
            return action, value.item(), 0
        
        
        if np.random.random() < self.epsilon:
            action = np.random.randint(4)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, value = self.actor_critic(state_tensor)
            return action, value.item(), 0
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action_probs, value = self.actor_critic(state_tensor)
        action = action_probs.argmax()
        log_prob = torch.log(action_probs[0, action])
        return action.item(), value.item(), log_prob.item()

    
    def update(self):
        if self.prioritized_buffer.size < self.batch_size:
            return
            
        
        batch = self.prioritized_buffer.sample(self.batch_size)
        
        
        states = torch.FloatTensor([t['state'] for t in batch]).to(self.device)
        actions = torch.LongTensor([t['action'] for t in batch]).to(self.device)
        rewards = torch.FloatTensor([t['reward'] for t in batch]).to(self.device)
        next_states = torch.FloatTensor([t['next_state'] for t in batch]).to(self.device)
        dones = torch.FloatTensor([t['done'] for t in batch]).to(self.device)
        old_values = torch.FloatTensor([t['value'] for t in batch]).to(self.device)
        old_log_probs = torch.FloatTensor([t['log_prob'] for t in batch]).to(self.device)

        states = torch.sign(states) * torch.log(1 + torch.abs(states))
        actions = actions 
        rewards = torch.sign(rewards) * torch.log(1 + torch.abs(rewards))
        next_states = torch.sign(next_states) * torch.log(1 + torch.abs(next_states))
        
        
        with torch.no_grad():
            _, next_values = self.actor_critic(next_states)
            next_values = next_values.squeeze()
            returns = rewards + self.gamma * next_values * (1 - dones)
            advantages = (returns - old_values)
            
            
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        
        for _ in range(10):
            action_probs, values = self.actor_critic(states)
            dist = Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            values = values.squeeze()
            
            ratios = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            
            
            
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
            
        
        episode_rewards = []
        episode_lengths = []
        V0_list = []
        discounted_rewards = []
        action_list = np.zeros(4)
        reward_history=[]
        
        initial_max_steps = 150
        final_max_steps = 200
        curriculum_length = max_episode // 2
        
        def get_max_steps(episode):
            if episode < curriculum_length:
                increment = episode // 5  
                current_length = initial_max_steps + increment
                return min(current_length, final_max_steps)
            return final_max_steps
        
        initial_heuristic_prob = self.heuristic_prob_beginning

        self.heuristic_prob = initial_heuristic_prob
        base_path_graph=os.path.join(base_path,'graphs')
        self.initialize_buffer_with_heuristic(env,save_dir=base_path_graph)

        print("Beginning of the training with curriculum")
        
        for episode in range(max_episode):
            state, _ = env.reset()
            episode_reward = 0
            step = 0
            V0 = None
            episode_transitions = []
            max_steps_current = get_max_steps(episode)
            while step < max_steps_current:
                
                action, value, log_prob = self.select_action(state)
                if V0 is None:
                    V0 = value

                self.heuristic_prob = initial_heuristic_prob * (1 - episode/max_episode) + self.heuristic_prob_end * (episode/max_episode)
                next_state, reward, done, trunc, _ = env.step(action)

                
                transition = {
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'next_state': next_state,
                    'done': done,
                    'value': value,
                    'log_prob': log_prob
                }
                episode_transitions.append(transition)
                
                action_list[action] += 1
                episode_reward += reward
                step += 1
                self.step_count += 1

                
                if done or trunc:
                    break
                    
                state = next_state
            
            # Ajout de l'épisode complet au buffer avec priorités TD
            for transition in episode_transitions:
                self.prioritized_buffer.add(transition)
            
            
            for _ in range(10):
                self.update()
            
            
            discounted_reward = sum(r['reward'] * (self.gamma ** i) for i, r in enumerate(episode_transitions))
            
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            V0_list.append(V0)
            discounted_rewards.append(discounted_reward)
            reward_history.append(episode_reward)
            
            print(f"Episode {episode:3d}, "
                f"episode return {episode_reward:.2e}, "
                f"length {step:3d}/{max_steps_current}, "
                f"V0 {V0:4.1f}, "
                f"action distribution {action_list}")
            
            if episode_reward > 1e9 and episode_reward>=max(reward_history):
                print("Great epsiode detected !")
                save_name = f"best_model.pth"
                base_path_model =  os.path.join(base_path, "models")
                self.save(base_path_model,save_name)

            
            if (episode + 1) % 100 == 0:
                save_name = f"model_ep{episode+1}_reward{int(episode_reward)}.pth"
                base_path_model =  os.path.join(base_path, "models")
        
                self.save(base_path_model,save_name)
                
                
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
                
            
            action_list = np.zeros(4)
        
        return [episode_lengths, discounted_rewards, episode_rewards, V0_list]
    
    def save(self, path,file_name):
        if not os.path.exists(path):
            os.makedirs(path)
            
        final_path=os.path.join(path,file_name)        
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
        
        torch.save(state_dict, final_path)
        
    def load(self):
        path = 'models/ppo_agent_20250114_120223/models/best_model.pth'

        try:
            checkpoint = torch.load(path)
            
            
            config = checkpoint['config']
            self.clip_epsilon = config['clip_epsilon']
            self.value_coef = config['value_coef']
            self.entropy_coef = config['entropy_coef']
            self.max_grad_norm = config['max_grad_norm']
            self.batch_size = config['batch_size']
            self.gamma = config['gamma']
            self.epsilon = 0.01
            self.heuristic_prob =  0.05
            
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"Model successfully loaded from {path}")
            
        except FileNotFoundError:
            print(f"File {path} not found")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def evaluate(self,env,nb_episode=10):
        rewards = np.zeros(nb_episode)
        for i in range(nb_episode):
            obs, info = env.reset()
            done = False
            truncated = False
            episode_reward = 0
            while not done and not truncated:
                action, _, _ = self.select_action(obs)
                obs, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
            rewards[i] = episode_reward
        
        return np.mean(rewards)


