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
from utils import PrioritizedBufferReccurent

class EnhancedHybridActorCriticGRU(nn.Module):
    def __init__(self, state_dim=6, action_dim=4, hidden_dim=64, sequence_length=10):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.sequence_length = sequence_length

        
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        self.encoder_residual = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.LayerNorm(64)
        )

        
        
        self.gru = nn.GRU(
            input_size=64,
            hidden_size=hidden_dim,
            num_layers=3,
            batch_first=True,
            dropout=0.15
        )

        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        
        
        self.n_ratios = (state_dim * (state_dim - 1)) // 2

        
        
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim + self.n_ratios, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SELU(),
            nn.Linear(64, action_dim)
        )

        
        
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.SELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.SELU(),
            nn.Linear(64, 1)
        )

    def compute_ratios(self, states):
        batch_size, seq_len = states.shape[:2]
        ratios_sequence = []
        
        for t in range(seq_len):
            ratios_t = []
            for i in range(self.state_dim):
                for j in range(i+1, self.state_dim):
                    ratio = states[:, t, i] / (states[:, t, j] + 1e-6)
                    ratios_t.append(torch.log1p(ratio))
            ratios_sequence.append(torch.stack(ratios_t, dim=1))
            
        return torch.stack(ratios_sequence, dim=1)

    def forward(self, state_sequence, hidden_state=None):
        batch_size, seq_len, state_dim = state_sequence.size()
        
        
        
        encoded_states = self.encoder(state_sequence.view(-1, state_dim)) + self.encoder_residual(state_sequence.view(-1, state_dim))
        encoded_sequence = encoded_states.view(batch_size, seq_len, -1)
        
        
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size, state_sequence.device)
        gru_out, new_hidden = self.gru(encoded_sequence, hidden_state)

        
        attention_weights = torch.softmax(self.attention(gru_out).squeeze(-1), dim=1).unsqueeze(-1)
        context_vector = torch.sum(gru_out * attention_weights, dim=1, keepdim=True)
        gru_out = gru_out + context_vector 
        

        
        ratios_sequence = self.compute_ratios(state_sequence)
        
        
        actor_inputs = torch.cat([gru_out, ratios_sequence], dim=-1)
        
        
        action_logits = self.actor(actor_inputs)
        action_probs = torch.softmax(action_logits, dim=-1)
        values = self.critic(gru_out)
        
        return action_probs, values, new_hidden

    def init_hidden(self, batch_size, device):
        return torch.zeros(3, batch_size, self.hidden_dim, device=device)
    

class PPOAgent_Recurrent_Seq:
    def __init__(self, config, state_dim=6, action_dim=4):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sequence_length = 5
        self.actor_critic = EnhancedHybridActorCriticGRU(state_dim, action_dim,sequence_length=self.sequence_length).to(self.device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=config.get('learning_rate', 3e-4))
        
        
        self.clip_epsilon = config.get('clip_epsilon', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.batch_size = config.get('batch_size', 64)
        self.gamma = config.get('gamma', 0.99)
        
        
        
        self.buffer = []


        self.prioritized_buffer = PrioritizedBufferReccurent(
            capacity=10000,
            sequence_length=self.sequence_length,
            alpha=0.6,  
            beta=0.4    
            
        )
        self.epsilon = 0.1

        

        self.heuristic = GreedyHeuristic()
        self.heuristic_prob_beginning = 0.35
        self.heuristic_prob_end= 0.2

    def initialize_buffer_with_heuristic(self, env, n_episodes=40, save_dir="output"):
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
            is_heuristic_episode = np.random.random() < 0.8
            
            
            while not done:
                if is_heuristic_episode:
                    action = self.heuristic.select_action(state)
                else:
                    action = np.random.randint(4)
                    
                next_state, reward, done, trunc, _ = env.step(action)
                episode_reward+=reward
                episode_states.append(state)
                
                state_sequence = torch.FloatTensor([state]).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _, value,_ = self.actor_critic(state_sequence)
                
                self.prioritized_buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    done=done or trunc,
                    value=value.item(),
                    log_prob=0
                 )
                state=next_state
                if trunc:
                    break
            
            if is_heuristic_episode:
                heuristic_rewards.append(episode_reward)
                heuristic_actions.append(episode_actions)
                heuristic_states.append(episode_states)
            else:
                random_rewards.append(episode_reward)
        
        
        if heuristic_rewards:
            avg_heuristic = np.mean(heuristic_rewards)
            std_heuristic = np.std(heuristic_rewards)
            print(f"Heuristic episodes ({len(heuristic_rewards)}): Average reward = {avg_heuristic:.2f} ± {std_heuristic:.2f}")
        
        if random_rewards:
            avg_random = np.mean(random_rewards)
            std_random = np.std(random_rewards)
            print(f"Random episodes ({len(random_rewards)}): Average reward = {avg_random:.2f} ± {std_random:.2f}")
        

        
        if heuristic_states:
            for episode_idx, states in enumerate(heuristic_states[:1]):
                df = pd.DataFrame(states, columns=['T1', 'T1star', 'T2', 'T2star', 'V', 'E'])
                df.to_csv(f"{save_dir}/heuristic_episode_{episode_idx}_states.csv", index=False)
                print(f"Saved states of heuristic episode {episode_idx}")
                
                for column in df.columns:
                    plt.figure()
                    plt.plot(df[column])
                    plt.title(f"Evolution of {column} in episode {episode_idx}")
                    plt.xlabel("Time step")
                    plt.ylabel(column)
                    plt.savefig(f"{save_dir}/heuristic_episode_{episode_idx}_{column}.png")
                    plt.close()
        
        print(f"\nBuffer initialized with {self.prioritized_buffer.size} transitions")
        
        return {
            'heuristic_rewards': heuristic_rewards,
            'random_rewards': random_rewards,
            'heuristic_actions': heuristic_actions,
            'heuristic_states': heuristic_states
        }

        
    def select_action(self, state_sequence):

        if state_sequence.size(1) < self.sequence_length or np.random.random()<self.heuristic_prob:
            current_state = state_sequence[:, -1].cpu().numpy()[0]
            action = self.heuristic.select_action(current_state)
            with torch.no_grad():
                _ , value ,_ = self.actor_critic(state_sequence)

            return action,value[:, -1, 0].item(), 0

        if np.random.random() < self.epsilon:
            
            action = np.random.randint(4)
            with torch.no_grad():
                _, value , _ = self.actor_critic(state_sequence)
            return action, value[:, -1, 0].item(), 0
        
        
        action_probs, value , _ = self.actor_critic(state_sequence)
        
        last_probs = action_probs[0, -1]  # [action_dim]
        last_value = value[0, -1, 0]  # scalaire
        
        action = last_probs.argmax()
        log_prob = torch.log(last_probs[action])
        return action.item(), last_value.item(), log_prob.item()



    
    def update(self):
        if self.prioritized_buffer.size < self.batch_size:
            return
            
        batch_dict, indices, weights = self.prioritized_buffer.sample(self.batch_size)
        if batch_dict is None:
            return
            
        states = batch_dict['states'].to(self.device)  # [batch_size, seq_len, state_dim]
        actions = batch_dict['actions'].to(self.device)  # [batch_size, seq_len]
        rewards = batch_dict['rewards'].to(self.device)  # [batch_size, seq_len]
        dones = batch_dict['dones'].to(self.device)  # [batch_size, seq_len]
        old_values = batch_dict['values'].to(self.device)  # [batch_size, seq_len]
        old_log_probs = batch_dict['log_probs'].to(self.device)  # [batch_size, seq_len]
        weights = batch_dict['weights'].to(self.device)  # [batch_size]
        
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        
        with torch.no_grad():
            returns = torch.zeros_like(rewards)
            advantages = torch.zeros_like(rewards)
            
            for t in reversed(range(seq_len)):
                if t == seq_len - 1:
                    next_value = old_values[:, -1]
                else:
                    next_value = old_values[:, t+1]
                    
                returns[:, t] = rewards[:, t] + self.gamma * next_value * (1 - dones[:, t])
                advantages[:, t] = returns[:, t] - old_values[:, t]
            
            advantages = advantages * weights.unsqueeze(1)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        
        for _ in range(10):
            
            action_probs, values, _ = self.actor_critic(states)  # [batch_size, seq_len, action_dim], [batch_size, seq_len, 1]
            values = values.squeeze(-1)  # [batch_size, seq_len]
            
            
            flat_action_probs = action_probs.reshape(-1, action_probs.size(-1))  # [batch_size*seq_len, action_dim]
            flat_actions = actions.reshape(-1)  # [batch_size*seq_len]
            
            
            dist = Categorical(flat_action_probs)
            new_log_probs = dist.log_prob(flat_actions)
            new_log_probs = new_log_probs.view(batch_size, seq_len)  # [batch_size, seq_len]
            
            
            ratios = torch.exp(new_log_probs - old_log_probs)  # [batch_size, seq_len]
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
            
            
            td_errors = (returns - values).detach().abs().mean(dim=1).cpu().numpy()
            self.prioritized_buffer.update_priorities(indices, td_errors)

    
    def train(self, env, max_episode):
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join("models", f"ppo_agent_{timestamp}")
        if not os.path.exists(base_path):
            os.makedirs(base_path)
            
        
        episode_rewards_all=[]
        episode_lengths = []
        V0_list = []
        discounted_rewards = []
        action_list = np.zeros(4)
        
        
        initial_max_steps = 50
        final_max_steps = 200
        curriculum_length = max_episode // 4
        
        def get_max_steps(episode):
            if episode < curriculum_length:
                progress = episode / curriculum_length
                current_length = int(initial_max_steps + (final_max_steps - initial_max_steps) * progress)
                return current_length
            return final_max_steps
        
        
        base_bath_graph=os.path.join(base_path,"graphs")
        self.initialize_buffer_with_heuristic(env,save_dir=base_bath_graph)
        self.heuristic_prob = self.heuristic_prob_beginning
        initial_heuristic_prob = self.heuristic_prob_beginning

        print("Beginning of the training with curriculum")
        
        for episode in range(max_episode):
            state, _ = env.reset()
            episode_reward = 0
            step = 0
            V0 = None
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_values = []
            episode_log_probs = []
            
            max_steps_current = get_max_steps(episode)
            while step < max_steps_current:
                
                episode_states.append(state)
                
                state_sequence = torch.FloatTensor(np.array(episode_states[-self.sequence_length:] 
                            if len(episode_states) >= self.sequence_length 
                            else episode_states)).unsqueeze(0).to(self.device)
                
                
                action, value, log_prob = self.select_action(state_sequence)
                if V0 is None:
                    V0 = value

                
                self.heuristic_prob = initial_heuristic_prob * (1 - episode/max_episode) + self.heuristic_prob_end * (episode/max_episode)
                
                
                next_state, reward, done, trunc, _ = env.step(action)
                
                
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_values.append(value)
                episode_log_probs.append(log_prob)
                
                action_list[action] += 1
                episode_reward += reward
                step += 1
                
                if done or trunc:
                    break
                    
                state = next_state

            
            
            for t in range(len(episode_states)):
                self.prioritized_buffer.add(
                    state=episode_states[t],
                    action=episode_actions[t],
                    reward=episode_rewards[t],
                    done=(done or trunc) if t == len(episode_states)-1 else False,
                    value=episode_values[t],
                    log_prob=episode_log_probs[t]
                )

            
            for _ in range(1):
                self.update()
            
            
            
            episode_rewards_all.append(episode_reward)
            episode_lengths.append(step)
            V0_list.append(V0)
            
            
            print(f"Episode {episode:3d}, "
                f"episode return {episode_reward:4.1f}, "
                f"length {step:3d}/{max_steps_current}, "
                f"V0 {V0:4.1f}, "
                f"action distribution {action_list}")
            
            
            if (episode + 1) % 10 == 0:
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
                
            
            action_list = np.zeros(4)
        
        return [episode_lengths, discounted_rewards, episode_rewards, V0_list]
    
    def save(self, path, save_name):
        if not os.path.exists(path):
            os.makedirs(path)
            
        final_path= os.path.join(path,save_name)        
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
        
    def load(self, path):
        try:
            checkpoint = torch.load(path)
            
            
            config = checkpoint['config']
            self.clip_epsilon = config['clip_epsilon']
            self.value_coef = config['value_coef']
            self.entropy_coef = config['entropy_coef']
            self.max_grad_norm = config['max_grad_norm']
            self.batch_size = config['batch_size']
            self.gamma = config['gamma']
            
            
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            print(f"Model successfully loaded from {path}")
            
        except FileNotFoundError:
            print(f"File {path} not found")
        except Exception as e:
            print(f"Error loading model: {str(e)}")