import torch
import numpy as np
from copy import deepcopy
from utils import ReplayBufferWithTrace
from utils import greedy_action
import os
from datetime import datetime
import matplotlib.pyplot as plt
from utils import StateNormalizer

class double_dqn_agent_trace:
    def __init__(self, config, model):
        device = "cuda" if next(model.parameters()).is_cuda else "cpu"
        self.nb_actions = config['nb_actions']
        self.gamma = config['gamma'] if 'gamma' in config.keys() else 0.95
        self.batch_size = config['batch_size'] if 'batch_size' in config.keys() else 100
        buffer_size = config['buffer_size'] if 'buffer_size' in config.keys() else int(1e5)
        self.target_update_freq = config['target_update_freq']
        self.lambda_trace=config['lambda_trace']
        self.memory = ReplayBufferWithTrace(buffer_size,device,self.lambda_trace)

        self.epsilon_max = config['epsilon_max'] if 'epsilon_max' in config.keys() else 1.
        self.epsilon_min = config['epsilon_min'] if 'epsilon_min' in config.keys() else 0.01
        self.epsilon_stop = config['epsilon_decay_period'] if 'epsilon_decay_period' in config.keys() else 1000
        self.epsilon_delay = config['epsilon_delay_decay'] if 'epsilon_delay_decay' in config.keys() else 20
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop
        self.model = model 
        self.target_model = deepcopy(self.model).to(device)
        self.criterion = config['criterion'] if 'criterion' in config.keys() else torch.nn.MSELoss()
        lr = config['learning_rate'] if 'learning_rate' in config.keys() else 0.001

        self.optimizer = config['optimizer'] if 'optimizer' in config.keys() else torch.optim.Adam(self.model.parameters(), lr=lr)
        self.nb_gradient_steps = config['gradient_steps'] if 'gradient_steps' in config.keys() else 1
        self.update_target_strategy = config['update_target_strategy'] if 'update_target_strategy' in config.keys() else 'replace'
        self.update_target_freq = config['update_target_freq'] if 'update_target_freq' in config.keys() else 20
        self.update_target_tau = config['update_target_tau'] if 'update_target_tau' in config.keys() else 0.005
        self.monitoring_nb_trials = config['monitoring_nb_trials'] if 'monitoring_nb_trials' in config.keys() else 0
        self.path = "models/dqn_model.pth"


    def gradient_step(self):
        if len(self.memory) > self.batch_size:
            X, A, R, Y, D, traces = self.memory.sample(self.batch_size)
            
            self.model.eval()
            with torch.no_grad():
                best_actions = self.model(Y).argmax(dim=1)
                QYmax = self.target_model(Y).gather(1, best_actions.unsqueeze(1)).squeeze(1)
            self.model.train()
            
            target = R + self.gamma * QYmax * (1-D)
            QXA = self.model(X).gather(1, A.unsqueeze(1)).squeeze(1)
            td_errors = target - QXA
            
            weighted_td_errors = td_errors * traces
            loss = self.criterion(QXA + weighted_td_errors, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def update_target(self):
        if self.update_target_strategy == 'replace':
            self.target_model.load_state_dict(self.model.state_dict())
        elif self.update_target_strategy == 'ema':
            for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(
                    self.update_target_tau * model_param.data + 
                    (1.0 - self.update_target_tau) * target_param.data
            )
    def train(self, env, max_episode):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = os.path.join("models", f"agent_{timestamp}")
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        print("Beginning of the train")
        episode_return = []
        episode = 0
        episode_cum_reward = 0
        state, _ = env.reset()
        epsilon = self.epsilon_max
        step = 0
        min_buffer_size = 5 * self.batch_size
        save_frequency = 10
        action_list_rd = np.zeros(4)
        action_list_greedy = np.zeros(4)
        self.model.train()
        while episode < max_episode:
            if step > self.epsilon_delay:
                epsilon = max(self.epsilon_min, epsilon-self.epsilon_step)
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
                action_list_rd[action] += 1
            else:
                action = greedy_action(self.model, state)
                action_list_greedy[action]+=1

            
            # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward
            terminated= done or trunc

            # train
            if len(self.memory) >= min_buffer_size:
                self.model.train()
                self.gradient_step()
                if episode % self.update_target_freq == 0:
                    self.update_target()

            step += 1
            if terminated:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                      ", epsilon ", '{:6.2f}'.format(epsilon), 
                      ", batch size ", '{:5d}'.format(len(self.memory)), 
                      ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                      ", action choosen at random  ",action_list_rd,
                      ", action choosen greddily  ",action_list_greedy,
                      sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                action_list_rd = np.zeros(4)
                action_list_greedy = np.zeros(4)
                episode_cum_reward = 0
                if episode % save_frequency == 0:
                    save_name = f"model_ep{episode}_reward{int(episode_cum_reward)}.pth"
                    save_path = os.path.join(base_path, save_name)
                    self.save(save_path)
                    # Création et sauvegarde du graphique
                    plt.figure(figsize=(10, 6))
                    plt.plot(episode_return, label='Episode Return')

                    # Ajout des lignes horizontales
                    plt.axhline(y=3432807, color='r', linestyle='--', label='Seuil 3432807')
                    plt.axhline(y=1e8, color='g', linestyle='--', label='Seuil 1e8')
                    plt.yscale('log')  # Passage en échelle logarithmique
                    plt.xlabel('Episode')
                    plt.ylabel('Cumulative Reward')
                    plt.title(f'Training Progress - Episode {episode}')
                    plt.legend()
                    plt.grid(True)
                    
                    # Sauvegarde du graphique
                    plot_name = f"progress.png"
                    plot_path = os.path.join(base_path, plot_name)
                    plt.savefig(plot_path)
                
            else:
                state = next_state

        return episode_return
    
    def save(self, path):
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, extension = os.path.splitext(path)
        path_with_timestamp = f"{filename}_{timestamp}{extension}"

        state_dict = {
            'config': {
                'nb_actions': self.nb_actions,
                'gamma': self.gamma,
                'batch_size': self.batch_size,
                'buffer_size': self.memory.capacity,
                'epsilon_max': self.epsilon_max,
                'epsilon_min': self.epsilon_min,
                'epsilon_decay_period': self.epsilon_stop,
                'epsilon_delay_decay': self.epsilon_delay,
                'epsilon_step': self.epsilon_step,
                'learning_rate': self.optimizer.param_groups[0]['lr'],
                'gradient_steps': self.nb_gradient_steps,
                'update_target_strategy': self.update_target_strategy,
                'update_target_freq': self.update_target_freq,
                'update_target_tau': self.update_target_tau,
                'monitoring_nb_trials': self.monitoring_nb_trials
            },
            
            'model_state': self.model.state_dict(),
            'target_model_state': self.target_model.state_dict(),
            
            'optimizer_state': self.optimizer.state_dict(),
            
            'memory': {
                'data': self.memory.data[:self.memory.size],
                'size': self.memory.size
            }
        }
    
        torch.save(state_dict, path_with_timestamp)

    def load(self) -> None:
        path = "dqn_agent.pth"
        device = "cuda" if next(self.model.parameters()).is_cuda else "cpu"
        
        try:
            state_dict = torch.load(path, map_location=device)
            
            config = state_dict['config']
            self.nb_actions = config['nb_actions']
            self.gamma = config['gamma']
            self.batch_size = config['batch_size']
            self.epsilon_max = config['epsilon_max']
            self.epsilon_min = config['epsilon_min']
            self.epsilon_stop = config['epsilon_decay_period']
            self.epsilon_delay = config['epsilon_delay_decay']
            self.epsilon_step = config['epsilon_step']
            self.nb_gradient_steps = config['gradient_steps']
            self.update_target_strategy = config['update_target_strategy']
            self.update_target_freq = config['update_target_freq']
            self.update_target_tau = config['update_target_tau']
            self.monitoring_nb_trials = config['monitoring_nb_trials']
            
            self.model.load_state_dict(state_dict['model_state'])
            self.target_model.load_state_dict(state_dict['target_model_state'])
            
            self.optimizer.load_state_dict(state_dict['optimizer_state'])
            
            memory_state = state_dict['memory']
            self.memory.states[:memory_state['size']] = memory_state['states']
            self.memory.actions[:memory_state['size']] = memory_state['actions']
            self.memory.rewards[:memory_state['size']] = memory_state['rewards']
            self.memory.next_states[:memory_state['size']] = memory_state['next_states']
            self.memory.dones[:memory_state['size']] = memory_state['dones']
            self.memory.size = memory_state['size']
            
            self.model.eval()
            self.target_model.eval()
            
        except FileNotFoundError:
            print(f"Fichier {path} non trouvé")
        except Exception as e:
            print(f"Erreur lors du chargement: {str(e)}")
