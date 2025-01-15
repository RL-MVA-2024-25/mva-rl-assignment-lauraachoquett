from gymnasium.wrappers import TimeLimit
# from env_hiv import HIVPatient
from fast_env import FastHIVPatient
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
from copy import deepcopy
import time
import matplotlib.pyplot as plt
from evaluate import evaluate_HIV, evaluate_HIV_population
from torch.optim.lr_scheduler import ReduceLROnPlateau

import torch.multiprocessing as mp
from utils import ReplayBuffer
    
class Dueling_DQN_Enhanced(nn.Module):
    def __init__(self, input=6, output=4, layer=256):
        super(Dueling_DQN_Enhanced, self).__init__()
        
        self.feature_network = nn.Sequential(
            nn.Linear(input, layer),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            
            nn.Linear(layer, layer*2),  
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1),
            
            nn.Linear(layer*2, layer), 
            nn.LeakyReLU(0.01),
        )
        
        self.value_stream = nn.Sequential(
            nn.Linear(layer, layer//2),
            nn.LeakyReLU(0.01),
            nn.Linear(layer//2, layer//4),
            nn.LeakyReLU(0.01),
            nn.Linear(layer//4, 1)
        )
        
        self.advantage_stream = nn.Sequential(
            nn.Linear(layer, layer//2),
            nn.LeakyReLU(0.01),
            nn.Linear(layer//2, layer//4),
            nn.LeakyReLU(0.01),
            nn.Linear(layer//4, output)
        )
        
        self.apply(self._initialize_weights)
        
    def forward(self, x):
        features = self.feature_network(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        advantages_mean = advantages.mean(dim=1, keepdim=True)
        advantages_std = advantages.std(dim=1, keepdim=True) + 1e-6
        normalized_advantages = (advantages - advantages_mean) / advantages_std
        
        qvals = values + normalized_advantages
        return qvals

    @staticmethod
    def _initialize_weights(m):
        if isinstance(m, nn.Linear):
            fan_in = m.weight.data.size()[0]
            scale = 1 / np.sqrt(fan_in)
            nn.init.orthogonal_(m.weight.data, gain=scale)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

class Dueling_DQN(nn.Module):
    def __init__(self, input=6, output=4, layer=256):
        super(Dueling_DQN, self).__init__()

        self.feature_network = nn.Sequential(
            nn.Linear(input, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, layer),
            nn.ReLU()
        )
        
        # Value stream - estimera V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, 1)  
        )
        
        # Advantage stream - estimera A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(layer, layer),
            nn.ReLU(),
            nn.Linear(layer, output) 
        )
        
        self._initialize_weights()

    def forward(self, x):
        features = self.feature_network(x)
        
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        qvals = values + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return qvals

    def _initialize_weights(self):

        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    


env = TimeLimit(env=FastHIVPatient(domain_randomization=True), max_episode_steps=200)  


class DoubleDuelingDQN:
    config = {
          'learning_rate': 0.001,
          'gamma': 0.98,
          'buffer_size': 1000000,
          'epsilon_min': 0.01,
          'epsilon_max': 1,
          'epsilon_decay_period': 25000,
          'epsilon_delay_decay': 3000,
          'batch_size': 216,
          'max_gradient_steps' : 10,
          'epsilon_seuil' : 0.25,
          'episode_seuil' : 20,
          'explore_episodes' : 35,
          'patience_lr' : 8,
          'udpate_target_freq' : 400}
    
    dueling_dqn_deterministic = Dueling_DQN()

    def __init__(self):        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_policy = self.dueling_dqn_deterministic.to(self.device)
        self.model_target = deepcopy(self.model_policy).to(self.device)
        self.gamma = self.config['gamma']

        self.batch_size = self.config['batch_size']
        self.memory = ReplayBuffer(self.config['buffer_size'], self.device) # self.max_episode
 

        self.lr = self.config['learning_rate']

        self.epsilon_max = self.config['epsilon_max']
        self.epsilon_min = self.config['epsilon_min']
        self.epsilon_stop = self.config['epsilon_decay_period']
        self.epsilon_delay = self.config['epsilon_delay_decay']
        self.update_target_frequency = self.config['udpate_target_freq']
        self.epsilon_step = (self.epsilon_max-self.epsilon_min)/self.epsilon_stop

        self.max_gradient_steps = self.config['max_gradient_steps']
        self.patience = self.config['patience_lr']
        self.explore_episodes = self.config['explore_episodes']
        self.criterion = torch.nn.SmoothL1Loss()
      
        self.optimizer = torch.optim.Adam(self.model_policy.parameters(), lr= self.lr)
        self.epsilon_seuil = self.config["epsilon_seuil"]
        self.scheduler  = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=self.patience, verbose= True)
        self.episode_seuil = self.config['episode_seuil']
        
        self.compteur_stop = 0
        self.sampling_time = 0
        self.episode_time = 0

        self.epsilon  = self.epsilon_max
        self.step = 1
        self.gradient_steps = 0
        self.var = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.mu = torch.tensor(0, device=self.device, dtype=torch.float32)
        self.previous_best = 0
        self.episode_seuil += self.explore_episodes


    def greedy_action(self, state):
        state = torch.Tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            Q,  = self.model_policy(state)
            return torch.argmax(Q).item()
        
    
    def act(self, state):
        state = np.sign(state)*np.log(1+np.abs(state))
        if self.step > self.epsilon_delay:
            self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_step)
        if np.random.rand() < self.epsilon:
            action = env.action_space.sample()
            return action
        else:
            action = self.greedy_action(state)           
        return action

    def update(self, double_dqn): 
        start_sampling = time.perf_counter()
        X, A, R, Y, D = self.memory.sample(self.batch_size)

  
        X, A, R, Y, D = X.to(self.device, non_blocking=True), A.to(self.device, non_blocking=True), R.to(self.device, non_blocking=True), Y.to(self.device, non_blocking=True), D.to(self.device, non_blocking=True)
        R = torch.sign(R) * torch.log(1 + torch.abs(R))
        X = torch.sign(X) * torch.log(1 + torch.abs(X))
        Y = torch.sign(Y) * torch.log(1 + torch.abs(Y))
        self.sampling_time += time.perf_counter() - start_sampling
        if double_dqn :
            next_actions = self.model_policy(Y).argmax(dim=1)  
            QY_next = self.model_target(Y).gather(1, next_actions.unsqueeze(1)).squeeze(1).detach()
            if len(R.shape) > 1:
                R = R.squeeze()
            if len(D.shape) > 1:
                D = D.squeeze()
            if len(QY_next.shape) > 1:
                QY_next = QY_next.squeeze()
            update = R + self.gamma * QY_next * (1 - D)  
            if A.shape[-1] > 1:
                QXA = self.model_policy(X).gather(1, A.to(torch.long).unsqueeze(1))
            else : 
                QXA = self.model_policy(X).gather(1, A.to(torch.long))
            loss = self.criterion(QXA, update.unsqueeze(1))
        else:
            QYmax = self.model_target(Y).max(1)[0].detach()
            update = torch.addcmul(R, 1-D, QYmax, value=self.gamma)
            QXA = self.model_policy(X).gather(1, A.to(torch.long).unsqueeze(1))
            loss = self.criterion(QXA, update.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() 
        

        
    def gradient_steps_calculation(self):
        k = 2
        min_steps = 1  
        max_steps = self.max_gradient_steps  
        if self.epsilon >0.55:
            return 1
        elif self.epsilon > self.epsilon_seuil:
            scale = np.exp(-k * (self.epsilon - self.epsilon_seuil) / (1 - self.epsilon_seuil))
        else:
            scale = 1- np.exp(-k * (self.epsilon - self.epsilon_min) / (self.epsilon_seuil - self.epsilon_min))
        self.gradient_steps =  int(min_steps + (max_steps - min_steps) * scale)


    def train(self, env,nb_epsiode):
        episode = 0
        episode_return = []
        state, _ = env.reset()
        episode_cum_reward = 0
        best_score = 0        
        nb_episode = nb_epsiode
        while episode < nb_episode:

            if episode != 0:
                if trunc == True :
                    self.episode_time = time.perf_counter()
           
            action = self.act(state)
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state , action, reward,next_state, trunc) # ,episode
            episode_cum_reward += reward
        
  
            if trunc == True:
                self.gradient_steps_calculation()
                if self.epsilon < self.epsilon_seuil:
                    self.scheduler.step(best_score)

            if len(self.memory) > self.batch_size:    
                for i in range(self.gradient_steps):
                    self.update(double_dqn=True)
            
            if self.step % self.update_target_frequency  == 0:
                self.model_target.load_state_dict(self.model_policy.state_dict())

            if done or trunc :           
                episode += 1
                episode_return.append(episode_cum_reward)
                path = "models/dueling_double_dqn"
                if episode > self.episode_seuil and episode_cum_reward>1e10 :
                    self.model_policy.eval()
                    validation_score_hiv =  evaluate_HIV(agent=self, nb_episode=2)
                    validation_score_population = evaluate_HIV_population(agent=self, nb_episode=10)
                    score = validation_score_hiv + validation_score_population
                    self.model_policy.train()

                    if score >= best_score:
                        best_score = score
                        self.save(path)
                else:
                    validation_score_hiv = 0
                    validation_score_population = 0
                    if episode_cum_reward >= best_score:
                        best_score = episode_cum_reward
                        self.save(path)


                print(f"Episode {episode}, ",
                    f"Reward {episode_cum_reward:.2e}, ",   
                    f"Default Patient {validation_score_hiv:.2e}, ",
                    f"Random {validation_score_population:.2e}, ",                 
                    f"Episode time {(time.perf_counter() - self.episode_time):1.1f},",        
                    sep='')
            
                if (episode + 1) % 10 == 0:
                    
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(episode_return, label='Episode Return')
                    plt.axhline(y=3432807, color='r', linestyle='--', label='Seuil 3432807')
                    plt.axhline(y=1e8, color='g', linestyle='--', label='Seuil 1e8')
                    plt.yscale('log')
                    plt.xlabel('Episode')
                    plt.ylabel('Cumulative Reward')
                    plt.title(f'Training Progress - Episode {episode+1}')
                    plt.legend()
                    plt.grid(True)
                    
                    plot_path = os.path.join(path, "progress.png")
                    plt.savefig(plot_path)
                    plt.close()
                #reset the parameters : 
                self.sampling_time = 0
                state, _ = env.reset()
                episode_cum_reward = 0


            else:
                state = next_state
            self.step += 1
            
        print(f"Best score {best_score:.2e}")

        return episode_return


    def save(self, path):
        os.makedirs(path, exist_ok=True)  
        torch.save(self.model_policy.state_dict(), os.path.join(path, "best_model.pth"))
        print(f"Model saved to {os.path.join(path, 'best_model.pth')}")

    def load(self):
        file_path = "./models/dueling_double_dqn_best/best_model.pth"
        self.model_policy.load_state_dict(torch.load(file_path))
        self.epsilon=0.01
        self.model_policy.eval()
        print(f"Model loaded from {file_path}")
        return 










