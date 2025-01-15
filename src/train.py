from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
#from dqn import dqn_agent
#from double_dqn import double_dqn_agent
#from double_dqn_trace import double_dqn_agent_trace
from ppo_dis import PPOAgent_test
#import matplotlib.pyplot as plt 
from utils import GreedyHeuristic
#import matplotlib.lines as mlines
from dueling_double_dqn import DoubleDuelingDQN

env = TimeLimit(
    env=HIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.

# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):

        config= {
            'agent_type': 'ppo',
            'clip_epsilon': 0.1, 
            'value_coef': 0.5,   
            'entropy_coef': 0.01, 
            'max_grad_norm': 0.5,
            'batch_size': 512,    
            'learning_rate': 1e-4,
            'gamma': 0.995,       
        }

        
        self.env=env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = self.env.observation_space.shape[0]
        self.n_action = self.env.action_space.n 
        self.nb_epsiode= 10000
        self.nb_neurons=24
        self.heuristic = GreedyHeuristic()
        self.heuristic_prob = 0.1
        agent_type = config.get('agent_type', 'dqn') 

        if agent_type == 'dueling_double_dqn':
            self.agent = DoubleDuelingDQN()
        elif agent_type == 'ppo':
            self.agent = PPOAgent_test(config=config)

        else:
            raise ValueError(f"Type d'agent non supporté: {agent_type}")



    def act(self, observation, use_random=False):
        return self.agent.act(state=observation)
    
    def train(self):
        ep_length, disc_rewards, tot_rewards, V0 = self.agent.train(self.env,400)
        return [ep_length,disc_rewards,tot_rewards,V0]
        

    def save(self, path):
        self.agent.save(path=path)

    def load(self ) -> None:
        self.agent.load()

    def evaluate_agent(self,env,nb_episode):
        reward = self.agent.evaluate(env,nb_episode)
        return reward
        

    def _build_dqn(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, self.nb_neurons),
            nn.BatchNorm1d(self.nb_neurons),
            nn.ReLU(),
            nn.Linear(self.nb_neurons, self.nb_neurons*2),
            nn.BatchNorm1d(self.nb_neurons*2),
            nn.ReLU(),
            nn.Linear(self.nb_neurons*2, self.nb_neurons),
            nn.BatchNorm1d(self.nb_neurons),
            nn.ReLU(), 
            nn.Linear(self.nb_neurons, self.n_action)
        ).to(self.device)

config_double_dqn_trace = {'agent_type': 'double_dqn_trace',
              'lambda_trace':0.9,
          'nb_actions': env.action_space.n,
          'update_target_tau':0.01,
          'target_update_freq':20,
          'learning_rate': 0.0003,
          'gamma': 0.99,
          'buffer_size': 5000,
          'epsilon_min': 0.05,
          'epsilon_max': 1.,
          'epsilon_decay_period': 15000,
          'epsilon_delay_decay': 4000,
          'batch_size': 248,
          'gradient_steps': 1,
          'update_target_strategy': 'ema', # or 'replace'
          'update_target_freq': 10,
          'update_target_tau': 0.01,
          'criterion': torch.nn.SmoothL1Loss(),
          'monitoring_nb_trials': 50}

config = {
    'agent_type': 'onlineSarsaLambda',
    'gamma': 0.99,
    'lambda_': 0.9,
    'learning_rate': 0.01,
    'epsilon_max': 1.0,
    'epsilon_min': 0.01,
    'epsilon_step': 0.001,
    'epsilon_delay': 0,
    'batch_size': 32  # optionnel
}
config_ppo = {
    'agent_type': 'ppo',
    'clip_epsilon': 0.1, 
    'value_coef': 0.5,   
    'entropy_coef': 0.01, 
    'max_grad_norm': 0.5,
    'batch_size': 512,    
    'learning_rate': 1e-4,
    'gamma': 0.995,       
}



#agent=ProjectAgent()
#returns = agent.train()
""" #print("END OF THE TRAINING")

agent.load()
env = TimeLimit(
    env=FastHIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.

reward = agent.evaluate_agent(env,nb_episode=10)
print(f"Reward pour des patients random {reward:.2e}")

env = TimeLimit(
    env=FastHIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.


reward = agent.evaluate_agent(env,nb_episode=10)
print(f"Reward pour le patient par défaut {reward:.2e}")
#agent.test_heuristic() 
 """





