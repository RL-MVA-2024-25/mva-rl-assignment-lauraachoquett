from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from fast_env import FastHIVPatient
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from dqn import dqn_agent
from double_dqn import double_dqn_agent
from double_dqn_trace import double_dqn_agent_trace
from ppo_dis import PPOAgent_test
import gymnasium as gym 
import matplotlib.pyplot as plt 
import pandas as pd
from utils import GreedyHeuristic
import os
import matplotlib.lines as mlines
from dueling_double_dqn import DoubleDuelingDQN

env = TimeLimit(
    env=FastHIVPatient(domain_randomization=True), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.

# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):

        config = {
            'agent_type': 'dueling_double_dqn',     
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

        if agent_type == 'dqn':
            self.model = self._build_dqn()
            self.agent = dqn_agent(config, self.model)

        elif agent_type == 'double_dqn_trace':
            self.model = self._build_dqn()
            self.agent = double_dqn_agent_trace(config, self.model)

        elif agent_type == 'double_dqn':
            self.model = self._build_dqn()
            self.agent = double_dqn_agent(config, self.model)

        elif agent_type == 'ppo':
            self.agent = PPOAgent_test(
                config=config,
                state_dim=6, 
                action_dim=4,
            )
        elif agent_type == 'dueling_double_dqn':
            self.agent = DoubleDuelingDQN()

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
    
    def test_heuristic(self, save_dir='./graphs_PostcrisisHeuristic_random_patient'):
        """Initialise le buffer avec un mélange de trajectoires heuristiques et aléatoires"""
        print("Initializing buffer with heuristic and random trajectories...")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        heuristic_rewards = []
        random_rewards = []
        heuristic_actions = []
        heuristic_states = []
        random_actions = []
        random_states = []
        
        # Environnement simulé
        env = TimeLimit(
            env=FastHIVPatient(domain_randomization=True), max_episode_steps=200
        )
        
        # Collecte des données
        for episode in range(20):
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
                    episode_actions.append(action)
                    episode_states.append(state)
                
                next_state, reward, done, trunc, _ = env.step(action)
                episode_reward += reward
                state = next_state
                
                if trunc:
                    break
            
            if is_heuristic_episode:
                heuristic_rewards.append(episode_reward)
                heuristic_actions.append(episode_actions)
                heuristic_states.append(episode_states)
            else:
                random_rewards.append(episode_reward)
                random_actions.append(episode_actions)
                random_states.append(episode_states)
        
        # Afficher les statistiques de récompense
        if heuristic_rewards:
            avg_heuristic = np.mean(heuristic_rewards)
            std_heuristic = np.std(heuristic_rewards)
            print(f"Heuristic episodes ({len(heuristic_rewards)}): Average reward = {avg_heuristic:.2f} ± {std_heuristic:.2f}")
        
        if random_rewards:
            avg_random = np.mean(random_rewards)
            std_random = np.std(random_rewards)
            print(f"Random episodes ({len(random_rewards)}): Average reward = {avg_random:.2f} ± {std_random:.2f}")
        
        # Meilleur épisode heuristique
        index_max = np.argmax(heuristic_rewards)
        best_heuristic_states = np.array(heuristic_states[index_max])
        best_heuristic_actions = np.array(heuristic_actions[index_max])
        
        # Meilleur épisode aléatoire
        index_max_rd = np.argmax(random_rewards)
        best_random_states = np.array(random_states[index_max_rd])
        best_random_actions = np.array(random_actions[index_max_rd])
        action_color_map = {
            0: 'grey',   # Action 0 : 'No Drug' (rien)
            1: 'blue',   # Action 1 : 'RTI'
            2: 'green',  # Action 2 : 'PI'
            3: 'red'     # Action 3 : 'Both'
        }
        def plot_episode(states, actions, title_prefix, save_prefix):
            os.makedirs(save_dir, exist_ok=True)
            df = pd.DataFrame(states, columns=['T1', 'T1star', 'T2', 'T2star', 'V', 'E'])
            df.to_csv(f"{save_dir}/{save_prefix}_states.csv", index=False)
            print(f"Saved states of {title_prefix.lower()} to {save_dir}/{save_prefix}_states.csv")
            
            colors = ['grey', 'blue', 'green', 'red']  
            for column in df.columns:
                plt.figure(figsize=(10, 6))
                plt.plot(df[column], label=column, linewidth=1)
                colors = [action_color_map[action] for action in actions]
                plt.scatter(range(len(actions)), df[column], c=colors, s=10, label='Actions')
                plt.yscale('log')  
                plt.title(f"{title_prefix}: Evolution of {column}")
                plt.xlabel("Time step")
                plt.ylabel(column)
                legend_elements = [
                    mlines.Line2D([], [], marker='o', color='grey', markerfacecolor='grey', markersize=10, label='No Drug (Action 0)'),
                    mlines.Line2D([], [], marker='o', color='blue', markerfacecolor='blue', markersize=10, label='RTI (Action 1)'),
                    mlines.Line2D([], [], marker='o', color='green', markerfacecolor='green', markersize=10, label='PI (Action 2)'),
                    mlines.Line2D([], [], marker='o', color='red', markerfacecolor='red', markersize=10, label='Both (Action 3)')
                ]

                plt.legend(handles=legend_elements, loc='best')
                plt.savefig(f"{save_dir}/{save_prefix}_{column}.png")
                print(f"Saved plot for {column} of {title_prefix.lower()} to {save_dir}/{save_prefix}_{column}.png")
                plt.close()

       
        plot_episode(
            best_heuristic_states, 
            best_heuristic_actions, 
            title_prefix="Best Heuristic Episode", 
            save_prefix="heuristic_best_episode"
        )
        
        plot_episode(
            best_random_states, 
            best_random_actions, 
            title_prefix="Best Random Episode", 
            save_prefix="random_best_episode"
        )

            

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





