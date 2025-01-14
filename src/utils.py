import random
import torch
import numpy as np
import numpy as np
import torch
from collections import namedtuple
import random
from collections import deque
import itertools

class SimpleBuffer:
    def __init__(self, capacity):
        """
        Initialise un buffer de replay simple avec une capacité fixe
        
        Args:
            capacity (int): Taille maximale du buffer
        """
        self.capacity = capacity
        self.transitions = []
        self.position = 0
        self.size = 0

    def add(self, transition):
        """
        Ajoute une transition au buffer
        
        Args:
            transition: Une transition (state, action, reward, next_state, done)
        """
        if self.size < self.capacity:
            self.transitions.append(transition)
            self.size += 1
        else:
            self.transitions[self.position] = transition
            
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Échantillonne aléatoirement un batch de transitions
        
        Args:
            batch_size (int): Taille du batch à échantillonner
            
        Returns:
            list: Un batch de transitions aléatoires ou None si le buffer n'est pas assez rempli
        """
        if self.size < batch_size:
            return None
        
        indices = np.random.choice(self.size, batch_size)
        batch = [self.transitions[idx] for idx in indices]
        return batch
    
class ReplayBuffer:
    def __init__(self, capacity, device):
        self.capacity = int(capacity) # capacity of the buffer
        self.data = []
        self.index = 0 # index of the next cell to be filled
        self.device = device
    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        # s, a, r, s_, d = torch.Tensor(s.to(self.device), a.to(self.device), r.to(self.device), s_.to(self.device), d.to(self.device)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
    def sample(self, batch_size):
        batch = random.sample(self.data, batch_size)
        return list(map(lambda x:torch.Tensor(np.array(x)).to(self.device), list(zip(*batch))))
    def __len__(self):
        return len(self.data)

class PrioritizedBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4,reward_threshold = 10e7):
        self.capacity = capacity
        self.alpha = alpha  # Détermine l'importance de la priorité
        self.beta = beta    # Facteur de correction du biais
        self.beta_increment = 0.001
        self.transitions = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.size = 0
        self.peak_rewards = []  # Pour garder trace des pics
        self.peak_threshold = None  # Sera mis à jour dynamiquement


    def add(self, transition, td_error=None, priority=50, episode_reward=0, is_peak=False):
        if is_peak:
            self.peak_rewards.append(episode_reward)
            priority = 2000.0  # Bonus de priorité pour les trajectoires de pics



        elif td_error is not None:
            priority = (abs(td_error) + 1e-5) ** self.alpha


        if self.size < self.capacity:
            self.transitions.append(transition)
            self.priorities[self.size] = priority
            self.size += 1
        else:
            self.transitions[self.position] = transition
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self.size < batch_size:
            return None
        
        # Normaliser les priorités
        probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
        
        # Échantillonnage basé sur les priorités
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        # Calcul des poids d'importance sampling
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        batch = [self.transitions[idx] for idx in indices]
        return batch, indices, weights

    def update_priorities(self, indices, td_errors):
        for idx, td_error in zip(indices, td_errors):
            self.priorities[idx] = (abs(td_error) + 1e-5) ** self.alpha


    

class ReplayBufferWithTrace:
    def __init__(self, capacity, device, lambda_trace=0.9):
        self.capacity = int(capacity)
        self.data = []
        self.index = 0
        self.device = device
        self.size = 0
        self.lambda_trace = lambda_trace

    def append(self, s, a, r, s_, d):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.index] = (s, a, r, s_, d)
        self.index = (self.index + 1) % self.capacity
        self.size = len(self.data)

    def sample(self, batch_size):
        # On prend une séquence consécutive pour les traces
        start_idx = random.randint(0, len(self.data) - batch_size)
        batch = [self.data[i] for i in range(start_idx, start_idx + batch_size)]
        
        # Conversion en tensors comme dans le buffer original
        states, actions, rewards, next_states, dones = list(zip(*batch))
        
        # Calcul des traces
        traces = torch.tensor([self.lambda_trace ** i for i in range(batch_size)], 
                            device=self.device)
        
        # Conversion en tensors et envoi sur le device avec les types appropriés
        return [
            torch.Tensor(np.array(states)).to(self.device),
            torch.LongTensor(np.array(actions)).to(self.device),  # Changé en LongTensor
            torch.Tensor(np.array(rewards)).to(self.device),
            torch.Tensor(np.array(next_states)).to(self.device),
            torch.Tensor(np.array(dones)).to(self.device),
        traces
    ]

    def __len__(self):
        return len(self.data)


def greedy_action(network, state):
    network.eval()  # Important avec BatchNorm
    device = "cuda" if next(network.parameters()).is_cuda else "cpu"
    with torch.no_grad():
        Q = network(torch.Tensor(state).unsqueeze(0).to(device))
        network.train()
        return torch.argmax(Q).item()
    
class StateNormalizer:
    def __init__(self, epsilon=1e-8):
        self.mean = None
        self.std = None
        self.epsilon = epsilon

    def fit(self, states):
        self.mean = np.mean(states, axis=0)
        self.std = np.std(states, axis=0) + self.epsilon

    def transform(self, state):
        return (state - self.mean) / self.std
    



class PrioritizedBufferReccurent:
    def __init__(self, capacity, sequence_length=10, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.sequence_length = sequence_length
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.transitions = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.size = 0
        self.gamma =0.995
        # Garder un buffer circulaire pour les états
        self.state_buffer = deque(maxlen=capacity)
        
    def add(self, state, action, reward, done, value, log_prob):
        # Ajouter l'état au buffer circulaire
        self.state_buffer.append({
            'state': state,
            'action': action,
            'reward': reward,
            'done': done,
            'value': value,
            'log_prob': log_prob
        })
        
        # Si nous avons assez d'états, créer une séquence
        if len(self.state_buffer) >= self.sequence_length:
            # Créer une séquence avec les derniers états
            sequence = list(itertools.islice(self.state_buffer, 
                          len(self.state_buffer)-self.sequence_length, 
                          len(self.state_buffer)))
            
            transition = {
                'state_sequence': np.array([s['state'] for s in sequence]),
                'action_sequence': np.array([s['action'] for s in sequence]),
                'reward_sequence': np.array([s['reward'] for s in sequence]),
                'done_sequence': np.array([s['done'] for s in sequence]),
                'value_sequence': np.array([s['value'] for s in sequence]),
                'log_prob_sequence': np.array([s['log_prob'] for s in sequence])
            }
            
            # Calculer la priorité basée sur la TD-error de la séquence
            priority = self._compute_sequence_priority(transition)
            
            if self.size < self.capacity:
                self.transitions.append(transition)
                self.priorities[self.size] = priority
                self.size += 1
            else:
                self.transitions[self.position] = transition
                self.priorities[self.position] = priority
                
            self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if self.size < batch_size:
            return None
        
        probs = self.priorities[:self.size] / self.priorities[:self.size].sum()
        indices = np.random.choice(self.size, batch_size, p=probs)
        
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        # Récupérer les transitions
        batch = [self.transitions[idx] for idx in indices]
        
        # Organiser les séquences pour le GRU
        state_sequences = torch.FloatTensor([t['state_sequence'] for t in batch])  # [batch_size, seq_len, state_dim]
        action_sequences = torch.LongTensor([t['action_sequence'] for t in batch])  # [batch_size, seq_len]
        reward_sequences = torch.FloatTensor([t['reward_sequence'] for t in batch])  # [batch_size, seq_len]
        done_sequences = torch.FloatTensor([t['done_sequence'] for t in batch])  # [batch_size, seq_len]
        value_sequences = torch.FloatTensor([t['value_sequence'] for t in batch])  # [batch_size, seq_len]
        log_prob_sequences = torch.FloatTensor([t['log_prob_sequence'] for t in batch])  # [batch_size, seq_len]
        
        # Créer un dictionnaire organisé
        batch_dict = {
            'states': state_sequences,
            'actions': action_sequences,
            'rewards': reward_sequences,
            'dones': done_sequences,
            'values': value_sequences,
            'log_probs': log_prob_sequences,
            'weights': torch.FloatTensor(weights)
        }
        
        return batch_dict, indices, weights
    def _compute_sequence_priority(self, transition):
        # Calcul de la TD-error sur la séquence complète
        rewards = transition['reward_sequence']
        values = transition['value_sequence']
        dones = transition['done_sequence']
        
        # Calculer la TD-error comme la somme des erreurs sur la séquence
        td_errors = np.zeros_like(rewards)
        for t in range(len(rewards)-1):
            td_errors[t] = rewards[t] + self.gamma * values[t+1] * (1-dones[t]) - values[t]
        # Pour le dernier pas de temps
        td_errors[-1] = rewards[-1] - values[-1]
        
        # Priorité comme moyenne des TD-errors absolues
        priority = np.mean(np.abs(td_errors)) + 1e-6
        return priority
    
    def update_priorities(self, indices, td_errors):
        """
        Met à jour les priorités du buffer pour les transitions données
        
        Args:
            indices (np.array): Indices des transitions à mettre à jour
            td_errors (np.array): Nouvelles TD-errors pour ces transitions
        """
        # Conversion des TD-errors en priorités
        # On utilise l'erreur absolue élevée à la puissance alpha
        new_priorities = (np.abs(td_errors) + 1e-6) ** self.alpha
        
        # Mise à jour des priorités aux indices spécifiés
        self.priorities[indices] = new_priorities

class GreedyHeuristic:
    def __init__(self):
        # Points d'équilibre
        self.healthy_T1 = 967839
        self.healthy_V = 415
        self.healthy_E = 353108

        self.unhealthy_T1 = 163573
        self.unhealthy_V = 63919
        self.unhealthy_E = 24

        # Historique des actions pour gérer les interruptions
        self.consecutive_treatments = 0
        self.force_break = False
        self.post_crisis = 0  # Nouveau compteur pour gérer la phase post-crise

        self.V_threshold_medium = 10e4
        self.V_threshold_medium_min = 10e4
        self.V_threshold_critic = 30e3

    def select_action(self, state):
        T1, T1star, T2, T2star, V, E = state

        # Gestion des pauses forcées
        if self.force_break:
            self.force_break = False
            self.consecutive_treatments = 0
            self.post_crisis = 0  # Réinitialiser le compteur post-crise
            return 0  # Pause forcée

        # Logique de base pour la sélection d'action
        action = self._base_select_action(state)

        # Mise à jour du compteur de traitements consécutifs
        if action > 0:  # Si un médicament est prescrit
            self.consecutive_treatments += 1
        else:
            self.consecutive_treatments = 0

        # Force une pause si trop de traitements consécutifs
        if self.consecutive_treatments >= 6:
            self.force_break = True

        return action

    def _base_select_action(self, state):
        T1, T1star, T2, T2star, V, E = state
        
        # Cas critique : charge virale très élevée ou T1 très bas
        if V >= self.unhealthy_V * 0.8 or T1 <= self.unhealthy_T1:
            self.post_crisis = 4  # Initialiser la phase post-crise
            return 3  # Les deux médicaments
        
        # Phase post-crise : transition progressive
        if self.post_crisis > 0:
            self.post_crisis -= 1
            if self.post_crisis > 2:
                return 2 # Maintenir les deux médicaments
            elif self.post_crisis > 1:
                return 2  # Passer au PI
            else:
                return 1  # Finir avec RTI
            
        # Cas préoccupant : charge virale élevée mais pas critique
        if V > self.healthy_V * 20:
            if E < self.healthy_E * 0.2:  # Système immunitaire faible
                return 2  # PI prioritaire
            return 1  # RTI pour protéger T1
            
        # Cas stable : laisser le système immunitaire travailler
        if T1 > self.healthy_T1 * 0.6 and V < self.healthy_V * 5 and E > self.healthy_E * 0.3:
            return 0
            
        # Cas intermédiaire : alternance selon les besoins
        if V > self.healthy_V * 10:
            return 2  # PI pour la charge virale
        if T1 < self.healthy_T1 * 0.5:
            return 1  # RTI pour T1
        
        return 0  # Par défaut, laisser le système immunitaire travailler
        
class ImprovedHeuristic:
    def __init__(self):
        # Points d'équilibre
        self.healthy_T1 = 967839
        self.healthy_V = 415
        self.healthy_E = 353108
        
        # Seuils pour V (charge virale)
        self.V_safe = 5000          # Zone verte
        self.V_warning = 20000      # Zone orange
        self.V_danger = 50000       # Zone rouge
        self.V_critical = 100000    # Zone critique
        
        # Seuils pour E (système immunitaire)
        self.E_critical = 30        # Système immunitaire très faible
        self.E_weak = 50           # Système immunitaire faible
        self.E_good = 80           # Système immunitaire correct
        
        # Gestion du traitement
        self.consecutive_treatments = 0
        self.rest_period = 0
        
    def select_action(self, state):
        T1, T1star, T2, T2star, V, E = state
        
        # Période de repos forcée
        if self.rest_period > 0:
            self.rest_period -= 1
            return 0
            
        # Situation critique : deux médicaments
        if V > self.V_critical or (V > self.V_danger and E < self.E_critical):
            self.consecutive_treatments += 1
            if self.consecutive_treatments > 4:
                self.rest_period = 2
                self.consecutive_treatments = 0
                return 0
            return 3
            
        # Zone de danger : un médicament approprié
        if V > self.V_warning:
            self.consecutive_treatments += 1
            if E < self.E_weak:
                return 2  # PI privilégié si système immunitaire faible
            return 1     # RTI sinon
            
        # Zone d'attention : alternance ou repos
        if V > self.V_safe:
            if self.consecutive_treatments > 0:
                self.consecutive_treatments = 0
                return 0
            return 1 if E > self.E_good else 2
            
        # Zone sûre : repos
        self.consecutive_treatments = 0
        return 0