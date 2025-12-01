import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from networks import DQN, DuelingDQN, RainbowDQN
from efficient_replay_buffer import EfficientPrioritizedReplayBuffer
from collections import deque
from config import *


class Agent:
    """Unified DQN/Rainbow Agent"""
    
    def __init__(self, n_actions, device, use_rainbow=True):
        self.n_actions = n_actions
        self.device = device
        self.gamma = GAMMA
        self.use_rainbow = use_rainbow
        
        #Exploration parameters
        self.epsilon = EPSILON_START
        self.epsilon_min = EPSILON_END
        self.epsilon_decay = EPSILON_DECAY
        
        #Multi-step buffer
        if USE_MULTI_STEP and use_rainbow:
            self.n_step = N_STEP
            self.n_step_buffer = []
        else:
            self.n_step = 1
        
        #Beta annealing for prioritized replay
        self.beta = PRIORITY_BETA_START
        self.beta_frames = PRIORITY_BETA_FRAMES
        self.frame_count = 0

        #Select network architecture
        if use_rainbow:
            self.policy_net = RainbowDQN(n_actions).to(device)
            self.target_net = RainbowDQN(n_actions).to(device)
        elif USE_DUELING:
            self.policy_net = DuelingDQN(n_actions).to(device)
            self.target_net = DuelingDQN(n_actions).to(device)
        else:
            self.policy_net = DQN(n_actions).to(device)
            self.target_net = DQN(n_actions).to(device)
        
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE, eps=ADAM_EPSILON)

        #Select replay buffer
        if use_rainbow and USE_PRIORITIZED_REPLAY:
            self.replay_buffer = EfficientPrioritizedReplayBuffer(REPLAY_BUFFER_SIZE, PRIORITY_ALPHA)
            print(f"Using memory-efficient prioritized replay buffer")
        else:
            self.replay_buffer = deque(maxlen=REPLAY_BUFFER_SIZE)
            print(f"Using simple replay buffer")
        
        self.steps = 0
    
    def select_action(self, state, training=True):
        """Select action using noisy networks or epsilon-greedy"""
        
        #Noisy networks exploration
        if USE_NOISY_NETS and self.use_rainbow:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                self.policy_net.train() if training else self.policy_net.eval()
                
                if USE_DISTRIBUTIONAL:
                    q_values = self.policy_net.get_q_values(state_tensor)
                else:
                    q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        
        #Epsilon-greedy exploration
        if training and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        
        #Greedy action selection
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            if self.use_rainbow and USE_DISTRIBUTIONAL:
                q_values = self.policy_net.get_q_values(state_tensor)
            else:
                q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def push_experience(self, state, action, reward, next_state, done):
        """Add experience to replay buffer"""
        if self.n_step > 1:
            self.n_step_buffer.append((state, action, reward, next_state, done))
            
            if len(self.n_step_buffer) == self.n_step:
                #Compute n-step return
                n_step_state = self.n_step_buffer[0][0]
                n_step_action = self.n_step_buffer[0][1]
                n_step_reward = sum([self.n_step_buffer[i][2] * (self.gamma ** i) 
                                    for i in range(self.n_step)])
                n_step_next_state = self.n_step_buffer[-1][3]
                n_step_done = self.n_step_buffer[-1][4]
                
                #Add to buffer
                if hasattr(self.replay_buffer, 'push'):
                    self.replay_buffer.push(n_step_state, n_step_action, n_step_reward, 
                                           n_step_next_state, n_step_done)
                else:
                    self.replay_buffer.append((n_step_state, n_step_action, n_step_reward, 
                                              n_step_next_state, n_step_done))
                
                self.n_step_buffer.pop(0)
            
            #Flush buffer at episode end
            if done and self.n_step_buffer:
                while self.n_step_buffer:
                    n = len(self.n_step_buffer)
                    n_step_state = self.n_step_buffer[0][0]
                    n_step_action = self.n_step_buffer[0][1]
                    n_step_reward = sum([self.n_step_buffer[i][2] * (self.gamma ** i) 
                                        for i in range(n)])
                    n_step_next_state = self.n_step_buffer[-1][3]
                    n_step_done = self.n_step_buffer[-1][4]
                    
                    if hasattr(self.replay_buffer, 'push'):
                        self.replay_buffer.push(n_step_state, n_step_action, n_step_reward,
                                               n_step_next_state, n_step_done)
                    else:
                        self.replay_buffer.append((n_step_state, n_step_action, n_step_reward,
                                                  n_step_next_state, n_step_done))
                    self.n_step_buffer.pop(0)
        else:
            if hasattr(self.replay_buffer, 'push'):
                self.replay_buffer.push(state, action, reward, next_state, done)
            else:
                self.replay_buffer.append((state, action, reward, next_state, done))
    
    def train_step(self, batch_size=BATCH_SIZE):
        """Perform one training step"""
        if len(self.replay_buffer) < MIN_REPLAY_SIZE:
            return None
        
        if len(self.replay_buffer) < batch_size:
            batch_size = len(self.replay_buffer)
        
        #Anneal beta for prioritized replay
        if hasattr(self, 'beta_frames'):
            self.frame_count += 1
            self.beta = min(1.0, PRIORITY_BETA_START + 
                           self.frame_count * (1.0 - PRIORITY_BETA_START) / self.beta_frames)
        
        #Sample from replay buffer
        if hasattr(self.replay_buffer, 'sample'):
            batch, indices, weights = self.replay_buffer.sample(batch_size, self.beta)
            if batch is None:
                return None
            weights = torch.FloatTensor(weights).to(self.device)
        else:
            sample_indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
            batch_list = [self.replay_buffer[i] for i in sample_indices]
            states = np.array([b[0] for b in batch_list])
            actions = np.array([b[1] for b in batch_list])
            rewards = np.array([b[2] for b in batch_list])
            next_states = np.array([b[3] for b in batch_list])
            dones = np.array([b[4] for b in batch_list])
            batch = (states, actions, rewards, next_states, dones)
            indices = None
            weights = torch.ones(batch_size).to(self.device)
        
        states, actions, rewards, next_states, dones = batch
        
        #Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        #Reset noise for noisy networks
        if USE_NOISY_NETS and self.use_rainbow:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
        
        #Compute loss
        if USE_DISTRIBUTIONAL and self.use_rainbow:
            loss, td_errors = self._distributional_loss(states, actions, rewards, 
                                                        next_states, dones, weights)
        else:
            loss, td_errors = self._standard_loss(states, actions, rewards, 
                                                  next_states, dones, weights)
        
        #Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        #Update priorities
        if indices is not None:
            self.replay_buffer.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        self.steps += 1
        
        #Update target network
        if self.steps % TARGET_UPDATE_FREQUENCY == 0:
            self.update_target_network()
        
        return loss.item()
    
    def _standard_loss(self, states, actions, rewards, next_states, dones, weights):
        """Standard DQN loss with Double DQN"""
        #Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        
        #Target Q values
        with torch.no_grad():
            if USE_DOUBLE_DQN:
                next_actions = self.policy_net(next_states).argmax(1)
                next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            else:
                next_q = self.target_net(next_states).max(1)[0]
            
            target_q = rewards + (1 - dones) * (self.gamma ** self.n_step) * next_q
        
        #TD errors
        td_errors = torch.abs(current_q - target_q)
        
        #Weighted loss
        loss = (weights * (current_q - target_q).pow(2)).mean()
        
        return loss, td_errors
    
    def _distributional_loss(self, states, actions, rewards, next_states, dones, weights):
        """Distributional RL (C51) loss"""
        batch_size = states.size(0)
        
        #Current distribution
        dist = self.policy_net(states)
        actions = actions.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, N_ATOMS)
        dist = dist.gather(1, actions).squeeze(1)
        
        #Target distribution
        with torch.no_grad():
            if USE_DOUBLE_DQN:
                next_q_values = self.policy_net.get_q_values(next_states)
                next_actions = next_q_values.argmax(1)
                next_dist = self.target_net(next_states)
                next_actions = next_actions.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, N_ATOMS)
                next_dist = next_dist.gather(1, next_actions).squeeze(1)
            else:
                next_dist = self.target_net(next_states)
                next_q_values = (next_dist * self.policy_net.atoms).sum(2)
                next_actions = next_q_values.argmax(1)
                next_actions = next_actions.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, N_ATOMS)
                next_dist = next_dist.gather(1, next_actions).squeeze(1)
            
            #Compute projected distribution
            atoms = self.policy_net.atoms
            delta_z = (V_MAX - V_MIN) / (N_ATOMS - 1)
            Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * (self.gamma ** self.n_step) * atoms
            Tz = Tz.clamp(V_MIN, V_MAX)

            #Project onto support
            b = (Tz - V_MIN) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            #Fix disappearing probability
            l[(u > 0) * (l == u)] -= 1
            u[(l < (N_ATOMS - 1)) * (l == u)] += 1
            l = l.clamp(0, N_ATOMS - 1)
            u = u.clamp(0, N_ATOMS - 1)
            
            #Distribute probability
            target_dist = torch.zeros_like(next_dist)
            offset = torch.linspace(0, (batch_size - 1) * N_ATOMS, batch_size).long()\
                .unsqueeze(1).expand(batch_size, N_ATOMS).to(self.device)
            
            target_dist.view(-1).index_add_(0, (l + offset).view(-1), 
                                           (next_dist * (u.float() - b)).view(-1))
            target_dist.view(-1).index_add_(0, (u + offset).view(-1), 
                                           (next_dist * (b - l.float())).view(-1))
        
        #Cross-entropy loss
        loss = -(target_dist * dist.clamp(min=1e-8).log()).sum(1)
        
        #TD errors for prioritized replay
        with torch.no_grad():
            current_q = (dist * atoms).sum(1)
            target_q = (target_dist * atoms).sum(1)
            td_errors = torch.abs(current_q - target_q)
        
        #Weighted loss
        loss = (weights * loss).mean()
        
        return loss, td_errors
    
    def update_target_network(self):
        """Update target network parameters"""
        tau = TAU
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)
    
    def decay_epsilon(self):
        """Decay exploration rate"""
        if USE_NOISY_NETS and self.use_rainbow:
            return
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, path):
        """Save model checkpoint"""
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, path)
    
    def load(self, path):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']