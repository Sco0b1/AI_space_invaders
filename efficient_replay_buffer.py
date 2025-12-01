import numpy as np
from collections import deque
from config import PRIORITY_ALPHA


class EfficientPrioritizedReplayBuffer:
    """Memory-efficient prioritized replay buffer that stores individual frames"""
    
    def __init__(self, capacity, alpha=PRIORITY_ALPHA):
        self.capacity = capacity
        self.alpha = alpha
        
        #Store individual frames
        self.frames = deque(maxlen=capacity + 4)
        
        #Store transitions as (frame_idx, action, reward, done)
        self.transitions = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
        self.frame_count = 0
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        #Detect episode start (4 identical frames)
        is_episode_start = np.array_equal(state[0], state[1]) and \
                          np.array_equal(state[1], state[2]) and \
                          np.array_equal(state[2], state[3])
        
        #Add frames to buffer
        if len(self.frames) == 0 or is_episode_start:
            #Episode start: add all 4 frames
            for i in range(4):
                self.frames.append(state[i].copy())
                self.frame_count += 1
        else:
            #Normal step: add only new frame
            latest_frame = state[-1].copy()
            self.frames.append(latest_frame)
            self.frame_count += 1
        
        #Store transition with frame index
        max_priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        
        transition = (self.frame_count - 1, action, reward, done)
        
        if len(self.transitions) < self.capacity:
            self.transitions.append(transition)
        else:
            self.transitions[self.position] = transition
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def _get_state(self, frame_idx):
        """Reconstruct 4-frame stack from frame index"""
        #Convert absolute frame_idx to deque position
        oldest_frame_idx = self.frame_count - len(self.frames)
        
        #Get 4 consecutive frames ending at frame_idx
        frames = []
        for i in range(4):
            abs_idx = frame_idx - (3 - i)
            
            #Convert to deque position
            deque_pos = abs_idx - oldest_frame_idx
            
            if deque_pos < 0 or abs_idx < 0:
                #Frame no longer available - use oldest frame
                frames.append(self.frames[0])
            else:
                frames.append(self.frames[deque_pos])
        
        return np.array(frames, dtype=np.float32)
    
    def sample(self, batch_size, beta=0.4):
        """Sample batch according to priorities"""
        if self.size == 0:
            return None, None, None
        
        #Calculate oldest frame still in deque
        oldest_frame_idx = self.frame_count - len(self.frames)
        
        #Get valid indices where frames haven't been overwritten
        valid_mask = np.zeros(self.size, dtype=bool)
        
        for i in range(self.size):
            frame_idx = self.transitions[i][0]
            #Check if frames still exist
            if frame_idx >= oldest_frame_idx and frame_idx + 1 < self.frame_count:
                valid_mask[i] = True
        
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) == 0:
            return None, None, None
        
        #Calculate sampling probabilities for valid indices
        priorities = self.priorities[valid_indices]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        #Sample from valid indices
        sampled_valid_indices = np.random.choice(
            len(valid_indices), 
            min(batch_size, len(valid_indices)), 
            p=probabilities, 
            replace=False
        )
        indices = valid_indices[sampled_valid_indices]
        
        #Reconstruct states from frame indices
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for idx in indices:
            frame_idx, action, reward, done = self.transitions[idx]
            
            #Reconstruct current state
            state = self._get_state(frame_idx)
            states.append(state)
            
            #Reconstruct next state
            next_state = self._get_state(frame_idx + 1)
            next_states.append(next_state)
            
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
        
        #Calculate importance sampling weights
        total = len(valid_indices)
        sampled_probs = probabilities[sampled_valid_indices]
        weights = (total * sampled_probs) ** (-beta)
        weights /= weights.max()
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        ), indices, np.array(weights, dtype=np.float32)
    
    def update_priorities(self, indices, priorities):
        """Update priorities for sampled experiences"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority + 1e-6
    
    def __len__(self):
        return self.size