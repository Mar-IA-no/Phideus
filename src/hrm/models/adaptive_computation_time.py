#!/usr/bin/env python3
"""
Adaptive Computation Time (ACT): Q-learning mechanism for HRM
Based on "Hierarchical Reasoning Model" paper - Sapient Intelligence

Implements ACT mechanism that dynamically determines when H-Module
should process based on harmonic complexity, using Q-learning for
optimal computational resource allocation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional
import numpy as np
from collections import deque
import random


class AdaptiveComputationTime(nn.Module):
    """
    ACT mechanism with Q-learning for dynamic H-Module activation.
    
    Key features:
    - Q-learning to decide halt vs continue
    - Complexity-based halting probability
    - Reward signal from prediction correctness
    - Experience replay for stable learning
    """
    
    def __init__(self,
                 l_output_dim: int = 256,
                 q_hidden_dim: int = 64,
                 halting_threshold: float = 0.99,
                 max_steps: int = 20,
                 min_steps: int = 2,
                 epsilon: float = 0.1,
                 gamma: float = 0.95,
                 learning_rate: float = 1e-4):
        super().__init__()
        
        self.l_output_dim = l_output_dim
        self.q_hidden_dim = q_hidden_dim
        self.halting_threshold = halting_threshold
        self.max_steps = max_steps
        self.min_steps = min_steps
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma      # Discount factor
        
        # Q-network for halting decisions
        self.q_network = nn.Sequential(
            nn.Linear(l_output_dim, q_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(q_hidden_dim, q_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(q_hidden_dim, 2)  # [halt, continue] Q-values
        )
        
        # Target network for stable Q-learning
        self.target_q_network = nn.Sequential(
            nn.Linear(l_output_dim, q_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(q_hidden_dim, q_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(q_hidden_dim, 2)
        )
        
        # Copy weights to target network
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        
        # Complexity predictor for auxiliary reward
        self.complexity_predictor = nn.Sequential(
            nn.Linear(l_output_dim, q_hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(q_hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Experience replay buffer
        self.replay_buffer = deque(maxlen=10000)
        self.batch_size = 32
        
        # Q-learning optimizer
        self.optimizer = torch.optim.Adam(
            list(self.q_network.parameters()) + list(self.complexity_predictor.parameters()),
            lr=learning_rate
        )
        
        # Target network update frequency
        self.target_update_freq = 100
        self.update_counter = 0
        
        # Statistics tracking
        self.register_buffer('halting_stats', torch.zeros(4))  # [total, halt, continue, avg_steps]
        
    def forward(self, 
                l_output: torch.Tensor,
                step_count: int,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        ACT forward pass with Q-learning decision making.
        
        Args:
            l_output: L-Module output (batch, l_output_dim)
            step_count: Number of steps since last H-Module activation
            training: Whether in training mode
            
        Returns:
            should_halt: Boolean mask for halting decision (batch,)
            halt_probabilities: Raw halting probabilities (batch,)
            q_values: Q-values for [halt, continue] (batch, 2)
        """
        batch_size = l_output.shape[0]
        
        # Get Q-values for current state
        q_values = self.q_network(l_output)  # (batch, 2)
        halt_q, continue_q = q_values[:, 0], q_values[:, 1]
        
        # Predict complexity for auxiliary reward
        complexity_score = self.complexity_predictor(l_output).squeeze(-1)  # (batch,)
        
        # Epsilon-greedy action selection during training
        if training and random.random() < self.epsilon:
            # Random action
            should_halt = torch.randint(0, 2, (batch_size,), dtype=torch.bool, device=l_output.device)
        else:
            # Greedy action based on Q-values
            should_halt = halt_q > continue_q
        
        # Force halting conditions
        forced_halt = step_count >= self.max_steps
        forced_continue = step_count < self.min_steps
        
        # Apply constraints
        should_halt = (should_halt & ~forced_continue) | forced_halt
        
        # Convert to probabilities for interpretability
        halt_probabilities = torch.sigmoid(halt_q - continue_q)
        
        # Update statistics
        self._update_statistics(should_halt, step_count)
        
        return should_halt, halt_probabilities, q_values
    
    def store_experience(self,
                        state: torch.Tensor,
                        action: torch.Tensor,
                        reward: torch.Tensor,
                        next_state: Optional[torch.Tensor] = None,
                        done: torch.Tensor = None):
        """
        Store experience in replay buffer for Q-learning.
        
        Args:
            state: L-Module output state
            action: Halting action (0=continue, 1=halt)
            reward: Reward signal
            next_state: Next L-Module output (None if episode ended)
            done: Episode termination flag
        """
        if next_state is None:
            next_state = torch.zeros_like(state)
        if done is None:
            done = torch.ones(state.shape[0], dtype=torch.bool)
        
        # Store each sample in the batch
        for i in range(state.shape[0]):
            experience = (
                state[i].detach(),
                action[i].detach(),
                reward[i].detach(),
                next_state[i].detach(),
                done[i].detach()
            )
            self.replay_buffer.append(experience)
    
    def update_q_network(self):
        """
        Update Q-network using experience replay.
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch from replay buffer
        batch = random.sample(self.replay_buffer, self.batch_size)
        
        states = torch.stack([exp[0] for exp in batch])
        actions = torch.stack([exp[1] for exp in batch])
        rewards = torch.stack([exp[2] for exp in batch])
        next_states = torch.stack([exp[3] for exp in batch])
        dones = torch.stack([exp[4] for exp in batch])
        
        # Current Q-values
        current_q_values = self.q_network(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_q_network(next_states)
            next_q = next_q_values.max(1)[0]
            target_q = rewards + self.gamma * next_q * (~dones).float()
        
        # Q-learning loss
        q_loss = F.mse_loss(current_q, target_q)
        
        # Update networks
        self.optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network periodically
        self.update_counter += 1
        if self.update_counter % self.target_update_freq == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
    
    def compute_reward(self,
                      predicted_output: torch.Tensor,
                      target_output: torch.Tensor,
                      complexity_bonus: torch.Tensor,
                      computational_cost: float = 0.01) -> torch.Tensor:
        """
        Compute reward signal for Q-learning.
        
        Args:
            predicted_output: Model prediction
            target_output: Ground truth
            complexity_bonus: Complexity score from predictor
            computational_cost: Cost per computation step
            
        Returns:
            reward: Reward signal (batch,)
        """
        # Accuracy reward (negative loss)
        accuracy_reward = -F.mse_loss(predicted_output, target_output, reduction='none').mean(1)
        
        # Efficiency bonus (higher complexity = higher bonus for halting)
        efficiency_bonus = complexity_bonus * 0.1
        
        # Computational cost penalty
        cost_penalty = -computational_cost
        
        # Combine rewards
        total_reward = accuracy_reward + efficiency_bonus + cost_penalty
        
        return total_reward
    
    def _update_statistics(self, should_halt: torch.Tensor, step_count: int):
        """Update halting statistics"""
        batch_size = should_halt.shape[0]
        
        self.halting_stats[0] += batch_size  # Total samples
        self.halting_stats[1] += should_halt.sum().item()  # Halt count
        self.halting_stats[2] += (~should_halt).sum().item()  # Continue count
        self.halting_stats[3] = (self.halting_stats[3] * 0.99 + step_count * 0.01)  # Avg steps
    
    def get_statistics(self) -> Dict[str, float]:
        """Get ACT statistics"""
        total = self.halting_stats[0].item()
        if total == 0:
            return {'halt_rate': 0.0, 'continue_rate': 0.0, 'avg_steps': 0.0}
        
        return {
            'halt_rate': self.halting_stats[1].item() / total,
            'continue_rate': self.halting_stats[2].item() / total,
            'avg_steps': self.halting_stats[3].item()
        }
    
    def reset_statistics(self):
        """Reset statistics counters"""
        self.halting_stats.fill_(0)


class EnhancedACT(AdaptiveComputationTime):
    """
    Enhanced ACT with additional features for better harmonic analysis
    """
    
    def __init__(self,
                 l_output_dim: int = 256,
                 **kwargs):
        super().__init__(l_output_dim, **kwargs)
        
        # Harmonic complexity analyzer
        self.harmonic_analyzer = nn.Sequential(
            nn.Linear(l_output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),  # Harmonic feature dimensions
            nn.Sigmoid()
        )
        
        # Temporal consistency tracker
        self.temporal_tracker = nn.LSTMCell(l_output_dim, 64)
        self.temporal_state = None
        self.temporal_cell = None
        
        # Multi-objective reward computation
        self.reward_weights = nn.Parameter(torch.tensor([1.0, 0.5, 0.3, 0.2]))
        
    def forward(self, 
                l_output: torch.Tensor,
                step_count: int,
                training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enhanced forward with harmonic and temporal analysis"""
        
        # Harmonic analysis
        harmonic_features = self.harmonic_analyzer(l_output)
        
        # Temporal consistency tracking
        batch_size = l_output.shape[0]
        device = l_output.device
        
        if self.temporal_state is None:
            self.temporal_state = torch.zeros(batch_size, 64, device=device)
            self.temporal_cell = torch.zeros(batch_size, 64, device=device)
        
        self.temporal_state, self.temporal_cell = self.temporal_tracker(
            l_output, (self.temporal_state, self.temporal_cell)
        )
        
        # Enhanced state representation
        enhanced_state = torch.cat([l_output, self.temporal_state], dim=1)
        
        # Enhanced Q-network input
        enhanced_q_values = self.q_network(enhanced_state[:, :self.l_output_dim])  # Keep compatible
        
        # Enhanced complexity prediction
        complexity_score = self.complexity_predictor(l_output).squeeze(-1)
        harmonic_complexity = harmonic_features.mean(dim=1)
        combined_complexity = (complexity_score + harmonic_complexity) / 2
        
        # Decision making (same as parent but with enhanced features)
        halt_q, continue_q = enhanced_q_values[:, 0], enhanced_q_values[:, 1]
        
        if training and random.random() < self.epsilon:
            should_halt = torch.randint(0, 2, (batch_size,), dtype=torch.bool, device=device)
        else:
            # Consider both Q-values and harmonic complexity
            halt_preference = halt_q - continue_q + combined_complexity * 0.1
            should_halt = halt_preference > 0
        
        # Apply constraints
        forced_halt = step_count >= self.max_steps
        forced_continue = step_count < self.min_steps
        should_halt = (should_halt & ~forced_continue) | forced_halt
        
        halt_probabilities = torch.sigmoid(halt_q - continue_q + combined_complexity * 0.1)
        
        self._update_statistics(should_halt, step_count)
        
        return should_halt, halt_probabilities, enhanced_q_values
    
    def compute_enhanced_reward(self,
                               predicted_output: torch.Tensor,
                               target_output: torch.Tensor,
                               harmonic_accuracy: torch.Tensor,
                               temporal_consistency: torch.Tensor,
                               computational_efficiency: torch.Tensor) -> torch.Tensor:
        """Enhanced reward computation with multiple objectives"""
        
        # Multiple reward components
        rewards = torch.stack([
            -F.mse_loss(predicted_output, target_output, reduction='none').mean(1),  # Accuracy
            harmonic_accuracy,  # Harmonic analysis quality
            temporal_consistency,  # Temporal coherence
            computational_efficiency  # Efficiency bonus
        ], dim=1)
        
        # Weighted combination
        weighted_rewards = rewards * F.softmax(self.reward_weights, dim=0)
        total_reward = weighted_rewards.sum(dim=1)
        
        return total_reward


# Factory function
def create_act_module(config: dict = None) -> AdaptiveComputationTime:
    """
    Factory function to create ACT module with configuration
    """
    if config is None:
        config = {}
    
    default_config = {
        'l_output_dim': 256,
        'q_hidden_dim': 64,
        'halting_threshold': 0.99,
        'max_steps': 20,
        'min_steps': 2,
        'epsilon': 0.1,
        'gamma': 0.95,
        'learning_rate': 1e-4
    }
    
    final_config = {**default_config, **config}
    
    # Choose ACT type
    act_type = final_config.pop('act_type', 'standard')
    
    if act_type == 'enhanced':
        return EnhancedACT(**final_config)
    else:
        return AdaptiveComputationTime(**final_config)


if __name__ == "__main__":
    # Test ACT implementation
    print("⏱️ Testing Adaptive Computation Time Implementation")
    
    # Create ACT module
    act = create_act_module({
        'l_output_dim': 256,
        'max_steps': 10,
        'min_steps': 2
    })
    
    # Test data
    batch_size = 4
    l_output = torch.randn(batch_size, 256)
    
    print(f"Input L-output shape: {l_output.shape}")
    
    # Test forward pass
    should_halt, halt_probs, q_values = act(l_output, step_count=5, training=True)
    
    print(f"Should halt: {should_halt}")
    print(f"Halt probabilities: {halt_probs}")
    print(f"Q-values shape: {q_values.shape}")
    
    # Test experience storage and Q-learning update
    actions = should_halt.long()
    rewards = torch.randn(batch_size)
    next_l_output = torch.randn(batch_size, 256)
    done = torch.tensor([False, True, False, True])
    
    act.store_experience(l_output, actions, rewards, next_l_output, done)
    print(f"Replay buffer size: {len(act.replay_buffer)}")
    
    # Fill replay buffer for testing
    for _ in range(50):
        test_state = torch.randn(batch_size, 256)
        test_action = torch.randint(0, 2, (batch_size,))
        test_reward = torch.randn(batch_size)
        test_next = torch.randn(batch_size, 256)
        test_done = torch.randint(0, 2, (batch_size,)).bool()
        act.store_experience(test_state, test_action, test_reward, test_next, test_done)
    
    # Test Q-learning update
    act.update_q_network()
    print("Q-network updated successfully")
    
    # Test statistics
    stats = act.get_statistics()
    print(f"ACT Statistics: {stats}")
    
    # Test enhanced version
    print("\n🔬 Testing Enhanced ACT")
    enhanced_act = create_act_module({
        'act_type': 'enhanced',
        'l_output_dim': 256
    })
    
    enhanced_halt, enhanced_probs, enhanced_q = enhanced_act(
        l_output, step_count=3, training=True
    )
    print(f"Enhanced halt decisions: {enhanced_halt}")
    print(f"Enhanced halt probabilities: {enhanced_probs}")
    
    print("✅ ACT tests completed successfully")