#!/usr/bin/env python3
"""
H-Module: High-level reasoning module for HRM architecture
Based on "Hierarchical Reasoning Model" paper - Sapient Intelligence

Implements abstract harmonic planning with slow timescale updates
and memory persistence for global context integration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List


class HModule(nn.Module):
    """
    High-level reasoning module for abstract harmonic planning.
    
    Key characteristics:
    - Slow updates (every N L-Module cycles)
    - Persistent memory via LSTM
    - Global context generation for L-Module guidance
    - Multi-cycle L-output aggregation
    """
    
    def __init__(self,
                 l_output_dim: int = 256,
                 h_hidden_dim: int = 128,
                 memory_depth: int = 10,
                 dropout: float = 0.1):
        super().__init__()
        
        self.l_output_dim = l_output_dim
        self.h_hidden_dim = h_hidden_dim
        self.memory_depth = memory_depth
        
        # L-Module output aggregator
        self.l_aggregator = nn.Sequential(
            nn.Linear(l_output_dim * memory_depth, h_hidden_dim * 2),
            nn.LayerNorm(h_hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h_hidden_dim * 2, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Long-term memory with LSTM for hierarchical integration
        self.memory_lstm = nn.LSTMCell(h_hidden_dim, h_hidden_dim)
        
        # Context generator for L-Module guidance
        self.context_generator = nn.Sequential(
            nn.Linear(h_hidden_dim, h_hidden_dim),
            nn.LayerNorm(h_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(h_hidden_dim, h_hidden_dim),
            nn.Tanh()  # Bounded context for stability
        )
        
        # Harmonic reasoning enhancement
        self.harmonic_attention = nn.MultiheadAttention(
            embed_dim=h_hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Initialize memory buffer for L-outputs
        self.l_output_buffer = []
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters following HRM paper recommendations"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    # Xavier uniform for better gradient flow
                    nn.init.xavier_uniform_(param, gain=1.0)
                else:
                    nn.init.zeros_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, 
                l_outputs_sequence: List[torch.Tensor],
                h_state: Optional[torch.Tensor] = None,
                h_cell: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for H-Module hierarchical reasoning.
        
        Args:
            l_outputs_sequence: List of L-Module outputs [(batch, l_output_dim)]
            h_state: Previous H-Module state (batch, h_hidden_dim)
            h_cell: Previous LSTM cell state (batch, h_hidden_dim)
            
        Returns:
            h_context: Context for next L-Module cycle (batch, h_hidden_dim)
            new_h_state: Updated H-Module state (batch, h_hidden_dim) 
            new_h_cell: Updated LSTM cell state (batch, h_hidden_dim)
        """
        batch_size = l_outputs_sequence[0].shape[0]
        device = l_outputs_sequence[0].device
        
        # Initialize states if None
        if h_state is None:
            h_state = torch.zeros(batch_size, self.h_hidden_dim, device=device)
        if h_cell is None:
            h_cell = torch.zeros(batch_size, self.h_hidden_dim, device=device)
        
        # Update L-output buffer with new sequence
        self.l_output_buffer.extend(l_outputs_sequence)
        
        # Maintain fixed memory depth
        if len(self.l_output_buffer) > self.memory_depth:
            self.l_output_buffer = self.l_output_buffer[-self.memory_depth:]
        
        # Pad with zeros if insufficient history
        current_buffer = self._pad_l_buffer(batch_size, device)
        
        # Aggregate L-Module outputs
        l_summary = self._aggregate_l_outputs(current_buffer)
        
        # Self-attention for harmonic reasoning enhancement
        l_summary_enhanced = self._apply_harmonic_attention(l_summary)
        
        # Update long-term memory with LSTM
        new_h_state, new_h_cell = self.memory_lstm(l_summary_enhanced, (h_state, h_cell))
        
        # Generate context for next L-Module cycle
        h_context = self.context_generator(new_h_state)
        
        return h_context, new_h_state, new_h_cell
    
    def _pad_l_buffer(self, batch_size: int, device: torch.device) -> List[torch.Tensor]:
        """Pad L-output buffer to memory_depth with zeros"""
        current_buffer = list(self.l_output_buffer)
        
        if len(current_buffer) < self.memory_depth:
            # Create zero padding
            zero_output = torch.zeros(batch_size, self.l_output_dim, device=device)
            padding_needed = self.memory_depth - len(current_buffer)
            current_buffer = [zero_output] * padding_needed + current_buffer
        
        return current_buffer
    
    def _aggregate_l_outputs(self, l_outputs: List[torch.Tensor]) -> torch.Tensor:
        """Aggregate L-Module outputs through learned transformation"""
        # Concatenate all L-outputs: (batch, memory_depth * l_output_dim)
        concatenated = torch.cat(l_outputs, dim=1)
        
        # Apply learned aggregation
        aggregated = self.l_aggregator(concatenated)
        
        return aggregated
    
    def _apply_harmonic_attention(self, l_summary: torch.Tensor) -> torch.Tensor:
        """Apply self-attention for harmonic reasoning enhancement"""
        # Add sequence dimension for attention: (batch, 1, h_hidden_dim)
        l_summary_seq = l_summary.unsqueeze(1)
        
        # Self-attention (attending to itself for internal reasoning)
        attended, _ = self.harmonic_attention(
            l_summary_seq, l_summary_seq, l_summary_seq
        )
        
        # Remove sequence dimension: (batch, h_hidden_dim)
        return attended.squeeze(1)
    
    def reset_l_buffer(self):
        """Reset L-Module output buffer (called after H-Module processing)"""
        self.l_output_buffer = []
    
    def get_memory_state(self) -> dict:
        """Get current memory state for debugging/analysis"""
        return {
            'buffer_length': len(self.l_output_buffer),
            'memory_depth': self.memory_depth,
            'h_hidden_dim': self.h_hidden_dim
        }


class AdaptiveHModule(HModule):
    """
    Enhanced H-Module with adaptive memory depth based on harmonic complexity
    """
    
    def __init__(self, 
                 l_output_dim: int = 256,
                 h_hidden_dim: int = 128,
                 base_memory_depth: int = 10,
                 max_memory_depth: int = 20,
                 complexity_threshold: float = 0.7,
                 **kwargs):
        super().__init__(l_output_dim, h_hidden_dim, base_memory_depth, **kwargs)
        
        self.base_memory_depth = base_memory_depth
        self.max_memory_depth = max_memory_depth
        self.complexity_threshold = complexity_threshold
        
        # Harmonic complexity estimator
        self.complexity_estimator = nn.Sequential(
            nn.Linear(l_output_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, 
                l_outputs_sequence: List[torch.Tensor],
                h_state: Optional[torch.Tensor] = None,
                h_cell: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Enhanced forward with adaptive memory depth"""
        
        # Estimate harmonic complexity from recent L-outputs
        if l_outputs_sequence:
            recent_output = l_outputs_sequence[-1]  # Most recent L-output
            complexity_score = self.complexity_estimator(recent_output).mean().item()
            
            # Adapt memory depth based on complexity
            if complexity_score > self.complexity_threshold:
                self.memory_depth = min(self.max_memory_depth, 
                                      int(self.base_memory_depth * (1 + complexity_score)))
            else:
                self.memory_depth = self.base_memory_depth
        
        # Call parent forward method
        return super().forward(l_outputs_sequence, h_state, h_cell)


# Factory function for easy instantiation
def create_h_module(config: dict = None) -> HModule:
    """
    Factory function to create H-Module with configuration
    
    Args:
        config: Configuration dictionary with module parameters
        
    Returns:
        Configured H-Module instance
    """
    if config is None:
        config = {}
    
    # Default configuration based on HRM paper
    default_config = {
        'l_output_dim': 256,
        'h_hidden_dim': 128, 
        'memory_depth': 10,
        'dropout': 0.1
    }
    
    # Merge with provided config
    final_config = {**default_config, **config}
    
    # Choose module type
    if final_config.get('adaptive_memory', False):
        return AdaptiveHModule(**final_config)
    else:
        return HModule(**final_config)


if __name__ == "__main__":
    # Test H-Module implementation
    print("🧠 Testing H-Module Implementation")
    
    # Create test H-Module
    h_module = create_h_module({
        'l_output_dim': 256,
        'h_hidden_dim': 128,
        'memory_depth': 5
    })
    
    # Test data
    batch_size = 2
    l_outputs = [torch.randn(batch_size, 256) for _ in range(3)]
    
    # Forward pass
    print(f"Input: {len(l_outputs)} L-outputs of shape {l_outputs[0].shape}")
    h_context, h_state, h_cell = h_module(l_outputs)
    
    print(f"Output context shape: {h_context.shape}")
    print(f"Output state shape: {h_state.shape}")
    print(f"Output cell shape: {h_cell.shape}")
    
    # Memory state
    memory_info = h_module.get_memory_state()
    print(f"Memory state: {memory_info}")
    
    print("✅ H-Module test completed successfully")