#!/usr/bin/env python3
"""
Hierarchical Convergence: Core mechanism for HRM architecture
Based on "Hierarchical Reasoning Model" paper - Sapient Intelligence

Implements the key innovation of HRM: hierarchical convergence where
L-Module converges locally while H-Module provides global guidance,
with periodic reset for computational depth without memory explosion.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict, Any
import math

from .h_module import HModule
from .l_module import LModule


class HierarchicalConvergence(nn.Module):
    """
    Hierarchical Convergence mechanism implementing the core HRM innovation.
    
    Key features:
    - L-Module: Fast local convergence (T steps)
    - H-Module: Slow global updates (every N cycles)
    - Reset mechanism: L-Module state reset after H-Module read
    - O(1) memory: Constant memory usage vs O(T) in standard RNNs
    """
    
    def __init__(self,
                 input_dim: int = 512 * 3,
                 l_hidden_dim: int = 256,
                 h_hidden_dim: int = 128,
                 N: int = 4,  # High-level cycles
                 T: int = 8,  # Low-level steps per cycle
                 convergence_threshold: float = 1e-4):
        super().__init__()
        
        self.input_dim = input_dim
        self.l_hidden_dim = l_hidden_dim
        self.h_hidden_dim = h_hidden_dim
        self.N = N  # Number of high-level cycles
        self.T = T  # Number of low-level steps per cycle
        self.convergence_threshold = convergence_threshold
        
        # Initialize modules
        self.l_module = LModule(
            input_dim=input_dim,
            hidden_dim=l_hidden_dim,
            h_context_dim=h_hidden_dim
        )
        
        self.h_module = HModule(
            l_output_dim=l_hidden_dim,
            h_hidden_dim=h_hidden_dim,
            memory_depth=T  # Remember T L-Module outputs
        )
        
        # Convergence monitoring
        self.register_buffer('convergence_history', 
                           torch.zeros(100))  # Track last 100 convergence measures
        self.convergence_step = 0
        
    def forward(self, 
                x: torch.Tensor,
                debug_mode: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Hierarchical convergence forward pass.
        
        Args:
            x: Input histogram (batch, 512, 3) 
            debug_mode: Return detailed debugging information
            
        Returns:
            final_output: Final H-Module state (batch, h_hidden_dim)
            debug_info: Dictionary with convergence information if debug_mode=True
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize states
        h_state = None
        h_cell = None
        h_context = torch.zeros(batch_size, self.h_hidden_dim, device=device)
        
        # Debug tracking
        debug_info = {
            'convergence_measures': [],
            'l_outputs': [],
            'h_contexts': [],
            'reset_points': []
        } if debug_mode else {}
        
        # Hierarchical convergence loop
        for cycle in range(self.N):
            # L-Module: Fast local processing for T steps
            l_outputs_cycle, convergence_measure = self._l_module_cycle(
                x, h_context, debug_mode
            )
            
            if debug_mode:
                debug_info['convergence_measures'].append(convergence_measure)
                debug_info['l_outputs'].extend(l_outputs_cycle)
                debug_info['h_contexts'].append(h_context.clone())
            
            # H-Module: Slow global update
            h_context, h_state, h_cell = self.h_module(
                l_outputs_cycle, h_state, h_cell
            )
            
            # Reset L-Module state (key HRM innovation)
            self.h_module.reset_l_buffer()
            
            if debug_mode:
                debug_info['reset_points'].append(cycle * self.T + self.T)
        
        return h_context, debug_info
    
    def _l_module_cycle(self, 
                       x: torch.Tensor,
                       h_context: torch.Tensor,
                       debug_mode: bool = False) -> Tuple[List[torch.Tensor], float]:
        """
        Execute one L-Module cycle with convergence monitoring.
        
        Returns:
            l_outputs: List of L-Module outputs for this cycle
            convergence_measure: Measure of how well L-Module converged
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Initialize L-Module state
        l_state = self.l_module.reset_state(batch_size, device)
        
        l_outputs = []
        previous_output = None
        convergence_deltas = []
        
        # T fast update steps
        for step in range(self.T):
            # L-Module forward pass
            l_output, l_state = self.l_module(x, h_context, l_state)
            l_outputs.append(l_output)
            
            # Monitor convergence
            if previous_output is not None:
                delta = torch.norm(l_output - previous_output, dim=1).mean().item()
                convergence_deltas.append(delta)
            
            previous_output = l_output
        
        # Calculate convergence measure
        if convergence_deltas:
            convergence_measure = sum(convergence_deltas) / len(convergence_deltas)
            self._update_convergence_history(convergence_measure)
        else:
            convergence_measure = float('inf')
        
        return l_outputs, convergence_measure
    
    def _update_convergence_history(self, measure: float):
        """Update convergence history buffer"""
        self.convergence_history[self.convergence_step % 100] = measure
        self.convergence_step += 1
    
    def get_convergence_stats(self) -> Dict[str, float]:
        """Get convergence statistics"""
        valid_measures = self.convergence_history[self.convergence_history != 0]
        
        if len(valid_measures) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        return {
            'mean': valid_measures.mean().item(),
            'std': valid_measures.std().item(),
            'min': valid_measures.min().item(),
            'max': valid_measures.max().item()
        }
    
    def is_converged(self) -> bool:
        """Check if system has reached stable convergence"""
        stats = self.get_convergence_stats()
        return stats['mean'] < self.convergence_threshold


class AdaptiveHierarchicalConvergence(HierarchicalConvergence):
    """
    Adaptive version that adjusts N and T based on convergence behavior
    """
    
    def __init__(self,
                 input_dim: int = 512 * 3,
                 l_hidden_dim: int = 256,
                 h_hidden_dim: int = 128,
                 base_N: int = 4,
                 base_T: int = 8,
                 max_N: int = 8,
                 max_T: int = 16,
                 adaptation_rate: float = 0.1):
        super().__init__(input_dim, l_hidden_dim, h_hidden_dim, base_N, base_T)
        
        self.base_N = base_N
        self.base_T = base_T
        self.max_N = max_N
        self.max_T = max_T
        self.adaptation_rate = adaptation_rate
        
        # Adaptive parameters (learnable)
        self.N_adapter = nn.Parameter(torch.tensor(0.0))  # Will be sigmoidized
        self.T_adapter = nn.Parameter(torch.tensor(0.0))  # Will be sigmoidized
    
    def get_adaptive_NT(self) -> Tuple[int, int]:
        """Get current adaptive N and T values"""
        # Sigmoid to [0,1], then scale to [base, max]
        N_factor = torch.sigmoid(self.N_adapter).item()
        T_factor = torch.sigmoid(self.T_adapter).item()
        
        adaptive_N = int(self.base_N + (self.max_N - self.base_N) * N_factor)
        adaptive_T = int(self.base_T + (self.max_T - self.base_T) * T_factor)
        
        return adaptive_N, adaptive_T
    
    def forward(self, x: torch.Tensor, debug_mode: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward with adaptive N and T"""
        # Update N and T based on learned adaptation
        self.N, self.T = self.get_adaptive_NT()
        
        # Update H-Module memory depth to match T
        self.h_module.memory_depth = self.T
        
        return super().forward(x, debug_mode)


class ResidualHierarchicalConvergence(HierarchicalConvergence):
    """
    Version with residual connections for improved gradient flow
    """
    
    def __init__(self, 
                 input_dim: int = 512 * 3,
                 l_hidden_dim: int = 256,
                 h_hidden_dim: int = 128,
                 **kwargs):
        super().__init__(input_dim, l_hidden_dim, h_hidden_dim, **kwargs)
        
        # Residual projection layers
        self.l_residual_proj = nn.Linear(input_dim, l_hidden_dim)
        self.h_residual_proj = nn.Linear(l_hidden_dim, h_hidden_dim)
        
        # Layer normalization for stability
        self.l_layer_norm = nn.LayerNorm(l_hidden_dim)
        self.h_layer_norm = nn.LayerNorm(h_hidden_dim)
    
    def _l_module_cycle(self, 
                       x: torch.Tensor,
                       h_context: torch.Tensor,
                       debug_mode: bool = False) -> Tuple[List[torch.Tensor], float]:
        """L-Module cycle with residual connections"""
        
        # Get base L-Module outputs
        l_outputs, convergence_measure = super()._l_module_cycle(x, h_context, debug_mode)
        
        # Add residual connection from input
        x_residual = self.l_residual_proj(x.view(x.shape[0], -1))
        
        # Apply residual to each L-Module output
        l_outputs_residual = []
        for l_output in l_outputs:
            residual_output = self.l_layer_norm(l_output + x_residual)
            l_outputs_residual.append(residual_output)
        
        return l_outputs_residual, convergence_measure
    
    def forward(self, x: torch.Tensor, debug_mode: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Forward with H-Module residual connection"""
        h_output, debug_info = super().forward(x, debug_mode)
        
        # Add residual connection to H-Module output
        # Use mean of L-outputs as residual (aggregate input information)
        if debug_info and 'l_outputs' in debug_info:
            l_outputs_mean = torch.stack(debug_info['l_outputs']).mean(0)
            h_residual = self.h_residual_proj(l_outputs_mean)
            h_output = self.h_layer_norm(h_output + h_residual)
        
        return h_output, debug_info


# Factory function
def create_hierarchical_convergence(config: dict = None) -> HierarchicalConvergence:
    """
    Factory function to create Hierarchical Convergence with configuration
    """
    if config is None:
        config = {}
    
    # Default configuration
    default_config = {
        'input_dim': 512 * 3,
        'l_hidden_dim': 256,
        'h_hidden_dim': 128,
        'N': 4,
        'T': 8,
        'convergence_threshold': 1e-4
    }
    
    # Merge configurations
    final_config = {**default_config, **config}
    
    # Choose convergence type
    convergence_type = final_config.pop('convergence_type', 'standard')
    
    if convergence_type == 'adaptive':
        return AdaptiveHierarchicalConvergence(**final_config)
    elif convergence_type == 'residual':
        return ResidualHierarchicalConvergence(**final_config)
    else:
        return HierarchicalConvergence(**final_config)


if __name__ == "__main__":
    # Test Hierarchical Convergence
    print("🔄 Testing Hierarchical Convergence Implementation")
    
    # Create test system
    convergence_system = create_hierarchical_convergence({
        'N': 2,  # Reduced for testing
        'T': 4,  # Reduced for testing
        'convergence_type': 'standard'
    })
    
    # Test data
    batch_size = 2
    x = torch.randn(batch_size, 512, 3)
    
    print(f"Input shape: {x.shape}")
    print(f"System config: N={convergence_system.N}, T={convergence_system.T}")
    
    # Forward pass with debug
    final_output, debug_info = convergence_system(x, debug_mode=True)
    
    print(f"Final output shape: {final_output.shape}")
    print(f"Convergence measures: {debug_info['convergence_measures']}")
    print(f"Reset points: {debug_info['reset_points']}")
    
    # Convergence stats
    conv_stats = convergence_system.get_convergence_stats()
    print(f"Convergence stats: {conv_stats}")
    print(f"Is converged: {convergence_system.is_converged()}")
    
    # Test adaptive version
    print("\n🎛️ Testing Adaptive Hierarchical Convergence")
    adaptive_system = create_hierarchical_convergence({
        'convergence_type': 'adaptive',
        'base_N': 2,
        'base_T': 4,
        'max_N': 4,
        'max_T': 8
    })
    
    adaptive_N, adaptive_T = adaptive_system.get_adaptive_NT()
    print(f"Adaptive N: {adaptive_N}, T: {adaptive_T}")
    
    adaptive_output, _ = adaptive_system(x)
    print(f"Adaptive output shape: {adaptive_output.shape}")
    
    print("✅ Hierarchical Convergence tests completed successfully")