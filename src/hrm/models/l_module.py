#!/usr/bin/env python3
"""
L-Module: Low-level computation module for HRM architecture
Based on "Hierarchical Reasoning Model" paper - Sapient Intelligence

Implements fast spectral computation with high temporal resolution
and rapid response to harmonic features in input histograms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LModule(nn.Module):
    """
    Low-level computation module for fast spectral processing.
    
    Key characteristics:
    - Fast updates (every frame/histogram)  
    - Local computation with H-Module context
    - High temporal resolution processing
    - Reset mechanism after H-Module reads state
    """
    
    def __init__(self,
                 input_dim: int = 512 * 3,  # Flattened histogram (512, 3)
                 hidden_dim: int = 256,
                 h_context_dim: int = 128,
                 num_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.h_context_dim = h_context_dim
        self.num_layers = num_layers
        
        # Histogram encoder for individual frame processing
        self.histogram_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # H-Module context integration
        self.context_integration = nn.Sequential(
            nn.Linear(hidden_dim + h_context_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Fast recurrent processing with GRU (faster than LSTM)
        self.recurrent_layers = nn.ModuleList([
            nn.GRUCell(hidden_dim if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        # Spectral feature enhancement
        self.spectral_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters for stable fast processing"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) > 1:
                    # He initialization for ReLU networks
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                else:
                    nn.init.zeros_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, 
                histogram: torch.Tensor,
                h_context: torch.Tensor,
                l_state: Optional[Tuple[torch.Tensor, ...]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Fast forward pass for L-Module computation.
        
        Args:
            histogram: Input histogram (batch, 512, 3)
            h_context: Context from H-Module (batch, h_context_dim)
            l_state: Previous L-Module states tuple of (batch, hidden_dim) for each layer
            
        Returns:
            l_output: L-Module output (batch, hidden_dim)
            new_l_state: Updated L-Module states tuple
        """
        batch_size = histogram.shape[0]
        device = histogram.device
        
        # Initialize L-Module states if None
        if l_state is None:
            l_state = tuple(
                torch.zeros(batch_size, self.hidden_dim, device=device)
                for _ in range(self.num_layers)
            )
        
        # Encode current histogram
        hist_flat = histogram.view(batch_size, -1)  # (batch, 512*3)
        hist_encoded = self.histogram_encoder(hist_flat)
        
        # Integrate with H-Module context
        combined_input = torch.cat([hist_encoded, h_context], dim=1)
        integrated = self.context_integration(combined_input)
        
        # Fast recurrent processing through multiple GRU layers
        current_input = integrated
        new_states = []
        
        for layer_idx, gru_cell in enumerate(self.recurrent_layers):
            new_state = gru_cell(current_input, l_state[layer_idx])
            new_states.append(new_state)
            current_input = new_state
        
        # Spectral attention enhancement
        enhanced_output = self._apply_spectral_attention(current_input)
        
        # Final output projection
        l_output = self.output_projection(enhanced_output)
        
        return l_output, tuple(new_states)
    
    def _apply_spectral_attention(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Apply self-attention for spectral feature enhancement"""
        # Add sequence dimension: (batch, 1, hidden_dim)
        hidden_seq = hidden_state.unsqueeze(1)
        
        # Self-attention for internal spectral reasoning
        attended, _ = self.spectral_attention(
            hidden_seq, hidden_seq, hidden_seq
        )
        
        # Remove sequence dimension and add residual connection
        enhanced = attended.squeeze(1) + hidden_state
        
        return enhanced
    
    def reset_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """
        Reset L-Module state (called by H-Module after reading)
        
        Returns:
            Fresh L-Module states tuple
        """
        return tuple(
            torch.zeros(batch_size, self.hidden_dim, device=device)
            for _ in range(self.num_layers)
        )
    
    def get_state_info(self, l_state: Tuple[torch.Tensor, ...]) -> dict:
        """Get L-Module state information for debugging"""
        if l_state is None:
            return {'num_layers': self.num_layers, 'state_initialized': False}
        
        return {
            'num_layers': len(l_state),
            'state_shapes': [state.shape for state in l_state],
            'state_norms': [state.norm().item() for state in l_state],
            'state_initialized': True
        }


class EnhancedLModule(LModule):
    """
    Enhanced L-Module with adaptive processing and harmonic feature detection
    """
    
    def __init__(self,
                 input_dim: int = 512 * 3,
                 hidden_dim: int = 256,
                 h_context_dim: int = 128,
                 **kwargs):
        super().__init__(input_dim, hidden_dim, h_context_dim, **kwargs)
        
        # Harmonic ratio detector for enhanced spectral analysis
        self.ratio_detector = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32),  # 32 most important harmonic ratios
            nn.Sigmoid()
        )
        
        # Ratio-aware feature integration
        self.ratio_integration = nn.Sequential(
            nn.Linear(hidden_dim + 32, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Temporal dynamics tracker
        self.temporal_tracker = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Current + previous
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Previous output buffer for temporal tracking
        self.previous_output = None
    
    def forward(self,
                histogram: torch.Tensor,
                h_context: torch.Tensor,
                l_state: Optional[Tuple[torch.Tensor, ...]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Enhanced forward pass with harmonic and temporal analysis"""
        
        # Call parent forward
        l_output, new_l_state = super().forward(histogram, h_context, l_state)
        
        # Extract harmonic ratio features
        hist_flat = histogram.view(histogram.shape[0], -1)
        ratio_features = self.ratio_detector(hist_flat)
        
        # Integrate ratio features
        ratio_enhanced = torch.cat([l_output, ratio_features], dim=1)
        ratio_integrated = self.ratio_integration(ratio_enhanced)
        
        # Temporal dynamics tracking
        if self.previous_output is not None and self.previous_output.shape == ratio_integrated.shape:
            temporal_input = torch.cat([ratio_integrated, self.previous_output], dim=1)
            temporal_enhanced = self.temporal_tracker(temporal_input)
        else:
            temporal_enhanced = ratio_integrated
        
        # Update previous output
        self.previous_output = ratio_integrated.detach()
        
        return temporal_enhanced, new_l_state
    
    def reset_state(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        """Enhanced reset that also clears temporal tracking"""
        # Reset temporal tracking
        self.previous_output = None
        
        # Call parent reset
        return super().reset_state(batch_size, device)


class SpectrumAwareLModule(LModule):
    """
    Spectrum-aware L-Module with frequency domain processing
    """
    
    def __init__(self,
                 input_dim: int = 512 * 3,
                 hidden_dim: int = 256,
                 h_context_dim: int = 128,
                 **kwargs):
        super().__init__(input_dim, hidden_dim, h_context_dim, **kwargs)
        
        # Frequency domain analysis
        self.freq_analyzer = nn.Sequential(
            nn.Conv1d(3, 32, kernel_size=15, padding=7, dilation=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, kernel_size=15, padding=14, dilation=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(hidden_dim)
        )
        
        # Spectral fusion
        self.spectral_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # Encoded + frequency features
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self,
                histogram: torch.Tensor,
                h_context: torch.Tensor,
                l_state: Optional[Tuple[torch.Tensor, ...]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """Forward pass with frequency domain analysis"""
        
        # Standard L-Module processing
        l_output, new_l_state = super().forward(histogram, h_context, l_state)
        
        # Frequency domain analysis
        # histogram shape: (batch, 512, 3) -> transpose for conv1d: (batch, 3, 512)
        freq_input = histogram.transpose(1, 2)
        freq_features = self.freq_analyzer(freq_input)  # (batch, hidden_dim)
        
        # Fuse spectral and encoded features
        fused_input = torch.cat([l_output, freq_features], dim=1)
        spectral_enhanced = self.spectral_fusion(fused_input)
        
        return spectral_enhanced, new_l_state


# Factory function for easy instantiation
def create_l_module(config: dict = None) -> LModule:
    """
    Factory function to create L-Module with configuration
    
    Args:
        config: Configuration dictionary with module parameters
        
    Returns:
        Configured L-Module instance
    """
    if config is None:
        config = {}
    
    # Default configuration based on HRM paper
    default_config = {
        'input_dim': 512 * 3,
        'hidden_dim': 256,
        'h_context_dim': 128,
        'num_layers': 2,
        'dropout': 0.1
    }
    
    # Merge with provided config
    final_config = {**default_config, **config}
    
    # Choose module type
    module_type = final_config.pop('module_type', 'standard')
    
    if module_type == 'enhanced':
        return EnhancedLModule(**final_config)
    elif module_type == 'spectrum_aware':
        return SpectrumAwareLModule(**final_config)
    else:
        return LModule(**final_config)


if __name__ == "__main__":
    # Test L-Module implementation
    print("⚡ Testing L-Module Implementation")
    
    # Create test L-Module
    l_module = create_l_module({
        'input_dim': 512 * 3,
        'hidden_dim': 256,
        'h_context_dim': 128,
        'num_layers': 2
    })
    
    # Test data
    batch_size = 2
    histogram = torch.randn(batch_size, 512, 3)
    h_context = torch.randn(batch_size, 128)
    
    # Forward pass
    print(f"Input histogram shape: {histogram.shape}")
    print(f"Input context shape: {h_context.shape}")
    
    l_output, l_state = l_module(histogram, h_context)
    
    print(f"Output shape: {l_output.shape}")
    print(f"State info: {l_module.get_state_info(l_state)}")
    
    # Test reset
    fresh_state = l_module.reset_state(batch_size, histogram.device)
    print(f"Reset state info: {l_module.get_state_info(fresh_state)}")
    
    # Test enhanced version
    print("\n🔬 Testing Enhanced L-Module")
    enhanced_l_module = create_l_module({
        'module_type': 'enhanced',
        'hidden_dim': 256
    })
    
    enhanced_output, enhanced_state = enhanced_l_module(histogram, h_context)
    print(f"Enhanced output shape: {enhanced_output.shape}")
    
    print("✅ L-Module tests completed successfully")