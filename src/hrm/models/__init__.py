#!/usr/bin/env python3
"""
HRM Models Package - Hierarchical Reasoning Model components

This package contains the complete implementation of the Hierarchical
Reasoning Model architecture based on the scientific paper by Sapient Intelligence.

Components:
- H-Module: High-level reasoning with slow timescale
- L-Module: Low-level computation with fast timescale  
- Hierarchical Convergence: Core mechanism with O(1) memory
- Adaptive Computation Time: Q-learning based dynamic activation
"""

from .h_module import HModule, AdaptiveHModule, create_h_module
from .l_module import LModule, EnhancedLModule, SpectrumAwareLModule, create_l_module
from .hierarchical_convergence import (
    HierarchicalConvergence, 
    AdaptiveHierarchicalConvergence,
    ResidualHierarchicalConvergence,
    create_hierarchical_convergence
)
from .adaptive_computation_time import (
    AdaptiveComputationTime,
    EnhancedACT,
    create_act_module
)

__all__ = [
    # H-Module
    'HModule',
    'AdaptiveHModule',
    'create_h_module',
    
    # L-Module
    'LModule', 
    'EnhancedLModule',
    'SpectrumAwareLModule',
    'create_l_module',
    
    # Hierarchical Convergence
    'HierarchicalConvergence',
    'AdaptiveHierarchicalConvergence', 
    'ResidualHierarchicalConvergence',
    'create_hierarchical_convergence',
    
    # ACT
    'AdaptiveComputationTime',
    'EnhancedACT',
    'create_act_module'
]

__version__ = '1.0.0'
__author__ = 'Phideus HRM Implementation Team'