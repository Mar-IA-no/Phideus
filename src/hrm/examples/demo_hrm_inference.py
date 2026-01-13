#!/usr/bin/env python3
"""
HRM Inference Demo
Simple demonstration of HRM inference on sample data
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.h_module import create_h_module
from models.l_module import create_l_module  
from models.hierarchical_convergence import create_hierarchical_convergence
from models.adaptive_computation_time import create_act_module
from training.train_hrm_hierarchical import PhideusHRM


def generate_sample_data(batch_size: int = 4, harmonic_ratios: list = None):
    """Generate sample harmonic data for demonstration"""
    if harmonic_ratios is None:
        harmonic_ratios = [1.0, 5/4, 3/2, 2.0, 5/2, 3.0]  # Common harmonics
    
    # Create histograms with peaks at harmonic ratios
    histograms = []
    
    for _ in range(batch_size):
        # Initialize empty histogram (512 bins, ratio range 1.0 to 6.0)
        hist_linear = np.zeros(512)
        hist_log = np.zeros(512)
        hist_entropy = np.zeros(512)
        
        # Add harmonic peaks
        for ratio in harmonic_ratios:
            if 1.0 <= ratio <= 6.0:
                # Linear histogram
                bin_idx = int((ratio - 1.0) / 5.0 * 511)
                hist_linear[bin_idx] = np.random.uniform(0.7, 1.0)
                
                # Log histogram (perceptual)
                log_ratio = np.log2(ratio)
                log_bin_idx = int(log_ratio / np.log2(6.0) * 511)
                hist_log[min(log_bin_idx, 511)] = np.random.uniform(0.6, 0.9)
                
                # Entropy histogram
                hist_entropy[bin_idx] = np.random.uniform(0.5, 0.8)
        
        # Add some noise
        hist_linear += np.random.normal(0, 0.05, 512)
        hist_log += np.random.normal(0, 0.05, 512)
        hist_entropy += np.random.normal(0, 0.03, 512)
        
        # Normalize
        hist_linear = np.maximum(0, hist_linear)
        hist_log = np.maximum(0, hist_log)
        hist_entropy = np.maximum(0, hist_entropy)
        
        # Stack into (512, 3) format
        combined_hist = np.stack([hist_linear, hist_log, hist_entropy], axis=1)
        histograms.append(combined_hist)
    
    return torch.FloatTensor(histograms)


def demo_individual_components():
    """Demonstrate individual HRM components"""
    print("🧪 Testing Individual HRM Components\n")
    
    # Sample data
    batch_size = 2
    sample_histograms = generate_sample_data(batch_size)
    print(f"📊 Generated sample data: {sample_histograms.shape}")
    
    # Test L-Module
    print("\n⚡ Testing L-Module:")
    l_module = create_l_module({
        'input_dim': 512 * 3,
        'hidden_dim': 256,
        'h_context_dim': 128
    })
    
    h_context = torch.randn(batch_size, 128)
    l_output, l_state = l_module(sample_histograms, h_context)
    
    print(f"  Input: {sample_histograms.shape}")
    print(f"  L-Output: {l_output.shape}")
    print(f"  L-State layers: {len(l_state)}")
    
    # Test H-Module
    print("\n🧠 Testing H-Module:")
    h_module = create_h_module({
        'l_output_dim': 256,
        'h_hidden_dim': 128,
        'memory_depth': 8
    })
    
    # Simulate sequence of L-Module outputs
    l_sequence = [torch.randn(batch_size, 256) for _ in range(4)]
    h_context, h_state, h_cell = h_module(l_sequence)
    
    print(f"  L-Sequence length: {len(l_sequence)}")
    print(f"  H-Context: {h_context.shape}")
    print(f"  H-State: {h_state.shape}")
    
    # Test ACT
    print("\n⏱️ Testing Adaptive Computation Time:")
    act_module = create_act_module({
        'l_output_dim': 256,
        'max_steps': 10
    })
    
    should_halt, halt_probs, q_values = act_module(l_output, step_count=5)
    
    print(f"  Should halt: {should_halt}")
    print(f"  Halt probabilities: {halt_probs}")
    print(f"  Q-values shape: {q_values.shape}")
    
    # Test Hierarchical Convergence
    print("\n🔄 Testing Hierarchical Convergence:")
    convergence_system = create_hierarchical_convergence({
        'input_dim': 512 * 3,
        'N': 2,  # Reduced for demo
        'T': 4   # Reduced for demo
    })
    
    final_output, debug_info = convergence_system(sample_histograms, debug_mode=True)
    
    print(f"  Input: {sample_histograms.shape}")
    print(f"  Final output: {final_output.shape}")
    print(f"  Convergence measures: {debug_info['convergence_measures']}")
    print(f"  Reset points: {debug_info['reset_points']}")


def demo_full_hrm_model():
    """Demonstrate full HRM model inference"""
    print("\n\n🏗️ Testing Complete HRM Model\n")
    
    # Create full HRM model
    model = PhideusHRM(
        input_dim=(512, 3),
        l_hidden_dim=256,
        h_hidden_dim=128,
        N=3,  # Reduced for demo
        T=6   # Reduced for demo
    )
    
    # Sample data with known harmonic content
    harmonic_ratios = [1.0, 5/4, 3/2, 2.0, 8/3, 3.0]  # Perfect fifth, octave, etc.
    sample_data = generate_sample_data(batch_size=4, harmonic_ratios=harmonic_ratios)
    
    print(f"📊 Input shape: {sample_data.shape}")
    print(f"🎵 Harmonic ratios: {harmonic_ratios}")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        output, debug_info = model(sample_data, debug_mode=True)
    
    print(f"\n🔬 Results:")
    print(f"  Output shape: {output.shape}")
    print(f"  Output statistics:")
    print(f"    Mean: {output.mean().item():.6f}")
    print(f"    Std: {output.std().item():.6f}")
    print(f"    Min: {output.min().item():.6f}")
    print(f"    Max: {output.max().item():.6f}")
    
    if debug_info:
        print(f"\n🐛 Debug Information:")
        if 'convergence_measures' in debug_info:
            print(f"  Convergence measures: {debug_info['convergence_measures']}")
        if 'reset_points' in debug_info:
            print(f"  Reset points: {debug_info['reset_points']}")
        if 'act_statistics' in debug_info:
            print(f"  ACT statistics: {debug_info['act_statistics']}")
    
    return output, sample_data


def visualize_results(output, input_data):
    """Create visualization of HRM inference results"""
    print("\n📈 Creating visualization...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('HRM Inference Demo Results', fontsize=16)
        
        # Plot input histograms
        sample_idx = 0
        
        # Linear histogram
        axes[0, 0].plot(input_data[sample_idx, :, 0].numpy(), 'b-', linewidth=2)
        axes[0, 0].set_title('Input: Linear Histogram')
        axes[0, 0].set_xlabel('Ratio Bin')
        axes[0, 0].set_ylabel('Magnitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Log histogram
        axes[0, 1].plot(input_data[sample_idx, :, 1].numpy(), 'g-', linewidth=2)
        axes[0, 1].set_title('Input: Log Histogram')
        axes[0, 1].set_xlabel('Ratio Bin')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Entropy histogram
        axes[1, 0].plot(input_data[sample_idx, :, 2].numpy(), 'r-', linewidth=2)
        axes[1, 0].set_title('Input: Entropy Histogram')
        axes[1, 0].set_xlabel('Ratio Bin')
        axes[1, 0].set_ylabel('Magnitude')
        axes[1, 0].grid(True, alpha=0.3)
        
        # HRM output visualization
        if len(output.shape) > 1 and output.shape[1] > 1:
            axes[1, 1].plot(output[sample_idx].numpy(), 'purple', linewidth=2)
            axes[1, 1].set_title('HRM Output (Latent Representation)')
            axes[1, 1].set_xlabel('Dimension')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # If output is 1D, show as bar chart
            axes[1, 1].bar(range(len(output[sample_idx])), output[sample_idx].numpy(), 
                          color='purple', alpha=0.7)
            axes[1, 1].set_title('HRM Output (Scalar Values)')
            axes[1, 1].set_xlabel('Sample Index')
            axes[1, 1].set_ylabel('Value')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path(__file__).parent / 'hrm_demo_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"💾 Visualization saved: {output_path}")
        
        # Also show if in interactive environment
        try:
            plt.show()
        except:
            pass
        
        plt.close()
        
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")


def save_demo_results(output, input_data):
    """Save demo results to JSON for analysis"""
    results = {
        'demo_metadata': {
            'model_type': 'PhideusHRM',
            'input_shape': list(input_data.shape),
            'output_shape': list(output.shape),
            'harmonic_ratios_used': [1.0, 5/4, 3/2, 2.0, 8/3, 3.0]
        },
        'input_samples': input_data[:2].tolist(),  # Save first 2 samples
        'output_samples': output[:2].tolist(),
        'statistics': {
            'input_mean': input_data.mean().item(),
            'input_std': input_data.std().item(),
            'output_mean': output.mean().item(),
            'output_std': output.std().item()
        }
    }
    
    output_path = Path(__file__).parent / 'hrm_demo_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved: {output_path}")


def main():
    """Main demo function"""
    print("🚀 HRM Inference Demo")
    print("=" * 50)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # Test individual components
        demo_individual_components()
        
        # Test full model
        output, input_data = demo_full_hrm_model()
        
        # Visualize results
        visualize_results(output, input_data)
        
        # Save results
        save_demo_results(output, input_data)
        
        print(f"\n✅ HRM demo completed successfully!")
        print(f"🔍 Check the generated files in: {Path(__file__).parent}")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()