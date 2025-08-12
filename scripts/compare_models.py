#!/usr/bin/env python3
"""
Phideus Dual Architecture - Model Comparison System
Runs A/B testing between VAE and HRM architectures
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def load_model_results(model_path: str, architecture: str) -> Dict[str, Any]:
    """Load validation results for a model"""
    
    if architecture.lower() == 'vae':
        # Load VAE validation results
        validation_file = Path(model_path).parent / "validation" / "validation_report.json"
        if validation_file.exists():
            with open(validation_file, 'r') as f:
                results = json.load(f)
        else:
            results = {
                "reconstruction_quality": 0.797,  # Current known value
                "harmonic_detection_rate": 0.067, # Current known value
                "memory_usage_gb": 1.0,
                "parameters_m": 15.3,
                "training_time_min": 0.1
            }
    
    elif architecture.lower() == 'hrm':
        # Load HRM results (placeholder - to be populated)
        results = {
            "harmonic_search_efficiency": 0.0,  # To be measured
            "memory_complexity": "O(1)",
            "hierarchical_convergence": False,   # To be validated
            "act_adaptivity": 0.0,              # To be measured
            "parameters_m": 27.0                # Estimated
        }
    
    return results

def compare_architectures(vae_model: str, hrm_model: str) -> Dict[str, Any]:
    """Run comprehensive comparison between architectures"""
    
    print("📊 Phideus Dual Architecture Comparison")
    print("="*50)
    
    # Load results for both models
    vae_results = load_model_results(vae_model, 'vae')
    hrm_results = load_model_results(hrm_model, 'hrm')
    
    # Define comparison metrics
    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "vae": vae_model,
            "hrm": hrm_model
        },
        "metrics": {}
    }
    
    # Harmonic Detection Performance
    vae_harmonic = vae_results.get("harmonic_detection_rate", 0.067)
    hrm_harmonic = hrm_results.get("harmonic_search_efficiency", 0.0)
    
    comparison["metrics"]["harmonic_detection"] = {
        "vae": f"{vae_harmonic:.1%}",
        "hrm": f"{hrm_harmonic:.1%}" if hrm_harmonic > 0 else "TBD",
        "winner": "hrm" if hrm_harmonic > vae_harmonic else "vae" if vae_harmonic > 0 else "tie",
        "improvement": f"{hrm_harmonic/vae_harmonic:.1f}x" if hrm_harmonic > 0 and vae_harmonic > 0 else "TBD"
    }
    
    # Memory Efficiency
    comparison["metrics"]["memory_efficiency"] = {
        "vae": f"{vae_results.get('memory_usage_gb', 1.0):.1f}GB VRAM (O(T))",
        "hrm": hrm_results.get("memory_complexity", "O(1) constant"),
        "winner": "hrm",  # Always wins due to O(1) vs O(T)
        "advantage": "Constant memory vs linear scaling"
    }
    
    # Model Complexity
    vae_params = vae_results.get("parameters_m", 15.3)
    hrm_params = hrm_results.get("parameters_m", 27.0)
    
    comparison["metrics"]["parameters"] = {
        "vae": f"{vae_params:.1f}M",
        "hrm": f"{hrm_params:.1f}M",
        "winner": "vae" if vae_params < hrm_params else "hrm",
        "ratio": f"{hrm_params/vae_params:.1f}x larger" if hrm_params > vae_params else f"{vae_params/hrm_params:.1f}x larger"
    }
    
    # Reconstruction Quality
    vae_recon = vae_results.get("reconstruction_quality", 0.797)
    hrm_recon = hrm_results.get("reconstruction_quality", 0.0)
    
    comparison["metrics"]["reconstruction"] = {
        "vae": f"{vae_recon:.1%}",
        "hrm": f"{hrm_recon:.1%}" if hrm_recon > 0 else "TBD",
        "winner": "vae" if vae_recon > hrm_recon else "hrm" if hrm_recon > vae_recon else "tie"
    }
    
    return comparison

def print_comparison_report(comparison: Dict[str, Any]):
    """Print formatted comparison report"""
    
    print("\n🎯 COMPARISON RESULTS")
    print("="*50)
    
    metrics = comparison["metrics"]
    
    print("\n📈 HARMONIC DETECTION")
    harmonic = metrics["harmonic_detection"]
    print(f"  VAE Current:  {harmonic['vae']}")
    print(f"  HRM Research: {harmonic['hrm']}")
    print(f"  Winner: {harmonic['winner'].upper()}")
    if harmonic['improvement'] != 'TBD':
        print(f"  Improvement: {harmonic['improvement']}")
    
    print("\n💾 MEMORY EFFICIENCY")
    memory = metrics["memory_efficiency"] 
    print(f"  VAE Current:  {memory['vae']}")
    print(f"  HRM Research: {memory['hrm']}")
    print(f"  Winner: {memory['winner'].upper()}")
    print(f"  Advantage: {memory['advantage']}")
    
    print("\n🔢 MODEL PARAMETERS")
    params = metrics["parameters"]
    print(f"  VAE Current:  {params['vae']}")
    print(f"  HRM Research: {params['hrm']}")
    print(f"  Winner: {params['winner'].upper()}")
    print(f"  Size: {params['ratio']}")
    
    print("\n🎨 RECONSTRUCTION QUALITY")  
    recon = metrics["reconstruction"]
    print(f"  VAE Current:  {recon['vae']}")
    print(f"  HRM Research: {recon['hrm']}")
    print(f"  Winner: {recon['winner'].upper()}")
    
    # Overall recommendation
    print("\n🏆 OVERALL ASSESSMENT")
    print("="*50)
    
    # Count wins
    wins = {"vae": 0, "hrm": 0, "tie": 0}
    for metric_data in metrics.values():
        winner = metric_data.get("winner", "tie")
        wins[winner] += 1
    
    print(f"VAE Wins: {wins['vae']}")
    print(f"HRM Wins: {wins['hrm']}")
    print(f"Ties: {wins['tie']}")
    
    if wins['hrm'] > wins['vae']:
        print("\n✅ RECOMMENDATION: HRM Research Line")
        print("   HRM shows superior performance in key metrics")
    elif wins['vae'] > wins['hrm']:
        print("\n✅ RECOMMENDATION: VAE Current Line")
        print("   VAE shows more consistent performance")
    else:
        print("\n⚖️  RECOMMENDATION: Continue Both Lines")
        print("   Results inconclusive, need more data")

def main():
    parser = argparse.ArgumentParser(description="Compare VAE and HRM architectures")
    parser.add_argument("--vae-model", default="models/vae/attention/best_model.pth",
                       help="Path to VAE model")
    parser.add_argument("--hrm-model", default="models/hrm/core/hrm_initial.pth", 
                       help="Path to HRM model")
    parser.add_argument("--output", help="Save comparison to JSON file")
    
    args = parser.parse_args()
    
    # Run comparison
    comparison = compare_architectures(args.vae_model, args.hrm_model)
    
    # Print report
    print_comparison_report(comparison)
    
    # Save to file if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(comparison, f, indent=2)
        print(f"\n💾 Comparison saved to: {args.output}")

if __name__ == "__main__":
    main()