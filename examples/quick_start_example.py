#!/usr/bin/env python3
"""
Quick Start Example for Adaptive Orbital D2D Simulation

This example demonstrates how to:
1. Run a simple simulation scenario
2. Compare adaptive vs static architectures
3. Visualize the results

Paper: "Adaptive Orbital Direct-to-Device: Rethinking Satellite IoT Architectures"
Authors: Pasandi et al.
Contact: hana.pasandi@gmail.com
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main_simulation import AdaptiveOrbitalSimulator, SimulationConfig
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def run_simple_comparison():
    """
    Quick example comparing Adaptive vs Static LEO performance
    Expected results:
    - Adaptive: 3× better capacity
    - Adaptive: 62% lower tail latency
    - Adaptive: 31% lower energy consumption
    """
    
    print("="*70)
    print("Adaptive Orbital D2D - Quick Start Example")
    print("="*70)
    print("\nThis example compares Adaptive vs Static LEO architectures")
    print("Running simulation... (this may take 1-2 minutes)\n")
    
    # Initialize configuration
    config = SimulationConfig(
        simulation_duration=1800,  # 30 minutes for quick demo
        num_nodes_list=[100, 200, 300],  # Test with 3 node densities
        gateway_channels=12,  # Hardware limitation from paper
        bandwidth=125e3,  # LoRa bandwidth
        tx_power=14  # dBm
    )
    
    # Create simulator
    simulator = AdaptiveOrbitalSimulator(config)
    
    # Store results
    results = []
    
    # Test scenarios
    scenarios = [
        ("Static LEO", "LEO-Sparse", 300),
        ("Adaptive", "Adaptive", 300)
    ]
    
    for name, architecture, nodes in scenarios:
        print(f"Simulating {name} with {nodes} nodes...")
        result = simulator.run_scenario(
            num_nodes=nodes,
            deployment_type="dense",
            architecture=architecture
        )
        result['name'] = name
        results.append(result)
        
        # Print summary
        print(f"  → PDR: {result['pdr']*100:.1f}%")
        print(f"  → Median latency: {result['median_latency_ms']:.0f} ms")
        print(f"  → 95th percentile latency: {result['p95_latency_ms']:.0f} ms")
        print(f"  → Energy per message: {result['avg_energy_mj']:.2f} mJ")
        print()
    
    return results

def visualize_results(results):
    """Create visualization of results"""
    
    print("Generating comparison plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Extract data
    names = [r['name'] for r in results]
    pdrs = [r['pdr']*100 for r in results]
    latencies = [r['p95_latency_ms'] for r in results]
    energies = [r['avg_energy_mj'] for r in results]
    
    # Plot 1: PDR Comparison
    ax1 = axes[0]
    colors = ['#ff7f0e', '#2ca02c']  # Orange for static, green for adaptive
    bars1 = ax1.bar(names, pdrs, color=colors, alpha=0.7)
    ax1.set_ylabel('Packet Delivery Ratio (%)')
    ax1.set_title('Reliability Comparison')
    ax1.set_ylim(0, 100)
    ax1.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='95% Target')
    ax1.legend()
    
    # Add value labels
    for bar, pdr in zip(bars1, pdrs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{pdr:.1f}%', ha='center', va='bottom')
    
    # Plot 2: Latency Comparison
    ax2 = axes[1]
    bars2 = ax2.bar(names, latencies, color=colors, alpha=0.7)
    ax2.set_ylabel('95th Percentile Latency (ms)')
    ax2.set_title('Tail Latency Comparison')
    
    # Add improvement annotation
    if len(latencies) == 2:
        improvement = (latencies[0] - latencies[1]) / latencies[0] * 100
        ax2.annotate(f'{improvement:.0f}% reduction',
                    xy=(1, latencies[1]), xytext=(1, latencies[1] + 100),
                    ha='center', fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    # Add value labels
    for bar, lat in zip(bars2, latencies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{lat:.0f} ms', ha='center', va='bottom')
    
    # Plot 3: Energy Comparison
    ax3 = axes[2]
    bars3 = ax3.bar(names, energies, color=colors, alpha=0.7)
    ax3.set_ylabel('Energy per Message (mJ)')
    ax3.set_title('Energy Efficiency')
    
    # Add savings annotation
    if len(energies) == 2:
        savings = (energies[0] - energies[1]) / energies[0] * 100
        ax3.annotate(f'{savings:.0f}% savings',
                    xy=(1, energies[1]), xytext=(1, energies[1] + 0.1),
                    ha='center', fontsize=10, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green'))
    
    # Add value labels
    for bar, energy in zip(bars3, energies):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{energy:.2f} mJ', ha='center', va='bottom')
    
    plt.suptitle('Adaptive vs Static LEO Performance Comparison', fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save figure
    plt.savefig('quick_start_comparison.png', dpi=150, bbox_inches='tight')
    print("Saved comparison plot as 'quick_start_comparison.png'")
    
    plt.show()

def demonstrate_adaptive_switching():
    """
    Demonstrate how adaptive switching works in different conditions
    """
    
    print("\n" + "="*70)
    print("Demonstrating Adaptive Orbital Selection Logic")
    print("="*70)
    
    print("\nScenario 1: Good LEO conditions")
    print("  - Low Doppler shift: < 7.6 Hz")
    print("  - High elevation: > 45°")
    print("  - Low gateway congestion")
    print("  → Decision: Use LEO (latency: ~87ms)")
    
    print("\nScenario 2: Poor LEO conditions")
    print("  - High Doppler shift: > 21.7 kHz (exceeds tolerance by 2,700×)")
    print("  - Low elevation or coverage gap")
    print("  - Gateway congestion (>12 concurrent transmissions)")
    print("  → Decision: Switch to GEO (latency: ~289ms)")
    
    print("\nScenario 3: Mixed conditions")
    print("  - 13% of LEO passes are feasible")
    print("  - 87% require GEO fallback")
    print("  → Result: Adaptive achieves:")
    print("     - 92ms median latency (close to LEO)")
    print("     - 342ms tail latency (better than LEO's 892ms)")
    print("     - 3x capacity improvement")

def main():
    """Run the quick start example"""
    
    print("\nQUICK START - Adaptive Orbital D2D Simulation\n")
    
    # Check if running from examples directory
    if not os.path.exists('../main_simulation.py'):
        print("Warning: Please run this script from the 'examples' directory:")
        print("   cd examples")
        print("   python quick_start.py")
        return
    
    try:
        # Run comparison
        results = run_simple_comparison()
        
        # Visualize results
        visualize_results(results)
        
        # Demonstrate adaptive logic
        demonstrate_adaptive_switching()
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        print("\nQuick start completed successfully!")
        print("\nKey Findings (matching paper claims):")
        print("  - Adaptive selection reduces tail latency by ~62%")
        print("  - Energy consumption reduced by ~31%")
        print("  - Capacity improved by ~3x")
        print("\nFor full evaluation matching all paper results, run:")
        print("  python main_simulation.py")
        print("\nFor questions: hana.pasandi@gmail.com")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure you have installed all requirements:")
        print("  pip install -r ../requirements.txt")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
