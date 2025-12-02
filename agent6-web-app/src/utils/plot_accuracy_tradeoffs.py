"""
Generate accuracy vs execution time tradeoff plots from profiler JSON.

Usage:
    python plot_accuracy_tradeoffs.py <path_to_profile.json> [output_dir]

This script generates publication-quality plots for research papers showing:
1. Accuracy vs Execution Time scatter plots (per test suite)
2. Speedup vs Accuracy Loss curves
3. Efficiency frontier visualization
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Set publication-quality defaults
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 10
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 11
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9


def load_profile(profile_path: str):
    """Load profiler JSON"""
    with open(profile_path, 'r') as f:
        return json.load(f)


def plot_accuracy_vs_time_per_suite(profile_data, output_dir):
    """
    Generate separate accuracy vs time plot for each test suite.
    Shows the tradeoff curve with quality levels annotated.
    """
    tradeoff_analysis = profile_data.get('benchmark_results', {}).get('accuracy_tradeoff_analysis', {})
    
    if not tradeoff_analysis:
        print("No tradeoff analysis found in profile")
        return
    
    for suite_name, suite_data in tradeoff_analysis.items():
        tradeoffs = suite_data.get('tradeoffs', [])
        
        if not tradeoffs:
            continue
        
        # Extract data
        times = [t['execution_time_seconds'] for t in tradeoffs]
        accuracies = [t['accuracy_retained_percent'] for t in tradeoffs]
        qualities = [t['quality_level'] for t in tradeoffs]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot tradeoff curve
        ax.plot(times, accuracies, 'o-', linewidth=2, markersize=8, 
                color='#2E86AB', label='Quality Levels')
        
        # Annotate quality levels
        for time, acc, q in zip(times, accuracies, qualities):
            ax.annotate(f'q={q}', (time, acc), 
                       textcoords="offset points", xytext=(5, 5),
                       fontsize=8, color='#333')
        
        # Styling
        ax.set_xlabel('Execution Time (seconds)', fontweight='bold')
        ax.set_ylabel('Accuracy Retained (%)', fontweight='bold')
        ax.set_title(f'Accuracy vs Execution Time Trade-off\n{suite_name.replace("_", " ").title()}',
                    fontweight='bold', pad=15)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim([0, 105])
        
        # Add ideal region annotation
        ax.axhspan(90, 100, alpha=0.1, color='green', label='High Accuracy Region (>90%)')
        ax.legend(loc='lower right')
        
        # Save
        output_path = Path(output_dir) / f'tradeoff_{suite_name}.png'
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved: {output_path}")


def plot_speedup_vs_accuracy_loss(profile_data, output_dir):
    """
    Plot speedup factor vs accuracy loss for all test suites.
    Shows efficiency of different quality levels.
    """
    tradeoff_analysis = profile_data.get('benchmark_results', {}).get('accuracy_tradeoff_analysis', {})
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(tradeoff_analysis)))
    
    for idx, (suite_name, suite_data) in enumerate(tradeoff_analysis.items()):
        tradeoffs = suite_data.get('tradeoffs', [])
        
        if not tradeoffs:
            continue
        
        speedups = [t['speedup_vs_baseline'] for t in tradeoffs]
        accuracy_losses = [t['accuracy_loss_percent'] for t in tradeoffs]
        qualities = [t['quality_level'] for t in tradeoffs]
        
        # Plot
        ax.plot(accuracy_losses, speedups, 'o-', linewidth=2, markersize=7,
               color=colors[idx], label=suite_name.replace('_', ' ')[:30])
        
        # Annotate first and last points
        if len(qualities) > 1:
            ax.annotate(f'q={qualities[0]}', (accuracy_losses[0], speedups[0]),
                       textcoords="offset points", xytext=(-10, 5), fontsize=7)
            ax.annotate(f'q={qualities[-1]}', (accuracy_losses[-1], speedups[-1]),
                       textcoords="offset points", xytext=(5, -5), fontsize=7)
    
    ax.set_xlabel('Accuracy Loss (%)', fontweight='bold')
    ax.set_ylabel('Speedup Factor (vs Baseline)', fontweight='bold')
    ax.set_title('Speedup vs Accuracy Loss Trade-off\nAcross Different Test Suites',
                fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper left', fontsize=8, ncol=2)
    
    # Add "acceptable loss" region
    ax.axvspan(0, 10, alpha=0.1, color='green', label='<10% Loss Region')
    
    output_path = Path(output_dir) / 'speedup_vs_accuracy_loss.png'
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def plot_efficiency_frontier(profile_data, output_dir):
    """
    Plot efficiency frontier: best accuracy for each time budget.
    Useful for finding Pareto-optimal quality settings.
    """
    viz_data = profile_data.get('benchmark_results', {}).get('visualization_data', [])
    
    if not viz_data:
        print("No visualization data found")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Group by test suite
    suites = {}
    for entry in viz_data:
        suite = entry['test_suite']
        if suite not in suites:
            suites[suite] = []
        suites[suite].append(entry)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(suites)))
    
    for idx, (suite_name, data) in enumerate(suites.items()):
        # Sort by execution time
        data = sorted(data, key=lambda x: x['execution_time'])
        
        times = [d['execution_time'] for d in data]
        accuracies = [d['accuracy_retained'] for d in data]
        
        ax.scatter(times, accuracies, s=80, alpha=0.7, color=colors[idx],
                  label=suite_name.replace('_', ' ')[:30], edgecolors='black', linewidth=0.5)
    
    ax.set_xlabel('Execution Time (seconds)', fontweight='bold')
    ax.set_ylabel('Accuracy Retained (%)', fontweight='bold')
    ax.set_title('Efficiency Frontier: All Test Suites\nHigher-Left is Better',
                fontweight='bold', pad=15)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=8, ncol=2)
    ax.set_xscale('log')  # Log scale for time to see patterns better
    
    output_path = Path(output_dir) / 'efficiency_frontier.png'
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_path}")


def generate_summary_table(profile_data, output_dir):
    """Generate LaTeX table of tradeoff analysis for paper"""
    tradeoff_analysis = profile_data.get('benchmark_results', {}).get('accuracy_tradeoff_analysis', {})
    
    output_path = Path(output_dir) / 'tradeoff_table.tex'
    
    with open(output_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Accuracy vs Execution Time Trade-offs}\n")
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\hline\n")
        f.write("Quality Level & Time (s) & Accuracy (\\%) & Speedup & Loss (\\%) \\\\\n")
        f.write("\\hline\n")
        
        # Take first suite as example
        first_suite = list(tradeoff_analysis.values())[0] if tradeoff_analysis else None
        if first_suite:
            for t in first_suite.get('tradeoffs', [])[:5]:  # Top 5
                f.write(f"{t['quality_level']} & "
                       f"{t['execution_time_seconds']:.2f} & "
                       f"{t['accuracy_retained_percent']:.1f} & "
                       f"{t['speedup_vs_baseline']:.2f}x & "
                       f"{t['accuracy_loss_percent']:.1f} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"✓ Saved LaTeX table: {output_path}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_accuracy_tradeoffs.py <profile.json> [output_dir]")
        sys.exit(1)
    
    profile_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './tradeoff_plots'
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"Loading profile from: {profile_path}")
    profile_data = load_profile(profile_path)
    
    print(f"\nGenerating plots in: {output_dir}")
    print("=" * 60)
    
    # Generate all plots
    plot_accuracy_vs_time_per_suite(profile_data, output_dir)
    plot_speedup_vs_accuracy_loss(profile_data, output_dir)
    plot_efficiency_frontier(profile_data, output_dir)
    generate_summary_table(profile_data, output_dir)
    
    print("=" * 60)
    print(f"✅ All plots generated successfully in: {output_dir}")
    print("\nGenerated files:")
    print("  - tradeoff_<suite_name>.png (per test suite)")
    print("  - speedup_vs_accuracy_loss.png (combined)")
    print("  - efficiency_frontier.png (Pareto frontier)")
    print("  - tradeoff_table.tex (LaTeX table)")


if __name__ == '__main__':
    main()
