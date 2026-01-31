# simulation/training/visualize_training.py
"""
Visualization tools for training metrics.

Generates plots for:
- Reward over time
- Imports/exports per day
- Loss curves
- Stage progression
"""
import os
import argparse
from pathlib import Path
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_daily_metrics(log_dir: str) -> pd.DataFrame:
    """Load daily metrics CSV."""
    csv_path = Path(log_dir) / "daily_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Daily metrics not found: {csv_path}")
    return pd.read_csv(csv_path)


def load_stage_metrics(log_dir: str) -> pd.DataFrame:
    """Load stage metrics CSV."""
    csv_path = Path(log_dir) / "stage_metrics.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Stage metrics not found: {csv_path}")
    return pd.read_csv(csv_path)


def plot_reward_over_time(df: pd.DataFrame, output_path: Optional[str] = None):
    """Plot total reward over training days."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # Get unique stages
    stages = df['stage'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(stages)))
    
    for i, stage in enumerate(stages):
        stage_data = df[df['stage'] == stage]
        ax.plot(
            stage_data['day_index'],
            stage_data['total_reward'],
            color=colors[i],
            alpha=0.7,
            linewidth=0.5
        )
    
    # Rolling average
    window = 30
    rolling_reward = df['total_reward'].rolling(window=window).mean()
    ax.plot(
        df['day_index'],
        rolling_reward,
        color='red',
        linewidth=2,
        label=f'{window}-day rolling avg'
    )
    
    # Stage boundaries
    for stage in stages[:-1]:
        last_day = df[df['stage'] == stage]['day_index'].max()
        ax.axvline(x=last_day, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Training Day')
    ax.set_ylabel('Daily Reward')
    ax.set_title('Training Reward Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_moves_over_time(df: pd.DataFrame, output_path: Optional[str] = None):
    """Plot moves executed over training days."""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    stages = df['stage'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(stages)))
    
    for i, stage in enumerate(stages):
        stage_data = df[df['stage'] == stage]
        ax.scatter(
            stage_data['day_index'],
            stage_data['moves_executed'],
            color=colors[i],
            alpha=0.3,
            s=5,
            label=f'Stage {stage}'
        )
    
    # Rolling average
    window = 30
    rolling_moves = df['moves_executed'].rolling(window=window).mean()
    ax.plot(
        df['day_index'],
        rolling_moves,
        color='red',
        linewidth=2,
        label=f'{window}-day rolling avg'
    )
    
    ax.set_xlabel('Training Day')
    ax.set_ylabel('Moves Executed')
    ax.set_title('Container Moves Over Time')
    ax.legend(loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_epsilon_decay(df: pd.DataFrame, output_path: Optional[str] = None):
    """Plot epsilon decay over training."""
    fig, ax = plt.subplots(figsize=(14, 4))
    
    ax.plot(df['day_index'], df['epsilon'], color='blue', linewidth=1)
    
    # Stage boundaries
    stages = df['stage'].unique()
    for stage in stages[:-1]:
        last_day = df[df['stage'] == stage]['day_index'].max()
        ax.axvline(x=last_day, color='gray', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Training Day')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate (ε) Over Training')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(df['epsilon']) * 1.1)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_stage_summary(stage_df: pd.DataFrame, output_path: Optional[str] = None):
    """Plot stage-level summary metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Average reward per day by stage
    ax1 = axes[0, 0]
    ax1.bar(stage_df['stage'], stage_df['avg_reward_per_day'], color='steelblue')
    ax1.set_xlabel('Stage')
    ax1.set_ylabel('Avg Reward/Day')
    ax1.set_title('Average Daily Reward by Stage')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Import cap vs avg moves
    ax2 = axes[0, 1]
    ax2.plot(stage_df['import_cap'], stage_df['avg_moves_per_day'], 
             'o-', color='green', markersize=8)
    ax2.set_xlabel('Import Cap')
    ax2.set_ylabel('Avg Moves/Day')
    ax2.set_title('Throughput vs Import Capacity')
    ax2.grid(True, alpha=0.3)
    
    # Total reward by stage
    ax3 = axes[1, 0]
    ax3.bar(stage_df['stage'], stage_df['total_reward'], color='coral')
    ax3.set_xlabel('Stage')
    ax3.set_ylabel('Total Reward')
    ax3.set_title('Total Reward by Stage')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Import cap progression
    ax4 = axes[1, 1]
    ax4.plot(stage_df['stage'], stage_df['import_cap'], 
             's-', color='purple', markersize=8, linewidth=2)
    ax4.set_xlabel('Stage')
    ax4.set_ylabel('Import Cap')
    ax4.set_title('Curriculum Progression')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_loss_curve(df: pd.DataFrame, output_path: Optional[str] = None):
    """Plot training loss over time."""
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Filter out zero losses
    loss_data = df[df['avg_loss'] > 0]
    
    if len(loss_data) == 0:
        print("No loss data to plot")
        return
    
    ax.scatter(
        loss_data['day_index'],
        loss_data['avg_loss'],
        alpha=0.3,
        s=3,
        color='blue'
    )
    
    # Rolling average
    window = 30
    rolling_loss = loss_data['avg_loss'].rolling(window=window).mean()
    ax.plot(
        loss_data['day_index'],
        rolling_loss,
        color='red',
        linewidth=2,
        label=f'{window}-day rolling avg'
    )
    
    ax.set_xlabel('Training Day')
    ax.set_ylabel('Avg Loss')
    ax.set_title('Training Loss Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def generate_all_plots(log_dir: str, output_dir: Optional[str] = None):
    """Generate all visualization plots."""
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    try:
        daily_df = load_daily_metrics(log_dir)
        print(f"Loaded {len(daily_df)} daily records")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    try:
        stage_df = load_stage_metrics(log_dir)
        print(f"Loaded {len(stage_df)} stage records")
    except FileNotFoundError:
        stage_df = None
        print("No stage metrics found (training may still be running)")
    
    # Generate plots
    print("\nGenerating plots...")
    
    output_base = output_dir if output_dir else None
    
    plot_reward_over_time(
        daily_df,
        str(output_base / "reward_over_time.png") if output_base else None
    )
    
    plot_moves_over_time(
        daily_df,
        str(output_base / "moves_over_time.png") if output_base else None
    )
    
    plot_epsilon_decay(
        daily_df,
        str(output_base / "epsilon_decay.png") if output_base else None
    )
    
    plot_loss_curve(
        daily_df,
        str(output_base / "loss_curve.png") if output_base else None
    )
    
    if stage_df is not None and len(stage_df) > 0:
        plot_stage_summary(
            stage_df,
            str(output_base / "stage_summary.png") if output_base else None
        )
    
    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Visualize training metrics")
    parser.add_argument("log_dir", type=str, help="Directory containing log files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save plots (shows interactively if not set)")
    
    args = parser.parse_args()
    generate_all_plots(args.log_dir, args.output_dir)


if __name__ == "__main__":
    main()
