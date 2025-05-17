# warp_main.py

import argparse
import numpy as np
import torch
import os
from datetime import datetime
import time

from warp_training import WarpCurriculumTrainer

def parse_args():
    """Parse command line arguments for Warp-accelerated training."""
    parser = argparse.ArgumentParser(description='Train terminal agents with Warp acceleration')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=None, help='Total number of episodes')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Checkpoints directory')
    parser.add_argument('--results', type=str, default='results', help='Results directory')
    parser.add_argument('--logs', type=str, default='logs', help='Logs directory')
    parser.add_argument('--device', type=str, default='cuda', help='Computation device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    parser.add_argument('--render', action='store_true', help='Render during evaluation')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--disable-memory-tracking', action='store_true', help='Disable GPU memory tracking')
    
    # New optimization arguments
    parser.add_argument('--optimize', action='store_true', help='Enable simulation optimizations')
    parser.add_argument('--precompute-events', action='store_true', help='Precompute vehicle arrivals')
    parser.add_argument('--precompute-movements', action='store_true', help='Precompute movement times')
    parser.add_argument('--precompute-stacking', action='store_true', help='Precompute stacking compatibility')
    parser.add_argument('--save-tables', action='store_true', help='Save precomputed tables to disk')
    parser.add_argument('--load-tables', action='store_true', help='Load precomputed tables from disk')
    parser.add_argument('--tables-dir', type=str, default='precomputed', help='Directory for precomputed tables')
    parser.add_argument('--compare-performance', action='store_true', help='Compare optimized vs standard performance')
    
    return parser.parse_args()

def main():
    """Main function for Warp-accelerated training of terminal agents."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Generate experiment name if not provided
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"warp_terminal_training_{timestamp}"
    
    # Verify CUDA availability if requested
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    
    # Create curriculum trainer
    trainer = WarpCurriculumTrainer(
        config_path=args.config,
        checkpoints_dir=args.checkpoints,
        results_dir=args.results,
        log_dir=args.logs,
        experiment_name=args.name,
        device=args.device,
        memory_tracking=not args.disable_memory_tracking
    )
    
    # Prepare optimization options
    optimization_options = None
    if args.optimize or args.precompute_events or args.precompute_movements or args.precompute_stacking:
        optimization_options = {
            'optimize': True,
            'precompute_events': args.precompute_events or args.optimize,
            'precompute_movements': args.precompute_movements or args.optimize,
            'precompute_stacking': args.precompute_stacking or args.optimize,
            'save_tables': args.save_tables,
            'load_tables': args.load_tables,
            'tables_dir': args.tables_dir
        }
        
        # Create tables directory if it doesn't exist
        if args.save_tables:
            os.makedirs(args.tables_dir, exist_ok=True)
    
    # Compare performance if requested
    if args.compare_performance and optimization_options:
        # First run without optimizations for baseline
        print("\n=== Running baseline training without optimizations ===\n")
        baseline_start = time.time()
        original_agent, original_metrics = trainer.train(
            total_episodes=args.episodes,
            resume_checkpoint=args.resume
        )
        baseline_time = time.time() - baseline_start
        
        # Then run with optimizations
        print("\n=== Running optimized training with precomputation ===\n")
        optimized_start = time.time()
        optimized_agent, optimized_metrics = trainer.train(
            total_episodes=args.episodes,
            resume_checkpoint=args.resume,
            optimization_options=optimization_options
        )
        optimized_time = time.time() - optimized_start
        
        # Compare and report results
        print("\n=== Performance Comparison ===\n")
        print(f"Baseline training time: {baseline_time:.2f} seconds")
        print(f"Optimized training time: {optimized_time:.2f} seconds")
        if baseline_time > 0:
            speedup = baseline_time / optimized_time
            print(f"Speedup factor: {speedup:.2f}x faster with optimizations")
        
        # Use the optimized agent for evaluation
        agent = optimized_agent
    else:
        # Train agents with or without optimizations
        agent, _ = trainer.train(
            total_episodes=args.episodes,
            resume_checkpoint=args.resume,
            optimization_options=optimization_options
        )
    
    # Evaluate if requested
    if args.evaluate:
        trainer.evaluate(agent, num_episodes=5, render=args.render, 
                         optimization_options=optimization_options)

if __name__ == "__main__":
    main()