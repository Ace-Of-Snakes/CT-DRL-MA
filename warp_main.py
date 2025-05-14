# warp_main.py

import argparse
import numpy as np
import torch
import os
from datetime import datetime

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
    
    # Train agents
    agent, _ = trainer.train(
        total_episodes=args.episodes,
        resume_checkpoint=args.resume
    )
    
    # Evaluate if requested
    if args.evaluate:
        trainer.evaluate(agent, num_episodes=5, render=args.render)

if __name__ == "__main__":
    main()