import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime
import os
import json
import argparse
from tqdm import tqdm
import time
from typing import List
from simulation.terminal_components.ContainerTerminal import ContainerTerminal
from simulation.agents.ContinuousTerminalAgent import ContinuousTerminalAgent
import random

class TerminalTrainer:
    """Trainer for continuous terminal operations."""
    
    def __init__(
        self,
        env_config: dict = None,
        agent_config: dict = None,
        save_dir: str = "checkpoints",
        log_interval: int = 100,
        save_interval: int = 10  # Save every N days
    ):
        # Create environment
        env_config = env_config or {
            'n_rows': 10,
            'n_bays': 20,
            'n_tiers': 5,
            'n_railtracks': 4,
            'split_factor': 4,
            'max_days': 365
        }
        self.env = ContainerTerminal(**env_config)
        
        # Create agent
        state_dim = self.env.observation_space.shape[0]
        agent_config = agent_config or {
            'hidden_dims': [512, 512, 256],
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'tau': 0.005
        }
        self.agent = ContinuousTerminalAgent(state_dim, **agent_config)
        
        # Training parameters
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Metrics tracking
        self.episode_rewards = []
        self.daily_metrics = {
            'rewards': [],
            'moves': [],
            'avg_distance': [],
            'late_trains': [],
            'queue_sizes': []
        }
        self.training_metrics = {
            'losses': deque(maxlen=1000),
            'epsilon': [],
            'learning_rate': []
        }
        
        # Initialize plot for live monitoring
        self.setup_plots()
        
    def benchmark_speed(self, n_steps: int = 100):
        """Run a quick benchmark to estimate training speed."""
        print("Running speed benchmark...")
        
        # Reset environment
        state, info = self.env.reset()
        
        # Time n_steps
        start_time = time.time()
        for _ in tqdm(range(n_steps), desc="Benchmarking", leave=False):
            # Get available actions
            available_actions = list(range(len(info.get('move_list', []))))
            
            # Random action for benchmark
            if available_actions:
                action = random.choice(available_actions)
            else:
                action = 0
                
            # Execute action
            next_state, reward, terminated, truncated, next_info = self.env.step(action)
            
            if terminated or truncated:
                state, info = self.env.reset()
            else:
                state = next_state
                info = next_info
        
        # Calculate statistics
        duration = time.time() - start_time
        steps_per_second = n_steps / duration
        estimated_time_per_day = 1000 / steps_per_second  # Assuming ~1000 steps per day
        
        print(f"\nBenchmark Results:")
        print(f"  Steps per second: {steps_per_second:.1f}")
        print(f"  Estimated time per simulated day: {estimated_time_per_day:.1f}s")
        print(f"  Estimated time for 365 days: {estimated_time_per_day * 365 / 60:.1f} minutes")
        print("-" * 60)
        
        return steps_per_second
        
    def setup_plots(self):
        """Setup matplotlib for live training visualization."""
        # Set up matplotlib
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Terminal Training Progress')
        
    def update_plots(self):
        """Update and save training plots."""
        # Clear axes
        for ax in self.axes.flat:
            ax.clear()
            
        # Plot 1: Daily rewards
        if self.daily_metrics['rewards']:
            self.axes[0, 0].plot(self.daily_metrics['rewards'][-100:])
            self.axes[0, 0].set_title('Daily Rewards (Last 100 days)')
            self.axes[0, 0].set_xlabel('Day')
            self.axes[0, 0].set_ylabel('Total Reward')
            self.axes[0, 0].grid(True)
            
        # Plot 2: Average distance per move
        if self.daily_metrics['avg_distance']:
            self.axes[0, 1].plot(self.daily_metrics['avg_distance'][-100:])
            self.axes[0, 1].set_title('Average Move Distance')
            self.axes[0, 1].set_xlabel('Day')
            self.axes[0, 1].set_ylabel('Distance (m)')
            self.axes[0, 1].grid(True)
            
        # Plot 3: Training loss
        if self.training_metrics['losses']:
            losses = list(self.training_metrics['losses'])
            self.axes[1, 0].plot(losses)
            self.axes[1, 0].set_title('Training Loss')
            self.axes[1, 0].set_xlabel('Update Step')
            self.axes[1, 0].set_ylabel('Loss')
            self.axes[1, 0].set_yscale('log')
            self.axes[1, 0].grid(True)
            
        # Plot 4: Exploration rate
        if self.training_metrics['epsilon']:
            self.axes[1, 1].plot(self.training_metrics['epsilon'])
            self.axes[1, 1].set_title('Exploration Rate')
            self.axes[1, 1].set_xlabel('Day')
            self.axes[1, 1].set_ylabel('Epsilon')
            self.axes[1, 1].grid(True)
            
        plt.tight_layout()
        
        # Save to file instead of showing (to avoid conflicts with tqdm)
        plot_path = os.path.join(self.save_dir, "training_progress_current.png")
        self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(self.fig)
        
        # Recreate figure for next update
        self.fig, self.axes = plt.subplots(2, 2, figsize=(12, 8))
        self.fig.suptitle('Terminal Training Progress')
            
        # Plot 1: Daily rewards
        if self.daily_metrics['rewards']:
            self.axes[0, 0].plot(self.daily_metrics['rewards'][-100:])
            self.axes[0, 0].set_title('Daily Rewards (Last 100 days)')
            self.axes[0, 0].set_xlabel('Day')
            self.axes[0, 0].set_ylabel('Total Reward')
            self.axes[0, 0].grid(True)
            
        # Plot 2: Average distance per move
        if self.daily_metrics['avg_distance']:
            self.axes[0, 1].plot(self.daily_metrics['avg_distance'][-100:])
            self.axes[0, 1].set_title('Average Move Distance')
            self.axes[0, 1].set_xlabel('Day')
            self.axes[0, 1].set_ylabel('Distance (m)')
            self.axes[0, 1].grid(True)
            
        # Plot 3: Training loss
        if self.training_metrics['losses']:
            losses = list(self.training_metrics['losses'])
            self.axes[1, 0].plot(losses)
            self.axes[1, 0].set_title('Training Loss')
            self.axes[1, 0].set_xlabel('Update Step')
            self.axes[1, 0].set_ylabel('Loss')
            self.axes[1, 0].set_yscale('log')
            self.axes[1, 0].grid(True)
            
        # Plot 4: Exploration rate
        if self.training_metrics['epsilon']:
            self.axes[1, 1].plot(self.training_metrics['epsilon'])
            self.axes[1, 1].set_title('Exploration Rate')
            self.axes[1, 1].set_xlabel('Day')
            self.axes[1, 1].set_ylabel('Epsilon')
            self.axes[1, 1].grid(True)
            
        plt.tight_layout()
        plt.pause(0.01)
        
    def train_continuous(self, total_days: int = 365):
        """Train agent continuously for specified number of days."""
        print(f"Starting continuous training for {total_days} days...")
        print("=" * 60)
        
        # Reset environment
        state, info = self.env.reset()
        episode_reward = 0
        step_count = 0
        
        # Daily tracking
        daily_reward = 0
        daily_moves = 0
        daily_distances = []
        
        # Time tracking
        start_time = time.time()
        day_start_time = time.time()
        
        # Create progress bar for days
        day_pbar = tqdm(total=total_days, desc="Training Days", unit="day", position=0)
        
        # Create progress bar for steps (will be updated each day)
        step_pbar = tqdm(desc=f"Day {self.env.current_day} Steps", unit="step", position=1, leave=False)
        
        while self.env.current_day < total_days:
            # Get available actions
            available_actions = list(range(len(info.get('move_list', []))))
            
            # Select action
            if available_actions:
                action = self.agent.select_action(state, available_actions, training=True)
            else:
                action = 0  # Wait action
                
            # Execute action
            next_state, reward, terminated, truncated, next_info = self.env.step(action)
            
            # Store experience
            self.agent.store_experience(state, action, reward, next_state, terminated, info)
            
            # Update agent
            loss = self.agent.update()
            if loss is not None:
                self.training_metrics['losses'].append(loss)
                
            # Track metrics
            episode_reward += reward
            daily_reward += reward
            if reward > 0:  # Actual move executed
                daily_moves += 1
                if 'distances' in self.env.daily_metrics:
                    daily_distances.extend(self.env.daily_metrics['distances'])
                    
            # Update step progress bar
            step_pbar.update(1)
            step_pbar.set_postfix({
                'reward': f'{daily_reward:.1f}',
                'moves': daily_moves,
                'eps': f'{self.agent.epsilon:.3f}',
                'loss': f'{loss:.4f}' if loss is not None else 'N/A'
            })
                    
            # Check for day end
            current_day = next_info['day']
            if current_day > self.env.current_day or (current_day == 0 and self.env.current_day > 0):
                # Calculate day statistics
                day_duration = time.time() - day_start_time
                steps_per_second = step_count / (time.time() - start_time) if step_count > 0 else 0
                
                # Update day progress bar
                day_pbar.update(1)
                day_pbar.set_postfix({
                    'total_reward': f'{episode_reward:.1f}',
                    'day_reward': f'{daily_reward:.1f}',
                    'steps/s': f'{steps_per_second:.1f}',
                    'day_time': f'{day_duration:.1f}s'
                })
                
                # Day ended - process daily metrics
                self.process_day_end(
                    self.env.current_day,
                    daily_reward,
                    daily_moves,
                    daily_distances,
                    info
                )
                
                # Reset daily tracking
                daily_reward = 0
                daily_moves = 0
                daily_distances = []
                day_start_time = time.time()
                
                # Agent day-end update with soft reset
                self.agent.day_end_update(self.env.current_day, daily_reward)
                
                # Save checkpoint
                if self.env.current_day % self.save_interval == 0:
                    self.save_checkpoint(self.env.current_day)
                    
                # Update plots
                if self.env.current_day % 5 == 0:  # Update every 5 days
                    self.update_plots()
                    
                # Reset step progress bar for new day
                step_pbar.close()
                step_pbar = tqdm(desc=f"Day {current_day} Steps", unit="step", position=1, leave=False)
                    
            # Update state and info
            state = next_state
            info = next_info
            step_count += 1
            
            # Log progress (less frequently with tqdm)
            if step_count % (self.log_interval * 10) == 0:
                self.log_progress(step_count, episode_reward)
                
            # Handle termination
            if terminated or truncated:
                print(f"\nEpisode ended after {self.env.current_day} days")
                print(f"Total episode reward: {episode_reward:.2f}")
                break
        
        # Close progress bars
        step_pbar.close()
        day_pbar.close()
        
        # Calculate final statistics
        total_time = time.time() - start_time
        avg_time_per_day = total_time / max(1, self.env.current_day)
        total_steps_per_second = step_count / total_time if total_time > 0 else 0
                
        # Final save
        self.save_checkpoint(self.env.current_day, final=True)
        self.save_metrics()
        
        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Total days simulated: {self.env.current_day}")
        print(f"Total episode reward: {episode_reward:.2f}")
        print(f"Average daily reward: {episode_reward / max(1, self.env.current_day):.2f}")
        print(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        print(f"Average time per day: {avg_time_per_day:.1f}s")
        print(f"Average steps per second: {total_steps_per_second:.1f}")
        print(f"Total steps: {step_count}")
        
    def process_day_end(
        self, 
        day: int, 
        daily_reward: float,
        daily_moves: int,
        daily_distances: List[float],
        info: dict
    ):
        """Process metrics at end of day."""
        # Calculate daily statistics
        avg_distance = np.mean(daily_distances) if daily_distances else 0
        
        # Store metrics
        self.daily_metrics['rewards'].append(daily_reward)
        self.daily_metrics['moves'].append(daily_moves)
        self.daily_metrics['avg_distance'].append(avg_distance)
        self.daily_metrics['late_trains'].append(
            len(self.env.daily_metrics.get('late_trains', []))
        )
        
        # Queue sizes
        queue_size = info.get('trains_in_terminal', 0) + info.get('trucks_in_terminal', 0)
        self.daily_metrics['queue_sizes'].append(queue_size)
        
        # Track exploration
        self.training_metrics['epsilon'].append(self.agent.epsilon)
        
        # Print daily summary (shorter version with tqdm)
        summary = f"Day {day}: R={daily_reward:.1f}, M={daily_moves}, D={avg_distance:.1f}m"
        if daily_reward > 0:
            summary += " ✓"
        else:
            summary += " ✗"
        tqdm.write(summary)
            
    def log_progress(self, step: int, total_reward: float):
        """Log training progress."""
        stats = self.agent.training_stats
        
        if stats['losses']:
            avg_loss = np.mean(list(stats['losses'])[-100:])
            avg_q = np.mean(list(stats['q_values'])[-100:]) if stats['q_values'] else 0
            avg_reward = np.mean(list(stats['rewards'])[-100:]) if stats['rewards'] else 0
            
            tqdm.write(f"\n=== Step {step} | Day {self.env.current_day} ===")
            tqdm.write(f"  Avg Loss: {avg_loss:.4f}")
            tqdm.write(f"  Avg Q-value: {avg_q:.2f}")
            tqdm.write(f"  Avg Reward: {avg_reward:.2f}")
            tqdm.write(f"  Total Reward: {total_reward:.2f}")
            
    def save_checkpoint(self, day: int, final: bool = False):
        """Save training checkpoint."""
        checkpoint_name = f"checkpoint_day_{day}.pt" if not final else "final_checkpoint.pt"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        
        # Save agent
        self.agent.save(checkpoint_path)
        
        # Save metrics
        metrics_path = os.path.join(self.save_dir, f"metrics_day_{day}.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'day': day,
                'daily_metrics': self.daily_metrics,
                'training_metrics': {
                    'epsilon': self.training_metrics['epsilon'],
                    'losses': list(self.training_metrics['losses'])[-1000:]
                }
            }, f, indent=2)
            
        tqdm.write(f"  ✓ Checkpoint saved: {checkpoint_name}")
        
    def save_metrics(self):
        """Save final training metrics and plots."""
        # Save detailed metrics
        metrics_path = os.path.join(self.save_dir, "training_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({
                'daily_metrics': self.daily_metrics,
                'episode_metrics': self.env.episode_metrics
            }, f, indent=2)
            
        # Save final plots
        self.update_plots()
        plot_path = os.path.join(self.save_dir, "training_progress.png")
        self.fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close(self.fig)
        tqdm.write(f"Training plots saved: {plot_path}")
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load training from checkpoint."""
        self.agent.load(checkpoint_path)
        print(f"Loaded checkpoint from: {checkpoint_path}")


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train Container Terminal Agent')
    parser.add_argument('--days', type=int, default=365, help='Number of days to train')
    parser.add_argument('--rows', type=int, default=10, help='Number of yard rows')
    parser.add_argument('--bays', type=int, default=20, help='Number of yard bays')
    parser.add_argument('--tiers', type=int, default=5, help='Number of yard tiers')
    parser.add_argument('--save-dir', type=str, default='checkpoints', help='Save directory')
    parser.add_argument('--load', type=str, default=None, help='Load checkpoint path')
    parser.add_argument('--no-benchmark', action='store_true', help='Skip speed benchmark')
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("Container Terminal DRL Training")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Yard: {args.rows} rows × {args.bays} bays × {args.tiers} tiers")
    print(f"  Training days: {args.days}")
    print(f"  Save directory: {args.save_dir}")
    print(f"  Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("=" * 60)
    
    # Environment configuration
    env_config = {
        'n_rows': args.rows,
        'n_bays': args.bays,
        'n_tiers': args.tiers,
        'n_railtracks': 4,
        'split_factor': 4,
        'max_days': args.days
    }
    
    # Agent configuration
    agent_config = {
        'hidden_dims': [512, 512, 256],
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'tau': 0.005
    }
    
    # Create trainer
    trainer = TerminalTrainer(
        env_config=env_config,
        agent_config=agent_config,
        save_dir=args.save_dir,
        log_interval=100,
        save_interval=10
    )
    
    # Load checkpoint if specified
    if args.load:
        trainer.load_checkpoint(args.load)
    
    # Run speed benchmark unless disabled
    if not args.no_benchmark:
        trainer.benchmark_speed(n_steps=100)
        
    # Start training
    print("\nStarting training...")
    trainer.train_continuous(total_days=args.days)
    

if __name__ == "__main__":
    main()