import numpy as np
import torch
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime, timedelta
import os
import json
import argparse
from tqdm import tqdm
import time
from typing import List, Dict, Tuple, Any
import random

# Import the EXISTING enhanced environment and agent
from simulation.terminal_components.ContainerTerminal import ContainerTerminal
from simulation.agents.ContinuousTerminalAgent import ContinuousTerminalAgent


class EnhancedTerminalTrainer:
    """Enhanced trainer that uses the temporal-aware ContainerTerminal."""
    
    def __init__(
        self,
        env_config: dict = None,
        agent_config: dict = None,
        save_dir: str = "checkpoints",
        log_interval: int = 100,
        save_interval: int = 10
    ):
        # Create environment using the EXISTING enhanced ContainerTerminal
        env_config = env_config or {
            'n_rows': 10,
            'n_bays': 20,
            'n_tiers': 5,
            'n_railtracks': 4,
            'split_factor': 4,
            'max_days': 365
        }
        self.env = ContainerTerminal(**env_config)
        
        # Create agent with updated config for temporal training
        state_dim = self.env.observation_space.shape[0]
        agent_config = agent_config or {
            'hidden_dims': [512, 512, 256],
            'learning_rate': 0.0001,
            'gamma': 0.99,
            'tau': 0.005
        }
        self.agent = ContinuousTerminalAgent(state_dim, **agent_config)
        
        # Override epsilon values if needed (these are hardcoded in the agent)
        # self.agent.epsilon = 1.0
        # self.agent.epsilon_decay = 0.997
        # self.agent.epsilon_min = 0.02
        
        # Training parameters
        self.save_dir = save_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Enhanced metrics tracking
        self.episode_rewards = []
        self.daily_metrics = {
            'rewards': [],
            'moves': [],
            'avg_distance': [],
            'late_trains': [],
            'queue_sizes': [],
            'urgent_moves_completed': [],  # NEW
            'average_urgency': [],         # NEW
            'wait_actions': []             # NEW - track if agent tried to wait
        }
        self.training_metrics = {
            'losses': deque(maxlen=1000),
            'epsilon': [],
            'learning_rate': [],
            'td_errors': deque(maxlen=1000),
            'reward_distribution': deque(maxlen=1000)  # NEW
        }
        
        # Initialize plot for live monitoring
        self.setup_plots()
        
        # Tracking for urgent moves
        self.urgent_moves_completed = 0
        
    def train_continuous(self, total_days: int = 365):
        """Train agent with temporal awareness using the enhanced environment."""
        print(f"Starting enhanced training for {total_days} days...")
        print("Using temporal-aware ContainerTerminal with:")
        print(f"  - {self.env.temporal_encoder.feature_dim} temporal state features")
        print(f"  - Progress-based reward system")
        print(f"  - Forced action selection (no waiting)")
        print("=" * 60)
        
        # Reset environment
        state, info = self.env.reset()
        episode_reward = 0
        step_count = 0
        
        # Daily tracking
        daily_reward = 0
        daily_moves = 0
        daily_distances = []
        daily_urgent_completed = 0
        daily_wait_attempts = 0
        
        # Track current day
        last_day = self.env.current_day
        
        # Time tracking
        start_time = time.time()
        day_start_time = time.time()
        
        # Progress bars
        day_pbar = tqdm(total=total_days, desc="Training Days", unit="day", position=0)
        step_pbar = tqdm(desc=f"Day {self.env.current_day} Steps", unit="step", position=1, leave=False)
        
        while self.env.current_day < total_days:
            # Get available actions and rankings from enhanced environment
            available_actions = list(range(len(info.get('move_list', []))))
            ranked_actions = info.get('ranked_move_list', [])  # This comes from _rank_moves_by_urgency
            
            # Select action with ranking support
            if available_actions:
                action = self.agent.select_action(
                    state, 
                    available_actions, 
                    training=True,
                    ranked_actions=ranked_actions
                )
            else:
                action = 0
                daily_wait_attempts += 1
                
            # Execute action (environment handles temporal reward calculation)
            next_state, reward, terminated, truncated, next_info = self.env.step(action)
            
            # Store experience with enhanced info
            info['available_actions'] = available_actions
            info['ranked_actions'] = ranked_actions
            self.agent.store_experience(state, action, reward, next_state, terminated, info)
            
            # Update agent
            loss = self.agent.update(current_day=self.env.current_day)
            if loss is not None:
                self.training_metrics['losses'].append(loss)
                
            # Track metrics
            episode_reward += reward
            daily_reward += reward
            if reward > 0:
                daily_moves += 1
                if 'distances' in self.env.daily_metrics:
                    daily_distances.extend(self.env.daily_metrics['distances'])
                    
            # Track reward distribution
            self.training_metrics['reward_distribution'].append(reward)
            
            # Track urgent moves (check if we did a high-urgency move)
            if reward > 15.0:  # High-value moves are typically urgent
                daily_urgent_completed += 1
                    
            # Update step progress
            step_pbar.update(1)
            step_pbar.set_postfix({
                'reward': f'{daily_reward:.1f}',
                'moves': daily_moves,
                'urgent': daily_urgent_completed,
                'eps': f'{self.agent.epsilon:.3f}',
                'loss': f'{loss:.4f}' if loss is not None else 'N/A'
            })
                    
            # Check for day change
            current_day = next_info['day']
            if current_day != last_day:
                # Day changed
                day_duration = time.time() - day_start_time
                steps_per_second = step_count / (time.time() - start_time) if step_count > 0 else 0
                
                # Update day progress
                day_pbar.update(1)
                day_pbar.set_postfix({
                    'total_reward': f'{episode_reward:.1f}',
                    'day_reward': f'{daily_reward:.1f}',
                    'urgent_done': daily_urgent_completed,
                    'steps/s': f'{steps_per_second:.1f}'
                })
                
                # Process daily metrics
                self.process_day_end(
                    last_day,
                    daily_reward,
                    daily_moves,
                    daily_distances,
                    daily_urgent_completed,
                    daily_wait_attempts,
                    info
                )
                
                # Reset daily tracking
                daily_reward = 0
                daily_moves = 0
                daily_distances = []
                daily_urgent_completed = 0
                daily_wait_attempts = 0
                day_start_time = time.time()
                
                # Agent day-end update
                self.agent.day_end_update(last_day, daily_reward)
                
                # Save checkpoint
                if last_day % self.save_interval == 0:
                    self.save_checkpoint(last_day)
                    
                # Update plots
                if last_day % 5 == 0:
                    self.update_plots()
                    
                # Reset progress bars
                step_pbar.close()
                step_pbar = tqdm(desc=f"Day {current_day} Steps", unit="step", position=1, leave=False)
                
                last_day = current_day
                
            # Update state and info
            state = next_state
            info = next_info
            step_count += 1
            
            # Log progress
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
        
        # Final statistics
        total_time = time.time() - start_time
        avg_time_per_day = total_time / max(1, self.env.current_day)
        
        # Final save
        self.save_checkpoint(self.env.current_day, final=True)
        self.save_metrics()
        
        print("\n" + "=" * 60)
        print("Enhanced training completed!")
        print(f"Total days: {self.env.current_day}")
        print(f"Total reward: {episode_reward:.2f}")
        print(f"Avg daily reward: {episode_reward / max(1, self.env.current_day):.2f}")
        print(f"Training time: {total_time:.1f}s ({total_time/60:.1f} min)")
        print(f"Avg time/day: {avg_time_per_day:.1f}s")
        
    def process_day_end(
        self, 
        day: int, 
        daily_reward: float,
        daily_moves: int,
        daily_distances: List[float],
        urgent_completed: int,
        wait_attempts: int,
        info: dict
    ):
        """Process enhanced metrics at end of day."""
        # Calculate statistics
        avg_distance = np.mean(daily_distances) if daily_distances else 0
        
        # Store metrics
        self.daily_metrics['rewards'].append(daily_reward)
        self.daily_metrics['moves'].append(daily_moves)
        self.daily_metrics['avg_distance'].append(avg_distance)
        self.daily_metrics['late_trains'].append(
            len(self.env.daily_metrics.get('late_trains', []))
        )
        self.daily_metrics['urgent_moves_completed'].append(urgent_completed)
        self.daily_metrics['wait_actions'].append(wait_attempts)
        
        # Calculate average urgency from reward distribution
        recent_rewards = list(self.training_metrics['reward_distribution'])[-daily_moves:]
        avg_reward_per_move = np.mean(recent_rewards) if recent_rewards else 0
        self.daily_metrics['average_urgency'].append(avg_reward_per_move)
        
        # Queue sizes
        queue_size = info.get('trains_in_terminal', 0) + info.get('trucks_in_terminal', 0)
        self.daily_metrics['queue_sizes'].append(queue_size)
        
        # Track exploration
        self.training_metrics['epsilon'].append(self.agent.epsilon)
        
        # Enhanced summary
        summary = f"Day {day}: R={daily_reward:.1f}, M={daily_moves}, "
        summary += f"Urg={urgent_completed}, D={avg_distance:.1f}m"
        
        if daily_reward > 100:
            summary += " ⭐"  # Excellent
        elif daily_reward > 50:
            summary += " ✓"   # Good
        elif daily_reward > 0:
            summary += " +"   # Positive
        else:
            summary += " ✗"   # Poor
            
        tqdm.write(summary)
        
    def setup_plots(self):
        """Setup enhanced visualization."""
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Enhanced Terminal Training Progress')
        
    def update_plots(self):
        """Update training plots with new metrics."""
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
            
        # Plot 2: Urgent moves completed
        if self.daily_metrics['urgent_moves_completed']:
            self.axes[0, 1].plot(self.daily_metrics['urgent_moves_completed'][-100:])
            self.axes[0, 1].set_title('Urgent Moves Completed')
            self.axes[0, 1].set_xlabel('Day')
            self.axes[0, 1].set_ylabel('Count')
            self.axes[0, 1].grid(True)
            
        # Plot 3: Average reward per move
        if self.daily_metrics['average_urgency']:
            self.axes[0, 2].plot(self.daily_metrics['average_urgency'][-100:])
            self.axes[0, 2].set_title('Average Reward per Move')
            self.axes[0, 2].set_xlabel('Day')
            self.axes[0, 2].set_ylabel('Avg Reward')
            self.axes[0, 2].grid(True)
            
        # Plot 4: Training loss
        if self.training_metrics['losses']:
            losses = list(self.training_metrics['losses'])
            self.axes[1, 0].plot(losses)
            self.axes[1, 0].set_title('Training Loss')
            self.axes[1, 0].set_xlabel('Update Step')
            self.axes[1, 0].set_ylabel('Loss')
            self.axes[1, 0].set_yscale('log')
            self.axes[1, 0].grid(True)
            
        # Plot 5: Distance efficiency
        if self.daily_metrics['avg_distance']:
            self.axes[1, 1].plot(self.daily_metrics['avg_distance'][-100:])
            self.axes[1, 1].set_title('Average Move Distance')
            self.axes[1, 1].set_xlabel('Day')
            self.axes[1, 1].set_ylabel('Distance (m)')
            self.axes[1, 1].grid(True)
            
        # Plot 6: Exploration rate vs performance
        if self.training_metrics['epsilon'] and self.daily_metrics['rewards']:
            ax = self.axes[1, 2]
            ax.plot(self.training_metrics['epsilon'][-100:], 'b-', label='Epsilon')
            ax.set_xlabel('Day')
            ax.set_ylabel('Epsilon', color='b')
            ax.tick_params(axis='y', labelcolor='b')
            ax.grid(True, alpha=0.3)
            
            # Overlay normalized rewards
            ax2 = ax.twinx()
            rewards = self.daily_metrics['rewards'][-100:]
            if rewards:
                normalized_rewards = (np.array(rewards) - np.min(rewards)) / (np.max(rewards) - np.min(rewards) + 1e-8)
                ax2.plot(normalized_rewards, 'r-', alpha=0.7, label='Norm. Rewards')
                ax2.set_ylabel('Normalized Rewards', color='r')
                ax2.tick_params(axis='y', labelcolor='r')
            
            ax.set_title('Exploration vs Performance')
            
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, "enhanced_training_progress.png")
        self.fig.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close(self.fig)
        
        # Recreate figure
        self.fig, self.axes = plt.subplots(2, 3, figsize=(18, 10))
        self.fig.suptitle('Enhanced Terminal Training Progress')
        
    def log_progress(self, step: int, total_reward: float):
        """Log enhanced training progress."""
        stats = self.agent.training_stats
        
        if stats['losses']:
            avg_loss = np.mean(list(stats['losses'])[-100:])
            avg_q = np.mean(list(stats['q_values'])[-100:]) if stats['q_values'] else 0
            avg_reward = np.mean(list(stats['rewards'])[-100:]) if stats['rewards'] else 0
            
            # Calculate reward distribution
            recent_rewards = list(self.training_metrics['reward_distribution'])[-100:]
            if recent_rewards:
                high_value_moves = sum(1 for r in recent_rewards if r > 10) / len(recent_rewards) * 100
            else:
                high_value_moves = 0
            
            tqdm.write(f"\n=== Step {step} | Day {self.env.current_day} ===")
            tqdm.write(f"  Avg Loss: {avg_loss:.4f}")
            tqdm.write(f"  Avg Q-value: {avg_q:.2f}")
            tqdm.write(f"  Avg Reward: {avg_reward:.2f}")
            tqdm.write(f"  High-value moves: {high_value_moves:.1f}%")
            tqdm.write(f"  Total Reward: {total_reward:.2f}")
            
    def save_checkpoint(self, day: int, final: bool = False):
        """Save enhanced checkpoint with proper JSON serialization."""
        checkpoint_name = f"enhanced_checkpoint_day_{day}.pt" if not final else "enhanced_final_checkpoint.pt"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        
        # Save agent
        self.agent.save(checkpoint_path)
        
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, deque):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        # Save enhanced metrics with conversion
        metrics_path = os.path.join(self.save_dir, f"enhanced_metrics_day_{day}.json")
        metrics_to_save = {
            'day': day,
            'daily_metrics': convert_to_serializable(self.daily_metrics),
            'training_metrics': {
                'epsilon': convert_to_serializable(self.training_metrics['epsilon']),
                'losses': convert_to_serializable(list(self.training_metrics['losses'])[-1000:]),
                'recent_rewards': convert_to_serializable(list(self.training_metrics['reward_distribution'])[-1000:])
            }
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
            
        tqdm.write(f"  ✓ Enhanced checkpoint saved: {checkpoint_name}")
        

    def save_metrics(self):
        """Save final enhanced metrics with proper JSON serialization."""
        # Convert numpy types to Python native types for JSON serialization
        def convert_to_serializable(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, np.float32) or isinstance(obj, np.float64):
                return float(obj)
            elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
                return int(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, deque):
                return [convert_to_serializable(item) for item in obj]
            else:
                return obj
        
        metrics_path = os.path.join(self.save_dir, "enhanced_training_metrics.json")
        metrics_to_save = {
            'daily_metrics': convert_to_serializable(self.daily_metrics),
            'episode_metrics': convert_to_serializable(dict(self.env.episode_metrics))
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
            
        # Generate final analysis
        self._generate_final_analysis()
            
    def _generate_final_analysis(self):
        """Generate final training analysis."""
        analysis_path = os.path.join(self.save_dir, "training_analysis.txt")
        
        with open(analysis_path, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("ENHANCED TERMINAL TRAINING ANALYSIS\n")
            f.write("=" * 60 + "\n\n")
            
            # Overall performance
            total_days = len(self.daily_metrics['rewards'])
            if total_days > 0:
                avg_daily_reward = np.mean(self.daily_metrics['rewards'])
                avg_moves_per_day = np.mean(self.daily_metrics['moves'])
                avg_urgent_completed = np.mean(self.daily_metrics['urgent_moves_completed'])
                
                f.write("Overall Performance:\n")
                f.write(f"  Total days trained: {total_days}\n")
                f.write(f"  Average daily reward: {avg_daily_reward:.2f}\n")
                f.write(f"  Average moves per day: {avg_moves_per_day:.1f}\n")
                f.write(f"  Average urgent moves completed: {avg_urgent_completed:.1f}\n\n")
                
                # Trend analysis
                if total_days > 20:
                    early_rewards = np.mean(self.daily_metrics['rewards'][:20])
                    late_rewards = np.mean(self.daily_metrics['rewards'][-20:])
                    improvement = ((late_rewards - early_rewards) / early_rewards) * 100
                    
                    f.write("Learning Progress:\n")
                    f.write(f"  Early performance (first 20 days): {early_rewards:.2f}\n")
                    f.write(f"  Late performance (last 20 days): {late_rewards:.2f}\n")
                    f.write(f"  Improvement: {improvement:.1f}%\n\n")
                
                # Efficiency metrics
                if self.daily_metrics['avg_distance']:
                    avg_distance = np.mean(self.daily_metrics['avg_distance'])
                    f.write("Efficiency Metrics:\n")
                    f.write(f"  Average move distance: {avg_distance:.1f}m\n")
                    f.write(f"  Late trains per day: {np.mean(self.daily_metrics['late_trains']):.2f}\n\n")
        
        print(f"Training analysis saved: {analysis_path}")
        
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
        
    def load_checkpoint(self, checkpoint_path: str):
        """Load training from checkpoint."""
        self.agent.load(checkpoint_path)
        print(f"Loaded checkpoint from: {checkpoint_path}")


def main():
    """Main training script with enhanced features."""
    parser = argparse.ArgumentParser(description='Train Enhanced Container Terminal Agent')
    parser.add_argument('--days', type=int, default=365, help='Number of days to train')
    parser.add_argument('--rows', type=int, default=15, help='Number of yard rows')
    parser.add_argument('--bays', type=int, default=20, help='Number of yard bays')
    parser.add_argument('--tiers', type=int, default=5, help='Number of yard tiers')
    parser.add_argument('--save-dir', type=str, default='enhanced_checkpoints', help='Save directory')
    parser.add_argument('--load', type=str, default=None, help='Load checkpoint path')
    parser.add_argument('--no-benchmark', action='store_true', help='Skip speed benchmark')
    args = parser.parse_args()
    
    # Print configuration
    print("=" * 60)
    print("ENHANCED Container Terminal DRL Training")
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
    
    # Agent configuration (enhanced for temporal awareness)
    agent_config = {
        'hidden_dims': [512, 512, 256],
        'learning_rate': 0.0001,
        'gamma': 0.99,
        'tau': 0.005
    }
    
    # Create trainer
    trainer = EnhancedTerminalTrainer(
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
        
    # Start enhanced training
    print("\nStarting enhanced training with temporal awareness...")
    print("Key improvements:")
    print("  ✓ 20 temporal state features")
    print("  ✓ Progress-based rewards (no distance penalty)")
    print("  ✓ Forced action selection (no waiting)")
    print("  ✓ Move ranking by urgency")
    print("  ✓ Smart exploration bias")
    print("")
    
    trainer.train_continuous(total_days=args.days)
    

if __name__ == "__main__":
    main()