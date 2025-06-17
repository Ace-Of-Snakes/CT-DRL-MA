# train_continuous_adaptive.py
import os
import json
import time
import numpy as np
import random
import torch
from datetime import datetime, timedelta
from collections import defaultdict, deque
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Any
from simulation.terminal_components.ContainerTerminal import ContainerTerminal

class ContinuousAdaptiveTrainer:
    """Trainer for continuous, lifelong learning in terminal operations."""
    
    def __init__(self,
                 agent_type: str = 'hierarchical',
                 terminal_config: Dict[str, int] = None,
                 save_dir: str = 'checkpoints_continuous',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        self.agent_type = agent_type
        self.device = device
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Terminal configuration
        if terminal_config is None:
            terminal_config = {
                'n_rows': 5,
                'n_bays': 50,
                'n_tiers': 5,
                'n_railtracks': 6,
                'split_factor': 4,
                'max_days': 365 * 5  # 5 years of operation
            }
        self.terminal_config = terminal_config
        
        # Create environment (no reset during operation!)
        self.env = ContainerTerminal(**terminal_config)
        
        # Create agent
        state_dim = self.env.observation_space.shape[0]
        self.create_agent(agent_type, state_dim)
        
        # Continuous learning parameters
        self.online_learning_rate = 1e-4  # Lower LR for online adaptation
        self.replay_ratio = 0.1  # Ratio of replay vs new experiences
        self.adaptation_window = 1000  # Steps for computing running statistics
        
        # Performance tracking (rolling windows)
        self.hourly_rewards = deque(maxlen=24)  # Last 24 hours
        self.daily_rewards = deque(maxlen=30)   # Last 30 days
        self.weekly_rewards = deque(maxlen=52)  # Last 52 weeks
        
        # Adaptive metrics
        self.performance_baseline = None
        self.drift_detector = DriftDetector(window_size=1000)
        
        # Continuous statistics
        self.total_steps = 0
        self.total_reward = 0
        self.current_hour_reward = 0
        self.current_day_reward = 0
        self.current_week_reward = 0
        
        # Track last move for display
        self.last_move_type = "none"
        self.last_move_time = 0
        self.last_reward = 0

        # Add move type abbreviations
        self.move_type_abbrev = {
            'wait': 'WAIT',
            'train_to_truck': 'T→TRK',
            'truck_to_train': 'TRK→T',
            'yard_to_train': 'Y→TRN',
            'train_to_yard': 'TRN→Y',
            'yard_to_truck': 'Y→TRK',
            'truck_to_yard': 'TRK→Y',
            'yard_to_yard': 'Y→Y',
            'yard_to_stack': 'Y→STK',
            'from_yard': 'FROM_Y',
            'to_yard': 'TO_Y'
        }

        self.move_type_counts = defaultdict(int)
        self.successful_moves = 0
        self.failed_moves = 0

    def train_continuous(self, target_days: int = 365):
        """
        Train continuously without episodes - just like real operations.
        """
        print(f"\n{'='*60}")
        print(f"CONTINUOUS ADAPTIVE TRAINING")
        print(f"Terminal: {self.terminal_config['n_rows']}×{self.terminal_config['n_bays']}×{self.terminal_config['n_tiers']}")
        print(f"Target: {target_days} days of continuous operation")
        print(f"{'='*60}\n")
        
        # Initialize environment (only once!)
        state, info = self.env.reset()
        
        # Progress tracking
        day_pbar = tqdm(total=target_days, desc="Operational Days", position=0)
        current_day = 0
        last_hour = 0
        
        # Initialize performance baseline with first reward
        self.performance_baseline = 100.0  # Starting baseline

        # Continuous operation loop
        while self.env.current_day < target_days:
            # Hour tracking
            current_hour = int(self.env.current_time / 3600)
            
            # CRITICAL: Always sync and get fresh moves
            self.env.logistics.sync_pickup_mappings()
            available_moves = self.env.logistics.find_moves_optimized()
            
            # Select and execute action
            action, move_info = self.select_action_with_exploration(
                state, available_moves, info
            )
            
            # Execute action
            next_state, reward, terminated, truncated, next_info = self.env.step(action)
            
            # Track move info with abbreviation
            move_type = move_info['move_data'].get('move_type', 'wait')
            self.last_move_type = self.move_type_abbrev.get(move_type, move_type[:6])
            self.last_move_time = next_info.get('last_move_time', 0)
            self.last_reward = reward

            # Additional stats
            if reward > 0:
                self.successful_moves += 1
                self.move_type_counts[move_type] += 1
            else:
                self.failed_moves += 1
            
            # Store experience
            self.agent.store_transition(
                state, action, reward, next_state,
                False,  # Never "done" in continuous operation
                {'available_moves': available_moves, 'move_info': move_info}
            )
            
            # Online learning - update every step!
            if self.total_steps > 0 and self.total_steps % 4 == 0:
                metrics = self.online_update()
                
                # Detect performance drift
                if self.drift_detector.detect_drift(reward):
                    self.handle_drift_detection()
            
            # Update tracking
            self.total_reward += reward
            self.current_hour_reward += reward
            self.current_day_reward += reward
            self.total_steps += 1
            
            # Hour change
            if current_hour != last_hour:
                self.hourly_rewards.append(self.current_hour_reward)
                
                # Update 6h average
                six_hour_avg = np.mean(list(self.hourly_rewards)[-6:]) if len(self.hourly_rewards) >= 6 else np.mean(self.hourly_rewards)
                
                # Update progress bar with hourly info
                day_pbar.set_postfix({
                    'hour': current_hour,
                    'h_reward': f'{self.current_hour_reward:.1f}',
                    '6h_avg': f'{six_hour_avg:.1f}',
                    'total': f'{self.total_reward:.0f}',
                    'move': self.last_move_type,  # Now shows full abbreviation
                    't': f'{self.last_move_time:.0f}s',  # Shortened label
                    'r': f'{self.last_reward:.1f}'  # Add last reward
                })
                
                self.current_hour_reward = 0
                last_hour = current_hour
            
            # Day change
            if self.env.current_day > current_day:
                self.daily_rewards.append(self.current_day_reward)
                self.log_daily_performance(self.env.current_day)
                
                # Update progress
                day_pbar.update(1)
                
                # Save checkpoint daily
                self.save_continuous_checkpoint(self.env.current_day)
                
                # Reset daily tracking
                self.current_day_reward = 0
                current_day = self.env.current_day
            
            # Update state
            state = next_state
            info = next_info
            
            # No episode resets! Continue with current state
        
        print(f"\n{'='*60}")
        print("CONTINUOUS OPERATION COMPLETE")
        print(f"Total days: {self.env.current_day}")
        print(f"Total reward: {self.total_reward:.2f}")
        print(f"Average daily reward: {self.total_reward/max(1, self.env.current_day):.2f}")
        print(f"{'='*60}")

    def online_update(self) -> Dict[str, float]:
        """
        Perform online learning update with experience replay.
        Balances new experiences with replay for stability.
        """
        batch_size = 32
        
        # Check if we have enough experiences
        if len(self.agent.memory) < batch_size:
            return {}
        
        # Sample recent experiences (online learning)
        recent_size = int(batch_size * (1 - self.replay_ratio))
        recent_size = min(recent_size, len(self.agent.memory))  # Don't exceed available
        recent_experiences = list(self.agent.memory)[-recent_size:]
        
        # Sample older experiences (replay)
        replay_size = batch_size - recent_size
        older_experiences = list(self.agent.memory)[:-recent_size] if recent_size > 0 else list(self.agent.memory)
        
        if len(older_experiences) > 0 and replay_size > 0:
            # Only sample what we actually have
            replay_size = min(replay_size, len(older_experiences))
            replay_experiences = random.sample(older_experiences, replay_size)
        else:
            replay_experiences = []
        
        # Combine batch
        batch = recent_experiences + replay_experiences
        
        if len(batch) < 16:  # Minimum batch size
            return {}
        
        # Update with lower learning rate for stability
        original_lr = self.get_current_learning_rate()
        self.set_learning_rate(self.online_learning_rate)
        
        # Perform update
        metrics = self.agent.update(batch_size=len(batch))
        
        # Restore original learning rate
        self.set_learning_rate(original_lr)
        
        return metrics
    
    def select_action_with_exploration(self, state, available_moves, info):
        """
        Select action with adaptive exploration based on performance.
        """
        # Adaptive exploration: reduce when performing well
        performance_ratio = self.get_performance_ratio()
        base_epsilon = 0.1  # Base exploration rate
        
        # Increase exploration if performance drops
        if performance_ratio < 0.8:
            epsilon = base_epsilon * 2
        elif performance_ratio < 0.9:
            epsilon = base_epsilon * 1.5
        else:
            epsilon = base_epsilon * 0.5
        
        # Time-based exploration boost (explore more during quiet hours)
        hour_of_day = (self.env.current_time % 86400) / 3600
        if 2 <= hour_of_day <= 6:  # Night hours
            epsilon *= 1.5
        
        # Create embeddings and select action
        if not available_moves:
            return 0, {'move_id': 'wait', 'move_data': {'move_type': 'wait'}}
        
        move_embeddings = self.create_move_embeddings(available_moves, info)
        
        # Select action
        action, move_info = self.agent.select_action(
            state, available_moves, move_embeddings, epsilon
        )
        
        return action, move_info
    
    def handle_drift_detection(self):
        """
        Handle detected concept drift by adjusting learning parameters.
        """
        print(f"\n⚠️  Performance drift detected at step {self.total_steps}")
        
        # Increase learning rate temporarily
        self.online_learning_rate *= 2
        
        # Increase exploration
        # This is handled in select_action_with_exploration
        
        # Clear old experiences to focus on recent patterns
        if len(self.agent.memory) > 10000:
            # Keep only recent 5000 experiences
            self.agent.memory = deque(list(self.agent.memory)[-5000:], maxlen=100000)
        
        # Schedule learning rate decay
        self.lr_decay_steps = 1000
    
    def get_performance_ratio(self) -> float:
        """Get current performance relative to baseline."""
        if not self.hourly_rewards:
            return 1.0
        
        current_avg = np.mean(list(self.hourly_rewards)[-6:])  # Last 6 hours
        
        if self.performance_baseline is None:
            self.performance_baseline = current_avg
            return 1.0
        
        return current_avg / max(1.0, self.performance_baseline)
    
    def get_adaptation_rate(self) -> float:
        """Get current adaptation rate (learning rate)."""
        return self.online_learning_rate
    
    def log_hourly_performance(self, hour: int):
        """Log performance metrics every hour - now minimal since it's in tqdm."""
        # Only log significant events
        if self.drift_detector.drift_detected:
            tqdm.write(f"⚠️ Performance drift detected at hour {hour}")
    
    def log_daily_performance(self, day: int):
        """Log detailed daily performance."""
        print(f"\n{'='*50}")
        print(f"Day {day} Summary:")
        print(f"  Daily reward: {self.current_day_reward:.2f}")
        print(f"  Hourly average: {np.mean(self.hourly_rewards):.2f}")
        print(f"  Total steps: {self.total_steps}")
        print(f"  Success rate: {self.successful_moves/(self.successful_moves+self.failed_moves):.2%}")
        
        # Show move type distribution
        print(f"  Move distribution:")
        total_moves = sum(self.move_type_counts.values())
        for move_type, count in sorted(self.move_type_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            percentage = count / total_moves * 100 if total_moves > 0 else 0
            print(f"    {self.move_type_abbrev.get(move_type, move_type)}: {count} ({percentage:.1f}%)")
        
        print(f"  Adaptation rate: {self.online_learning_rate:.6f}")
        print(f"  Performance ratio: {self.get_performance_ratio():.2%}")
        print(f"{'='*50}\n")
    
    def save_continuous_checkpoint(self, day: int):
        """Save checkpoint with continuous learning context."""
        checkpoint_path = os.path.join(self.save_dir, f'day_{day:04d}.pt')
        
        # Save agent
        self.agent.save(checkpoint_path)
        
        # Save continuous learning state - FIXED: Convert numpy types to Python natives
        cl_state = {
            'day': int(day),
            'total_steps': int(self.total_steps),
            'total_reward': float(self.total_reward),
            'performance_baseline': float(self.performance_baseline) if self.performance_baseline else None,
            'online_learning_rate': float(self.online_learning_rate),
            'hourly_rewards': [float(r) for r in self.hourly_rewards],  # Convert to float
            'daily_rewards': [float(r) for r in self.daily_rewards],   # Convert to float
            'terminal_config': self.terminal_config
        }
        
        state_path = checkpoint_path.replace('.pt', '_state.json')
        with open(state_path, 'w') as f:
            json.dump(cl_state, f, indent=2)
    
    def create_agent(self, agent_type: str, state_dim: int):
        """Create the specified agent type."""
        # Import here to avoid circular imports
        from simulation.agents.hierarchical_agent import HierarchicalAttentionAgent
        from simulation.agents.gnn_agent import GNNAgent
        
        if agent_type == 'hierarchical':
            self.agent = HierarchicalAttentionAgent(state_dim, device=self.device)
        elif agent_type == 'gnn':
            self.agent = GNNAgent(
                state_dim,
                self.terminal_config['n_rows'],
                self.terminal_config['n_bays'],
                self.terminal_config['n_tiers'],
                device=self.device
            )
    
    def create_move_embeddings(self, available_moves: Dict, info: Dict) -> torch.Tensor:
        """Create embeddings for available moves."""
        from simulation.agents.base_agent import MoveEmbedder
        
        embedder = MoveEmbedder(
            self.terminal_config['n_rows'],
            self.terminal_config['n_bays'],
            self.terminal_config['n_tiers']
        )
        
        embeddings = []
        terminal_state = self._get_terminal_state_context(info)
        
        for move_id, move_data in available_moves.items():
            embedding = embedder.embed_move(move_data, None, terminal_state)
            embeddings.append(embedding)
        
        # Fix: Convert list of numpy arrays to single numpy array first
        embeddings_array = np.stack(embeddings) if embeddings else np.array([])
        return torch.from_numpy(embeddings_array).float().to(self.device)
    
    def _get_terminal_state_context(self, info: Dict) -> Dict[str, float]:
        """Extract normalized terminal state features."""
        return {
            'train_urgency': min(info.get('trains_departing_soon', 0) / 5.0, 1.0),
            'truck_waiting': min(info.get('trucks_waiting', 0) / 10.0, 1.0),
            'yard_congestion': info.get('containers_in_yard', 0) / 
                             (self.terminal_config['n_rows'] * 
                              self.terminal_config['n_bays'] * 
                              self.terminal_config['n_tiers']),
            'time_pressure': info.get('time_of_day', 0.5)
        }
    
    def set_learning_rate(self, lr: float):
        """Set learning rate for all optimizers."""
        if hasattr(self.agent, 'actor_optimizer'):
            for param_group in self.agent.actor_optimizer.param_groups:
                param_group['lr'] = lr
        if hasattr(self.agent, 'critic_optimizer'):
            for param_group in self.agent.critic_optimizer.param_groups:
                param_group['lr'] = lr
    
    def get_current_learning_rate(self) -> float:
        """Get current learning rate."""
        if hasattr(self.agent, 'actor_optimizer'):
            return self.agent.actor_optimizer.param_groups[0]['lr']
        return self.online_learning_rate


class DriftDetector:
    """Detect concept drift in reward distribution."""
    
    def __init__(self, window_size: int = 1000, threshold: float = 3.0):
        self.window_size = window_size
        self.threshold = threshold
        self.rewards = deque(maxlen=window_size)
        self.baseline_mean = None
        self.baseline_std = None
        
    def detect_drift(self, reward: float) -> bool:
        """Detect if current performance indicates drift."""
        self.rewards.append(reward)
        
        if len(self.rewards) < 100:
            return False
        
        # Establish baseline
        if self.baseline_mean is None:
            self.baseline_mean = np.mean(self.rewards)
            self.baseline_std = np.std(self.rewards) + 1e-6
            return False
        
        # Check recent performance
        recent_rewards = list(self.rewards)[-50:]
        recent_mean = np.mean(recent_rewards)
        
        # Z-score test
        z_score = abs(recent_mean - self.baseline_mean) / self.baseline_std
        
        if z_score > self.threshold:
            # Update baseline after drift
            self.baseline_mean = recent_mean
            self.baseline_std = np.std(recent_rewards) + 1e-6
            return True
        
        return False


def main():
    """Main continuous training script."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Continuous Adaptive Terminal Training')
    parser.add_argument('--agent', type=str, default='hierarchical',
                       choices=['hierarchical', 'gnn'])
    parser.add_argument('--days', type=int, default=365,
                       help='Days to run continuous training')
    parser.add_argument('--rows', type=int, default=5)
    parser.add_argument('--bays', type=int, default=10)
    parser.add_argument('--tiers', type=int, default=3)
    
    args = parser.parse_args()
    
    terminal_config = {
        'n_rows': args.rows,
        'n_bays': args.bays,
        'n_tiers': args.tiers,
        'n_railtracks': 6,
        'split_factor': 4,
        'max_days': args.days + 1
    }
    
    trainer = ContinuousAdaptiveTrainer(
        agent_type=args.agent,
        terminal_config=terminal_config
    )
    
    trainer.train_continuous(target_days=args.days)


if __name__ == '__main__':
    main()