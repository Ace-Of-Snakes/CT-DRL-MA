# training_optimized.py (GPU-accelerated version)

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time
import argparse
from tqdm import tqdm
import pandas as pd
import sys
from pathlib import Path
import logging
import json

# Add project root to path to ensure imports work correctly
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our modules
from simulation.agents.dual_head_q import OptimizedTerminalAgent
from simulation.terminal_env_optimized import OptimizedTerminalEnvironment  # Updated import
from simulation import TerminalConfig

class OptimizedTrainingLogger:
    """Enhanced logger for GPU-accelerated training with performance monitoring."""
    
    def __init__(self, log_dir='logs', experiment_name=None, quiet_console=True, device='cpu'):
        """
        Initialize optimized logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (auto-generated if None)
            quiet_console: If True, only show warnings, errors and tqdm on console
            device: Training device for performance monitoring
        """
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            device_suffix = "gpu" if device == 'cuda' else "cpu"
            experiment_name = f"terminal_training_{device_suffix}_{timestamp}"
        
        self.experiment_name = experiment_name
        self.log_path = os.path.join(log_dir, f"{experiment_name}.log")
        self.quiet_console = quiet_console
        self.device = device
        
        # Configure logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create file handler - ALWAYS logs everything to file
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler - set to WARNING level if quiet_console is True
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING if quiet_console else logging.INFO)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize metrics tracking with GPU performance
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'curriculum_stages': [],
            'action_types': {'crane': 0, 'truck_parking': 0, 'terminal_truck': 0},
            'times': [],
            'gpu_utilization': [],  # Track GPU usage if available
            'action_mask_times': [],  # Track action mask generation time
            'storage_operations': [],  # Track storage operation counts
            'proximity_calculations': []  # Track proximity calculation counts
        }
        
        self.start_time = time.time()
        
        # Log start message
        self.logger.info(f"Started optimized training experiment: {experiment_name}")
        self.logger.info(f"Training device: {device}")
        self.logger.info(f"Log file: {self.log_path}")
        
        # GPU information
        if device == 'cuda' and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        
        # Important messages should use warning level to always show on console
        if quiet_console:
            self.logger.warning(f"Running in quiet mode - only warnings and errors will show on console")
            self.logger.warning(f"All logs are still saved to: {self.log_path}")
    
    def log_episode(self, episode, reward, steps, loss, stage, action_counts=None, performance_stats=None):
        """
        Log episode metrics with performance statistics.
        
        Args:
            episode: Episode number
            reward: Episode total reward
            steps: Number of steps in episode
            loss: Mean loss for episode
            stage: Curriculum stage
            action_counts: Dictionary of action type counts
            performance_stats: Performance statistics from environment
        """
        # Record metrics
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_lengths'].append(steps)
        self.metrics['losses'].append(loss if loss is not None else 0)
        self.metrics['curriculum_stages'].append(stage)
        self.metrics['times'].append(time.time() - self.start_time)
        
        # Record performance statistics
        if performance_stats:
            self.metrics['action_mask_times'].append(performance_stats.get('action_mask_time', 0))
            self.metrics['storage_operations'].append(performance_stats.get('storage_operations', 0))
            self.metrics['proximity_calculations'].append(performance_stats.get('proximity_calculations', 0))
        
        # Update action counts if provided
        if action_counts:
            for action_type, count in action_counts.items():
                if action_type in self.metrics['action_types']:
                    self.metrics['action_types'][action_type] += count
        
        # GPU utilization tracking
        if self.device == 'cuda' and torch.cuda.is_available():
            gpu_util = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 0
            self.metrics['gpu_utilization'].append(gpu_util)
        
        # Calculate averages for logging
        avg_reward = np.mean(self.metrics['episode_rewards'][-10:]) if len(self.metrics['episode_rewards']) >= 10 else np.mean(self.metrics['episode_rewards'])
        avg_steps = np.mean(self.metrics['episode_lengths'][-10:]) if len(self.metrics['episode_lengths']) >= 10 else np.mean(self.metrics['episode_lengths'])
        
        # Format loss string properly outside the f-string
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        
        # Log episode summary
        self.logger.info(f"Episode {episode} | Stage {stage} | Reward: {reward:.2f} | Avg(10): {avg_reward:.2f} | Steps: {steps} | Loss: {loss_str}")
        
        # Log performance statistics every 10 episodes
        if episode % 10 == 0:
            if action_counts:
                action_str = " | ".join([f"{key}: {count}" for key, count in action_counts.items()])
                self.logger.info(f"Action counts for last episode: {action_str}")
            
            if performance_stats:
                perf_str = f"Mask time: {performance_stats.get('action_mask_time', 0):.4f}s | "
                perf_str += f"Storage ops: {performance_stats.get('storage_operations', 0)} | "
                perf_str += f"Proximity calcs: {performance_stats.get('proximity_calculations', 0)}"
                self.logger.info(f"Performance stats: {perf_str}")
                
            # GPU memory usage
            if self.device == 'cuda' and torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated(0) / 1e9
                gpu_memory_cached = torch.cuda.memory_reserved(0) / 1e9
                self.logger.info(f"GPU memory: {gpu_memory_used:.2f}GB used, {gpu_memory_cached:.2f}GB cached")
    
    def save_metrics(self, results_dir='results'):
        """Save training metrics with performance data to files."""
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics to CSV
        results_file = os.path.join(results_dir, f"{self.experiment_name}_metrics_{timestamp}.csv")
        
        # Prepare data for CSV
        base_data = {
            'episode': range(1, len(self.metrics['episode_rewards']) + 1),
            'reward': self.metrics['episode_rewards'],
            'length': self.metrics['episode_lengths'],
            'loss': self.metrics['losses'],
            'stage': self.metrics['curriculum_stages'],
            'time': self.metrics['times']
        }
        
        # Add performance metrics if available
        if self.metrics['action_mask_times']:
            base_data['action_mask_time'] = self.metrics['action_mask_times']
        if self.metrics['storage_operations']:
            base_data['storage_operations'] = self.metrics['storage_operations']
        if self.metrics['proximity_calculations']:
            base_data['proximity_calculations'] = self.metrics['proximity_calculations']
        if self.metrics['gpu_utilization']:
            base_data['gpu_utilization'] = self.metrics['gpu_utilization']
        
        results_df = pd.DataFrame(base_data)
        results_df.to_csv(results_file, index=False)
        self.logger.info(f"Training metrics saved to {results_file}")
        
        # Save action counts
        action_counts_file = os.path.join(results_dir, f"{self.experiment_name}_actions_{timestamp}.json")
        with open(action_counts_file, 'w') as f:
            json.dump(self.metrics['action_types'], f, indent=2)
        
        # Create and save plots
        self._save_training_plots(results_dir, timestamp)
    
    def _save_training_plots(self, results_dir, timestamp):
        """Create and save enhanced plots of training metrics."""
        # Create figure with more subplots for performance metrics
        fig, axs = plt.subplots(3, 2, figsize=(15, 15))
        
        # Plot 1: rewards
        axs[0, 0].plot(self.metrics['episode_rewards'])
        axs[0, 0].set_title('Episode Rewards')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].grid(True)
        
        # Plot 2: smoothed rewards
        if len(self.metrics['episode_rewards']) > 10:
            smoothed_rewards = pd.Series(self.metrics['episode_rewards']).rolling(10).mean()
            axs[0, 1].plot(smoothed_rewards)
            axs[0, 1].set_title('Smoothed Episode Rewards (10-ep window)')
            axs[0, 1].set_xlabel('Episode')
            axs[0, 1].set_ylabel('Reward')
            axs[0, 1].grid(True)
        
        # Plot 3: episode lengths
        axs[1, 0].plot(self.metrics['episode_lengths'])
        axs[1, 0].set_title('Episode Lengths')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Steps')
        axs[1, 0].grid(True)
        
        # Plot 4: losses
        if any(self.metrics['losses']):
            axs[1, 1].plot(self.metrics['losses'])
            axs[1, 1].set_title('Training Loss')
            axs[1, 1].set_xlabel('Episode')
            axs[1, 1].set_ylabel('Loss')
            axs[1, 1].grid(True)
        
        # Plot 5: Action mask generation time
        if self.metrics['action_mask_times']:
            axs[2, 0].plot(self.metrics['action_mask_times'])
            axs[2, 0].set_title('Action Mask Generation Time')
            axs[2, 0].set_xlabel('Episode')
            axs[2, 0].set_ylabel('Time (seconds)')
            axs[2, 0].grid(True)
        
        # Plot 6: GPU utilization (if available)
        if self.metrics['gpu_utilization']:
            axs[2, 1].plot(self.metrics['gpu_utilization'])
            axs[2, 1].set_title('GPU Utilization')
            axs[2, 1].set_xlabel('Episode')
            axs[2, 1].set_ylabel('Utilization (%)')
            axs[2, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(results_dir, f"{self.experiment_name}_plots_{timestamp}.png")
        plt.savefig(plot_file, dpi=300)
        plt.close()
        self.logger.info(f"Training plots saved to {plot_file}")

def action_to_tuple(action, action_type):
    """Convert action dictionary to tuple based on action type."""
    if action_type == 0:  # Crane Movement
        return tuple(action['crane_movement'])
    elif action_type == 1:  # Truck Parking
        return tuple(action['truck_parking'])
    elif action_type == 2:  # Terminal Truck
        return tuple(action['terminal_truck'])
    return None

class OptimizedCurriculumTrainer:
    """
    GPU-accelerated curriculum learning trainer for container terminal agents.
    """
    
    def __init__(self, 
                 base_config_path=None, 
                 checkpoints_dir='checkpoints',
                 results_dir='results',
                 log_dir='logs',
                 experiment_name=None,
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 quiet_console=True):
        """
        Initialize the optimized curriculum trainer.
        
        Args:
            base_config_path: Path to base configuration file
            checkpoints_dir: Directory to save model checkpoints
            results_dir: Directory to save training results
            log_dir: Directory to save training logs
            experiment_name: Name for this experiment
            device: Computation device ('cuda' or 'cpu')
            quiet_console: Whether to suppress console output
        """
        self.base_config_path = base_config_path
        self.checkpoints_dir = checkpoints_dir
        self.results_dir = results_dir
        self.device = device
        
        # Create directories if they don't exist
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize optimized logger
        self.logger = OptimizedTrainingLogger(
            log_dir, experiment_name, quiet_console=quiet_console, device=device
        )
        
        # Log GPU/CPU setup
        if device == 'cuda' and torch.cuda.is_available():
            self.logger.logger.info(f"Using GPU acceleration: {torch.cuda.get_device_name(0)}")
            self.logger.logger.info(f"CUDA version: {torch.version.cuda}")
            self.logger.logger.info(f"PyTorch version: {torch.__version__}")
        else:
            self.logger.logger.info("Using CPU training")
        
        # Curriculum stages configuration (optimized for GPU training)
        self.curriculum_stages = [
            # Stage 1: Basic terminal operations (small terminal for faster GPU training)
            {
                'name': 'Basic Operations (GPU Optimized)',
                'num_railtracks': 3,
                'num_railslots_per_track': 15,
                'num_storage_rows': 4,
                'num_terminal_trucks': 2,
                'max_trucks_per_day': 24,
                'max_trains_per_day': 8,
                'target_reward': 75,
                'max_episodes': 150,
                'max_simulation_time': 86400 * 3  # 3 days
            },
            # Stage 2: Intermediate terminal (medium size)
            {
                'name': 'Intermediate Terminal (GPU Optimized)',
                'num_railtracks': 4,
                'num_railslots_per_track': 20,
                'num_storage_rows': 5,
                'num_terminal_trucks': 3,
                'max_trucks_per_day': 15,
                'max_trains_per_day': 4,
                'target_reward': 150,
                'max_episodes': 200,
                'max_simulation_time': 86400 * 5  # 5 days
            },
            # Stage 3: Advanced terminal (full size)
            {
                'name': 'Advanced Terminal (GPU Optimized)',
                'num_railtracks': 6,
                'num_railslots_per_track': 29,
                'num_storage_rows': 5,
                'num_terminal_trucks': 3,
                'max_trucks_per_day': 25,
                'max_trains_per_day': 6,
                'target_reward': 250,
                'max_episodes': 300,
                'max_simulation_time': 86400 * 7  # 7 days
            }
        ]
        
        # Initialize current stage
        self.current_stage = 0
    
    def create_environment(self, stage_config):
        """Create optimized environment with GPU acceleration for current curriculum stage."""
        # Load base configuration
        terminal_config = TerminalConfig(self.base_config_path)
        
        # Create environment with GPU acceleration
        env = OptimizedTerminalEnvironment(
            terminal_config=terminal_config,
            max_simulation_time=stage_config['max_simulation_time'],
            num_cranes=2,  # Fixed for consistency
            num_terminal_trucks=stage_config['num_terminal_trucks'],
            device=self.device  # GPU acceleration
        )
        
        # Modify the terminal layout with our curriculum parameters
        env.terminal = self._create_custom_terminal(
            stage_config['num_railtracks'],
            stage_config['num_railslots_per_track'],
            stage_config['num_storage_rows']
        )
        
        # Re-initialize components that depend on terminal layout
        env._setup_position_mappings()
        env.storage_yard = self._create_custom_storage_yard(env)
        
        # Set vehicle limits
        env.set_vehicle_limits(
            max_trucks=stage_config['max_trucks_per_day'],
            max_trains=stage_config['max_trains_per_day']
        )
        
        # Enable simplified rendering for faster training
        env.set_simplified_rendering(True)
        
        return env, terminal_config
        
    def _create_custom_terminal(self, num_railtracks, num_railslots_per_track, num_storage_rows):
        """Create a container terminal with custom dimensions."""
        from simulation.terminal_layout.CTSimulator import ContainerTerminal
        
        return ContainerTerminal(
            layout_order=['rails', 'parking', 'driving_lane', 'yard_storage'],
            num_railtracks=num_railtracks,
            num_railslots_per_track=num_railslots_per_track,
            num_storage_rows=num_storage_rows,
            # Keep ratios the same
            parking_to_railslot_ratio=1.0,
            storage_to_railslot_ratio=2.0,
            # Keep physical dimensions the same
            rail_slot_length=24.384,
            track_width=2.44,
            space_between_tracks=2.05,
            space_rails_to_parking=1.05,
            space_driving_to_storage=0.26,
            parking_width=4.0,
            driving_lane_width=4.0,
            storage_slot_width=2.5
        )
        
    def _create_custom_storage_yard(self, env):
        """Create an optimized bitmap storage yard matching the terminal layout."""
        from simulation.terminal_components.BitmapYard import BitmapStorageYard
        
        # Define special areas for different container types
        special_areas = {
            'reefer': [],
            'dangerous': [],
            'trailer': [],  # Specialized area for trailers
            'swap_body': []  # Specialized area for swap bodies
        }
        
        # Add reefer areas in first and last column of each row
        for row in env.terminal.storage_row_names:
            special_areas['reefer'].append((row, 1, 1))
            last_bay = env.terminal.num_storage_slots_per_row
            special_areas['reefer'].append((row, last_bay, last_bay))
            
        # Add dangerous goods area in middle columns
        middle_bay = env.terminal.num_storage_slots_per_row // 2
        for row in env.terminal.storage_row_names:
            special_areas['dangerous'].append((row, middle_bay-1, middle_bay+1))
        
        # Add trailer and swap body areas in the first row (closest to driving lane)
        first_row = env.terminal.storage_row_names[0]
        trailer_start = int(env.terminal.num_storage_slots_per_row * 0.2)
        trailer_end = int(env.terminal.num_storage_slots_per_row * 0.35)
        swap_body_start = int(env.terminal.num_storage_slots_per_row * 0.6)
        swap_body_end = int(env.terminal.num_storage_slots_per_row * 0.75)
        
        special_areas['trailer'].append((first_row, trailer_start, trailer_end))
        special_areas['swap_body'].append((first_row, swap_body_start, swap_body_end))
        
        return BitmapStorageYard(
            num_rows=env.terminal.num_storage_rows,
            num_bays=env.terminal.num_storage_slots_per_row,
            max_tier_height=4,
            row_names=env.terminal.storage_row_names,
            special_areas=special_areas,
            device=self.device  # GPU acceleration
        )
    
    def create_agent(self, env):
        """Create agent for the current environment with GPU acceleration."""
        # Get the state dimension
        sample_obs, _ = env.reset()
        flat_state = self._flatten_state(sample_obs)
        state_dim = len(flat_state)
        
        # Get action dimensions
        action_dims = {
            'crane_movement': env.action_space['crane_movement'].nvec,
            'truck_parking': env.action_space['truck_parking'].nvec,
            'terminal_truck': env.action_space['terminal_truck'].nvec
        }
        
        # Create agent with GPU support
        agent = OptimizedTerminalAgent(
            state_dim, 
            action_dims,
            hidden_dims=[512, 512],  # Larger networks for GPU
            head_dims=[256],
            device=self.device  # GPU acceleration
        )
        return agent
    
    def _flatten_state(self, state):
        """Flatten state dictionary to vector for the agent."""
        # Extract relevant features and flatten
        crane_positions = state['crane_positions'].flatten()
        crane_available_times = state['crane_available_times'].flatten()
        terminal_truck_available_times = state['terminal_truck_available_times'].flatten()
        current_time = state['current_time'].flatten()
        yard_state = state['yard_state'].flatten()
        parking_status = state['parking_status'].flatten()
        rail_status = state['rail_status'].flatten()
        queue_sizes = state['queue_sizes'].flatten()
        
        # Concatenate all features
        flat_state = np.concatenate([
            crane_positions, 
            crane_available_times,
            terminal_truck_available_times,
            current_time, 
            yard_state, 
            parking_status, 
            rail_status, 
            queue_sizes
        ])
        
        return flat_state
    
    def train(self, total_episodes=None, resume_checkpoint=None):
        """
        Train agents through curriculum stages with GPU acceleration.
        
        Args:
            total_episodes: Total episodes to train (overrides per-stage limits)
            resume_checkpoint: Path to checkpoint for resuming training
            
        Returns:
            Trained agent and training metrics
        """
        # Initialize tracking variables
        episode_count = 0
        
        # Initialize agent with first stage environment
        self.logger.logger.info(f"Creating GPU-accelerated environment for stage 1 ({self.curriculum_stages[0]['name']})...")
        env, _ = self.create_environment(self.curriculum_stages[self.current_stage])
        agent = self.create_agent(env)
        
        # Load checkpoint if resuming
        if resume_checkpoint:
            self.logger.logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
            agent.load(resume_checkpoint)
        
        self.logger.logger.info(f"Starting GPU-accelerated curriculum training with {len(self.curriculum_stages)} stages")
        stage_rewards = []
        
        try:
            # Train through curriculum stages
            while self.current_stage < len(self.curriculum_stages):
                stage_config = self.curriculum_stages[self.current_stage]
                stage_name = stage_config['name']
                self.logger.logger.info(f"\nStarting Stage {self.current_stage + 1}: {stage_name}")
                
                # Create environment for current stage
                env, _ = self.create_environment(stage_config)
                
                # Reset stage tracking
                stage_rewards = []
                stage_episodes = 0
                max_stage_episodes = stage_config['max_episodes']
                
                # Set max episodes if provided
                if total_episodes is not None:
                    max_stage_episodes = min(max_stage_episodes, total_episodes - episode_count)
                
                # Training progress bar
                stage_pbar = tqdm(total=max_stage_episodes, desc=f"Stage {self.current_stage + 1}/{len(self.curriculum_stages)}")
                
                # Train for this stage
                for episode in range(max_stage_episodes):
                    # Track episode
                    episode_count += 1
                    stage_episodes += 1
                    
                    # Run episode
                    try:
                        episode_reward, episode_steps, mean_loss, action_counts, performance_stats = self.train_episode(env, agent)
                        
                        # Track metrics
                        stage_rewards.append(episode_reward)
                        
                        # Log episode with performance stats
                        self.logger.log_episode(
                            episode_count, 
                            episode_reward,
                            episode_steps,
                            mean_loss,
                            self.current_stage + 1,
                            action_counts,
                            performance_stats
                        )
                        
                        # Update progress bar
                        avg_reward = np.mean(stage_rewards[-10:]) if len(stage_rewards) >= 10 else np.mean(stage_rewards)
                        stage_pbar.set_postfix({
                            'reward': f"{episode_reward:.2f}",
                            'avg10': f"{avg_reward:.2f}",
                            'steps': episode_steps,
                            'loss': f"{mean_loss:.4f}" if mean_loss else "N/A"
                        })
                        stage_pbar.update(1)
                        
                        # Save checkpoint periodically
                        if episode_count % 25 == 0:  # More frequent saves for GPU training
                            self.save_checkpoint(agent, episode_count)
                        
                        # Check if we reached the target reward
                        if avg_reward >= stage_config['target_reward'] and stage_episodes >= 20:
                            self.logger.logger.info(f"Target reward of {stage_config['target_reward']} reached!")
                            self.logger.log_stage_complete(self.current_stage + 1, stage_episodes, avg_reward)
                            
                            # Save checkpoint before advancing
                            self.save_checkpoint(agent, episode_count, f"stage_{self.current_stage + 1}_complete")
                            break
                    
                    except Exception as e:
                        self.logger.logger.error(f"Error during episode: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue with next episode
                
                # Close progress bar
                stage_pbar.close()
                
                # Advance to next stage
                self.current_stage += 1
                if self.current_stage >= len(self.curriculum_stages):
                    self.logger.logger.info("GPU-accelerated curriculum training complete!")
                    break
                    
                # Clear replay buffer before starting the next stage to avoid dimension mismatches
                self.logger.logger.info("Clearing replay buffer for new stage...")
                agent.replay_buffer.clear()
                
                # Clear GPU cache between stages
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.logger.info("GPU cache cleared")
            
            # Save final metrics and model
            training_time = time.time() - self.logger.start_time
            self.logger.log_training_complete(episode_count, training_time)
            self.logger.save_metrics(self.results_dir)
            self.save_checkpoint(agent, episode_count, "final")
            
        except KeyboardInterrupt:
            self.logger.logger.info("\nTraining interrupted by user!")
            # Save checkpoint and results so far
            training_time = time.time() - self.logger.start_time
            self.logger.log_training_complete(episode_count, training_time)
            self.logger.save_metrics(self.results_dir)
            self.save_checkpoint(agent, episode_count, "interrupted")
        except Exception as e:
            self.logger.logger.error(f"Unexpected error during training: {e}")
            import traceback
            traceback.print_exc()
            # Save checkpoint and results so far
            training_time = time.time() - self.logger.start_time
            self.logger.log_training_complete(episode_count, training_time)
            self.logger.save_metrics(self.results_dir)
            self.save_checkpoint(agent, episode_count, "error")
        
        return agent, self.logger.metrics
    
    def train_episode(self, env: OptimizedTerminalEnvironment, agent: OptimizedTerminalAgent):
        """Train for a single episode with GPU-accelerated operations."""
        # Reset environment
        state, info = env.reset()
        episode_reward = 0
        episode_steps = 0
        losses = []
        done = False
        
        # Extract performance stats from reset
        performance_stats = info.get('performance_stats', {})
        
        # Calculate maximum simulation time as 90% of env's maximum
        max_simulation_time = env.max_simulation_time * 0.9
        
        # Action tracking
        action_counts = {'crane': 0, 'truck_parking': 0, 'terminal_truck': 0}
        
        # Time advancement tracking
        consecutive_wait_count = 0
        last_stack_check_time = env.current_simulation_time
        
        # Main training loop with GPU acceleration
        while not done and env.current_simulation_time < max_simulation_time:
            # Check for pre-marshalling needs using GPU operations
            needs_premarshalling = hasattr(env, 'evaluate_need_for_premarshalling') and env.evaluate_need_for_premarshalling()
            
            # Select action based on current state
            action_masks = state['action_mask']
            
            # Use GPU-accelerated action selection
            if needs_premarshalling:
                # Try to get a pre-marshalling action if the method exists
                if hasattr(agent, 'select_premarshalling_action'):
                    action, action_type, flat_state = agent.select_premarshalling_action(state, action_masks, env)
                else:
                    # Fallback to regular action selection
                    action, action_type, flat_state = agent.select_action(state, action_masks, env)
            else:
                # Normal action selection
                action, action_type, flat_state = agent.select_action(state, action_masks, env)
            
            # Handle case where no actions are available
            if action is None:
                consecutive_wait_count += 1
                
                # Check stacks every 15 minutes of simulation time
                stack_check_interval = 15 * 60  # 15 minutes
                time_since_last_check = env.current_simulation_time - last_stack_check_time
                
                # First, use small time increments to check frequently for new optimization opportunities
                if time_since_last_check >= stack_check_interval:
                    # Update last check time
                    last_stack_check_time = env.current_simulation_time
                    
                    # Check if pre-marshalling is needed now
                    needs_premarshalling = hasattr(env, 'evaluate_need_for_premarshalling') and env.evaluate_need_for_premarshalling()
                    
                    # If no pre-marshalling needed and no vehicles, advance time more aggressively
                    if not needs_premarshalling and env.truck_queue.size() == 0 and env.train_queue.size() == 0:
                        # More aggressive time jump - but still max 1 hour as requested
                        advance_time = min(60 * 60, 5 * 60 * consecutive_wait_count)  # Increase time jumps but cap at 1 hour
                    else:
                        # Reset consecutive wait counter if we found something to do
                        consecutive_wait_count = 0
                        advance_time = 5 * 60  # 5 minutes
                else:
                    # Small time jumps between stack checks
                    advance_time = 5 * 60  # 5 minutes
                
                # Execute a wait action and force time advancement
                wait_action = {'action_type': 0, 'crane_movement': np.array([0, 0, 0]), 'truck_parking': np.array([0, 0])}
                next_state, reward, done, truncated, info = env.step(wait_action)
                
                # Force time advancement
                env.current_simulation_time += advance_time
                env.current_simulation_datetime += timedelta(seconds=advance_time)
                
                # Process vehicles
                env._process_vehicle_arrivals(advance_time)
                env._process_vehicle_departures()
                
                episode_steps += 1
                
                # Update state and performance stats
                next_state = env._get_observation()
                state = next_state
                continue
                
            # Valid action found - reset consecutive wait counter
            consecutive_wait_count = 0
            
            # Track action type
            if action_type == 0:
                action_counts['crane'] += 1
            elif action_type == 1:
                action_counts['truck_parking'] += 1
            elif action_type == 2:
                action_counts['terminal_truck'] += 1
            
            # Execute action in environment with GPU acceleration
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # Update performance stats
            episode_performance_stats = info.get('performance_stats', {})
            for key, value in episode_performance_stats.items():
                if key in performance_stats:
                    performance_stats[key] = max(performance_stats[key], value)
                else:
                    performance_stats[key] = value
            
            # Store experience and update networks with GPU acceleration
            next_flat_state = self._flatten_state(next_state) if next_state is not None else None
            agent.store_experience(
                flat_state, 
                action_to_tuple(action, action_type), 
                action_type, 
                reward, 
                next_flat_state, 
                done or truncated, 
                action_masks
            )
            
            # Update networks with GPU acceleration
            if len(agent.replay_buffer) > agent.batch_size:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)
            
            # Update state
            state = next_state
        
        # Calculate mean loss
        mean_loss = np.mean(losses) if losses else None
        
        return episode_reward, episode_steps, mean_loss, action_counts, performance_stats
    
    def save_checkpoint(self, agent, episode, suffix=None):
        """Save agent checkpoint with GPU state."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.logger.experiment_name}_ep{episode}"
        if suffix:
            filename += f"_{suffix}"
        filename += f"_{timestamp}.pt"
        filepath = os.path.join(self.checkpoints_dir, filename)
        agent.save(filepath)
        self.logger.logger.info(f"GPU checkpoint saved to {filepath}")
    
    def evaluate(self, agent, num_episodes=5, render=False):
        """
        Evaluate trained agent on the full environment with GPU acceleration.
        
        Args:
            agent: Trained agent
            num_episodes: Number of evaluation episodes
            render: Whether to render the environment
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Create full-sized environment for evaluation
        full_config = self.curriculum_stages[-1]
        env, _ = self.create_environment(full_config)
        
        # Disable simplified rendering for evaluation
        env.set_simplified_rendering(not render)
        
        # Run evaluation episodes
        eval_rewards = []
        eval_steps = []
        eval_times = []
        action_counts = {'crane': 0, 'truck_parking': 0, 'terminal_truck': 0}
        performance_stats = {'action_mask_time': 0, 'storage_operations': 0, 'proximity_calculations': 0}
        
        self.logger.logger.info(f"\nEvaluating agent with GPU acceleration for {num_episodes} episodes...")
        eval_pbar = tqdm(total=num_episodes, desc="Evaluating")
        
        for episode in range(num_episodes):
            state, info = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done and episode_steps < 2000:  # Limit eval episodes
                # Select action with no exploration
                action_masks = state['action_mask']
                action, action_type, _ = agent.select_action(state, action_masks, env, epsilon=0)
                
                # Handle case where no actions are available
                if action is None:
                    # Wait until next available time
                    advance_time = 300  # 5 minutes
                    env.current_simulation_time += advance_time
                    env.current_simulation_datetime += timedelta(seconds=advance_time)
                    env._process_vehicle_arrivals(advance_time)
                    env._process_vehicle_departures()
                    next_state = env._get_observation()
                else:
                    # Take action in environment
                    next_state, reward, done, truncated, info = env.step(action)
                    episode_reward += reward
                    
                    # Update action count
                    if action_type == 0:
                        action_counts['crane'] += 1
                    elif action_type == 1:
                        action_counts['truck_parking'] += 1
                    elif action_type == 2:
                        action_counts['terminal_truck'] += 1
                    
                    # Update performance stats
                    episode_perf_stats = info.get('performance_stats', {})
                    for key, value in episode_perf_stats.items():
                        if key in performance_stats:
                            performance_stats[key] += value
                        else:
                            performance_stats[key] = value
                
                episode_steps += 1
                
                # Render if requested
                if render:
                    env.render()
                    time.sleep(0.1)
                
                # Update state
                state = next_state
                done = done or truncated
            
            # Track metrics
            eval_rewards.append(episode_reward)
            eval_steps.append(episode_steps)
            eval_times.append(env.current_simulation_time / 86400)  # Convert to days
            
            # Update progress bar
            eval_pbar.update(1)
        
        eval_pbar.close()
        
        # Calculate evaluation metrics
        evaluation_metrics = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_steps': np.mean(eval_steps),
            'mean_days': np.mean(eval_times),
            'action_counts': action_counts,
            'performance_stats': performance_stats
        }
        
        # Log evaluation results
        self.logger.logger.info("\nEvaluation Results:")
        self.logger.logger.info(f"  Mean Reward: {evaluation_metrics['mean_reward']:.2f} ± {evaluation_metrics['std_reward']:.2f}")
        self.logger.logger.info(f"  Mean Steps: {evaluation_metrics['mean_steps']:.2f}")
        self.logger.logger.info(f"  Mean Simulation Days: {evaluation_metrics['mean_days']:.2f}")
        
        # Log action distribution
        action_str = " | ".join([f"{key}: {count}" for key, count in action_counts.items()])
        self.logger.logger.info(f"  Action counts: {action_str}")
        
        # Log performance statistics
        perf_str = " | ".join([f"{key}: {value}" for key, value in performance_stats.items()])
        self.logger.logger.info(f"  Performance stats: {perf_str}")
        
        return evaluation_metrics

def parse_args():
    """Parse command line arguments with GPU options."""
    parser = argparse.ArgumentParser(description='Train terminal agents with GPU-accelerated curriculum learning')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=None, help='Total number of episodes')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Checkpoints directory')
    parser.add_argument('--results', type=str, default='results', help='Results directory')
    parser.add_argument('--logs', type=str, default='logs', help='Logs directory')
    parser.add_argument('--device', type=str, default='auto', 
                      choices=['auto', 'cuda', 'cpu'],
                      help='Computation device (auto/cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    parser.add_argument('--render', action='store_true', help='Render during evaluation')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--force-cpu', action='store_true', help='Force CPU training even if GPU available')
    
    return parser.parse_args()

def main():
    """Main function for GPU-accelerated training of terminal agents."""
    args = parse_args()
    
    # Determine device
    if args.force_cpu:
        device = 'cpu'
    elif args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
        
    # Validate device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = 'cpu'
    
    print(f"Using device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        # Set deterministic behavior for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Generate experiment name if not provided
    if args.name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        device_suffix = "gpu" if device == 'cuda' else "cpu"
        args.name = f"terminal_training_{device_suffix}_{timestamp}"
    
    # Create optimized curriculum trainer
    trainer = OptimizedCurriculumTrainer(
        base_config_path=args.config,
        checkpoints_dir=args.checkpoints,
        results_dir=args.results,
        log_dir=args.logs,
        experiment_name=args.name,
        device=device
    )
    
    # Train agents with GPU acceleration
    agent, _ = trainer.train(
        total_episodes=args.episodes,
        resume_checkpoint=args.resume
    )
    
    # Evaluate if requested
    if args.evaluate:
        trainer.evaluate(agent, num_episodes=5, render=args.render)
    
    # Final GPU memory cleanup
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared")

if __name__ == "__main__":
    main()