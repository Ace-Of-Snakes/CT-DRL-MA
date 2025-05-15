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
import logging
import json
from pathlib import Path

# Import previously implemented classes
from simulation.deprecated_components.OptimizedTerminalAgent import OptimizedTerminalAgent
from simulation.TerminalConfig import TerminalConfig
from simulation.deprecated_components.GPUTerminalEnvironment import GPUTerminalEnvironment

class GPUTrainingLogger:
    """GPU-optimized logger for training with file output and console display."""
    
    def __init__(self, log_dir='logs', experiment_name=None, quiet_console=True):
        """Initialize logger with GPU performance tracking."""
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"gpu_terminal_training_{timestamp}"
        
        self.experiment_name = experiment_name
        self.log_path = os.path.join(log_dir, f"{experiment_name}.log")
        self.quiet_console = quiet_console
        
        # Configure logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_path)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING if quiet_console else logging.INFO)
        
        # Create formatter and add to handlers
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Initialize metrics tracking
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'curriculum_stages': [],
            'action_types': {'crane': 0, 'truck_parking': 0, 'terminal_truck': 0},
            'times': [],
            'env_step_times': [],
            'gpu_memory_usage': [],  # Track GPU memory usage
            'training_step_times': [],  # Track training step times
        }
        
        self.start_time = time.time()
        
        # Log start message
        self.logger.info(f"Started GPU-accelerated training experiment: {experiment_name}")
        self.logger.info(f"Log file: {self.log_path}")
        
        # Log CUDA info if available
        if torch.cuda.is_available():
            cuda_device = torch.cuda.current_device()
            self.logger.info(f"CUDA Device: {torch.cuda.get_device_name(cuda_device)}")
            self.logger.info(f"CUDA Version: {torch.version.cuda}")
        else:
            self.logger.warning("CUDA not available. Training will run on CPU.")
    
    def log_episode(self, episode, reward, steps, loss, stage, action_counts=None, env_step_time=None, gpu_memory=None):
        """Log episode metrics with GPU-specific information."""
        # Record metrics
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_lengths'].append(steps)
        self.metrics['losses'].append(loss if loss is not None else 0)
        self.metrics['curriculum_stages'].append(stage)
        self.metrics['times'].append(time.time() - self.start_time)
        
        if env_step_time is not None:
            self.metrics['env_step_times'].append(env_step_time)
        
        if gpu_memory is not None:
            self.metrics['gpu_memory_usage'].append(gpu_memory)
        
        # Update action counts if provided
        if action_counts:
            for action_type, count in action_counts.items():
                if action_type in self.metrics['action_types']:
                    self.metrics['action_types'][action_type] += count
        
        # Calculate averages for logging
        avg_reward = np.mean(self.metrics['episode_rewards'][-10:]) if len(self.metrics['episode_rewards']) >= 10 else np.mean(self.metrics['episode_rewards'])
        avg_steps = np.mean(self.metrics['episode_lengths'][-10:]) if len(self.metrics['episode_lengths']) >= 10 else np.mean(self.metrics['episode_lengths'])
        
        # Format loss string
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        
        # Log episode summary
        log_str = f"Episode {episode} | Stage {stage} | Reward: {reward:.2f} | Avg(10): {avg_reward:.2f} | Steps: {steps} | Loss: {loss_str}"
        if env_step_time is not None:
            log_str += f" | Env Step: {env_step_time:.2f}ms"
        if gpu_memory is not None:
            log_str += f" | GPU Mem: {gpu_memory:.1f}MB"
        
        self.logger.info(log_str)
        
        # Log action type distribution every 10 episodes
        if episode % 10 == 0 and action_counts:
            action_str = " | ".join([f"{key}: {count}" for key, count in action_counts.items()])
            self.logger.info(f"Action counts for last episode: {action_str}")
    
    def log_stage_complete(self, stage, episodes, avg_reward):
        """Log completion of a curriculum stage."""
        self.logger.info(f"Stage {stage} completed after {episodes} episodes!")
        self.logger.info(f"Average reward for last 10 episodes: {avg_reward:.2f}")
    
    def log_training_complete(self, total_episodes, total_time):
        """Log completion of training with GPU performance stats."""
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        self.logger.info(f"Training completed! Total episodes: {total_episodes}")
        self.logger.info(f"Total training time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
        
        # Log final action type distribution
        action_counts = self.metrics['action_types']
        total_actions = sum(action_counts.values())
        if total_actions > 0:
            action_pcts = {key: count/total_actions*100 for key, count in action_counts.items()}
            action_str = " | ".join([f"{key}: {count} ({pct:.1f}%)" for key, (count, pct) in zip(action_counts.keys(), action_pcts.items())])
            self.logger.info(f"Action type distribution: {action_str}")
        
        # Log GPU performance statistics
        if self.metrics['env_step_times']:
            avg_step_time = np.mean(self.metrics['env_step_times'])
            min_step_time = np.min(self.metrics['env_step_times'])
            max_step_time = np.max(self.metrics['env_step_times'])
            self.logger.info(f"Environment performance: Avg step time: {avg_step_time:.2f}ms, Min: {min_step_time:.2f}ms, Max: {max_step_time:.2f}ms")
        
        if self.metrics['gpu_memory_usage']:
            avg_memory = np.mean(self.metrics['gpu_memory_usage'])
            max_memory = np.max(self.metrics['gpu_memory_usage'])
            self.logger.info(f"GPU memory usage: Avg: {avg_memory:.1f}MB, Max: {max_memory:.1f}MB")
    
    def save_metrics(self, results_dir='results'):
        """Save training metrics to files."""
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save metrics to CSV
        results_file = os.path.join(results_dir, f"{self.experiment_name}_metrics_{timestamp}.csv")
        results_df = pd.DataFrame({
            'episode': range(1, len(self.metrics['episode_rewards']) + 1),
            'reward': self.metrics['episode_rewards'],
            'length': self.metrics['episode_lengths'],
            'loss': self.metrics['losses'],
            'stage': self.metrics['curriculum_stages'],
            'time': self.metrics['times']
        })
        
        # Add step times if available
        if self.metrics['env_step_times']:
            padded_step_times = self.metrics['env_step_times'] + [None] * (len(self.metrics['episode_rewards']) - len(self.metrics['env_step_times']))
            results_df['env_step_time'] = padded_step_times[:len(results_df)]
        
        # Add GPU memory usage if available
        if self.metrics['gpu_memory_usage']:
            padded_memory = self.metrics['gpu_memory_usage'] + [None] * (len(self.metrics['episode_rewards']) - len(self.metrics['gpu_memory_usage']))
            results_df['gpu_memory'] = padded_memory[:len(results_df)]
            
        results_df.to_csv(results_file, index=False)
        self.logger.info(f"Training metrics saved to {results_file}")
        
        # Save action counts
        action_counts_file = os.path.join(results_dir, f"{self.experiment_name}_actions_{timestamp}.json")
        with open(action_counts_file, 'w') as f:
            json.dump(self.metrics['action_types'], f, indent=2)
        
        # Create and save plots
        self._save_training_plots(results_dir, timestamp)
    
    def _save_training_plots(self, results_dir, timestamp):
        """Create and save plots of training metrics with GPU stats."""
        # Determine number of plots
        n_plots = 3 if self.metrics['env_step_times'] else 2
        n_plots += 1 if self.metrics['gpu_memory_usage'] else 0
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, n_plots, figsize=(15, 10))
        
        # Plot rewards
        axs[0, 0].plot(self.metrics['episode_rewards'])
        axs[0, 0].set_title('Episode Rewards')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].grid(True)
        
        # Plot smoothed rewards
        if len(self.metrics['episode_rewards']) > 10:
            smoothed_rewards = pd.Series(self.metrics['episode_rewards']).rolling(10).mean()
            axs[0, 1].plot(smoothed_rewards)
            axs[0, 1].set_title('Smoothed Episode Rewards (10-ep window)')
            axs[0, 1].set_xlabel('Episode')
            axs[0, 1].set_ylabel('Reward')
            axs[0, 1].grid(True)
        
        # Plot episode lengths
        axs[1, 0].plot(self.metrics['episode_lengths'])
        axs[1, 0].set_title('Episode Lengths')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Steps')
        axs[1, 0].grid(True)
        
        # Plot losses
        if any(self.metrics['losses']):
            axs[1, 1].plot(self.metrics['losses'])
            axs[1, 1].set_title('Training Loss')
            axs[1, 1].set_xlabel('Episode')
            axs[1, 1].set_ylabel('Loss')
            axs[1, 1].grid(True)
        
        # Plot env step times if available
        plot_idx = 2
        if self.metrics['env_step_times']:
            axs[0, plot_idx].plot(self.metrics['env_step_times'])
            axs[0, plot_idx].set_title('Environment Step Times')
            axs[0, plot_idx].set_xlabel('Episode')
            axs[0, plot_idx].set_ylabel('Time (ms)')
            axs[0, plot_idx].grid(True)
            plot_idx += 1
        
        # Plot GPU memory usage if available
        if self.metrics['gpu_memory_usage']:
            axs[1, plot_idx-1].plot(self.metrics['gpu_memory_usage'])
            axs[1, plot_idx-1].set_title('GPU Memory Usage')
            axs[1, plot_idx-1].set_xlabel('Episode')
            axs[1, plot_idx-1].set_ylabel('Memory (MB)')
            axs[1, plot_idx-1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(results_dir, f"{self.experiment_name}_plots_{timestamp}.png")
        plt.savefig(plot_file)
        plt.close()
        self.logger.info(f"Training plots saved to {plot_file}")

def action_to_tuple(action, action_type):
    """Convert action dictionary to tuple based on action type."""
    if action is None or action_type is None:
        return (0, 0, 0)  # Default tuple for None actions
        
    if action_type == 0:  # Crane Movement
        # Handle torch tensor or numpy array
        if isinstance(action['crane_movement'], torch.Tensor):
            return tuple(action['crane_movement'].cpu().numpy())
        else:
            return tuple(action['crane_movement'])
    elif action_type == 1:  # Truck Parking
        if isinstance(action['truck_parking'], torch.Tensor):
            return tuple(action['truck_parking'].cpu().numpy())
        else:
            return tuple(action['truck_parking'])
    elif action_type == 2:  # Terminal Truck
        if isinstance(action['terminal_truck'], torch.Tensor):
            return tuple(action['terminal_truck'].cpu().numpy())
        else:
            return tuple(action['terminal_truck'])
    return (0, 0, 0)  # Fallback default tuple

class GPUCurriculumTrainer:
    """
    GPU-accelerated curriculum learning trainer for container terminal agents.
    Progressively increases environment complexity as agents improve.
    """
    
    def __init__(self, 
                 base_config_path=None, 
                 checkpoints_dir='checkpoints',
                 results_dir='results',
                 log_dir='logs',
                 experiment_name=None,
                 device='cuda',
                 quiet_console=True,
                 memory_tracking=True):
        """
        Initialize the GPU-accelerated curriculum trainer.
        
        Args:
            base_config_path: Path to base configuration file
            checkpoints_dir: Directory to save model checkpoints
            results_dir: Directory to save training results
            log_dir: Directory to save training logs
            experiment_name: Name for this experiment
            device: Computation device (CPU/GPU)
            quiet_console: Whether to suppress console output
            memory_tracking: Whether to track GPU memory usage
        """
        self.base_config_path = base_config_path
        self.checkpoints_dir = checkpoints_dir
        self.results_dir = results_dir
        self.device = device
        self.memory_tracking = memory_tracking
        
        # Create directories if they don't exist
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize GPU-optimized logger
        self.logger = GPUTrainingLogger(log_dir, experiment_name, quiet_console=quiet_console)
        
        # Log GPU device information
        if device == 'cuda' and torch.cuda.is_available():
            cuda_device = torch.cuda.current_device()
            self.logger.logger.info(f"Using GPU: {torch.cuda.get_device_name(cuda_device)}")
            
            # Log initial memory usage
            if self.memory_tracking:
                init_memory = torch.cuda.memory_allocated() / 1e6  # MB
                self.logger.logger.info(f"Initial GPU memory usage: {init_memory:.1f} MB")
        
        # Curriculum stages configuration (same as original)
        self.curriculum_stages = [
            # Stage 1: Basic terminal operations (small terminal)
            {
                'name': 'Basic Operations',
                'num_railtracks': 2,
                'num_railslots_per_track': 5,  # Even smaller for initial testing
                'num_storage_rows': 3,  # Reduced for initial testing
                'num_terminal_trucks': 1,
                'max_trucks_per_day': 8,  # Reduced for initial testing
                'max_trains_per_day': 2,  # Reduced for initial testing
                'target_reward': 20,
                'max_episodes': 100,
                'max_simulation_time': 86400 * 1  # 1 day
            },
            # Stage 2: Small terminal operations
            {
                'name': 'Small Terminal',
                'num_railtracks': 2,
                'num_railslots_per_track': 10, 
                'num_storage_rows': 3,
                'num_terminal_trucks': 1,
                'max_trucks_per_day': 5,
                'max_trains_per_day': 2,
                'target_reward': 50,
                'max_episodes': 100,
                'max_simulation_time': 86400 * 2  # 2 days
            },
            # Stage 3: Intermediate terminal (medium size)
            {
                'name': 'Intermediate Terminal',
                'num_railtracks': 3,
                'num_railslots_per_track': 15,
                'num_storage_rows': 4,
                'num_terminal_trucks': 2,
                'max_trucks_per_day': 10,
                'max_trains_per_day': 3,
                'target_reward': 100,
                'max_episodes': 150,
                'max_simulation_time': 86400 * 5  # 5 days
            },
            # Stage 4: Advanced terminal (full size)
            {
                'name': 'Advanced Terminal',
                'num_railtracks': 6,
                'num_railslots_per_track': 29,
                'num_storage_rows': 5,
                'num_terminal_trucks': 3,
                'max_trucks_per_day': 20,
                'max_trains_per_day': 5,
                'target_reward': 200,
                'max_episodes': 200,
                'max_simulation_time': 86400 * 10  # 10 days
            }
        ]
        
        # Initialize current stage
        self.current_stage = 0
    
    def create_environment(self, stage_config):
        """Create GPU-accelerated environment with configuration for current curriculum stage."""
        # Load base configuration
        terminal_config = TerminalConfig(self.base_config_path)
        
        # Create GPU-accelerated environment
        env = GPUTerminalEnvironment(
            terminal_config=terminal_config,
            max_simulation_time=stage_config['max_simulation_time'],
            num_cranes=2,  # Fixed for consistency
            num_terminal_trucks=stage_config['num_terminal_trucks'],
            device=self.device
        )
            
        # Enable performance logging
        env.log_performance = True
        
        # Modify the terminal layout with our curriculum parameters
        env.terminal = self._create_custom_terminal(
            stage_config['num_railtracks'],
            stage_config['num_railslots_per_track'],
            stage_config['num_storage_rows']
        )
        
        # Re-initialize components that depend on terminal layout
        env._setup_position_mappings()
        
        # Create appropriate storage yard
        env.storage_yard = self._create_gpu_storage_yard(env)
        
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
    
    def _create_gpu_storage_yard(self, env):
        """Create a GPU-accelerated storage yard with special areas."""
        # Define special areas for different container types
        num_bays = env.terminal.num_storage_slots_per_row
        
        # Calculate proportional positions for special areas
        special_areas = {
            'reefer': [],
            'dangerous': [],
            'trailer': [],
            'swap_body': []
        }
        
        # Add reefer areas in first and last column of each row
        for row in env.terminal.storage_row_names:
            special_areas['reefer'].append((row, 1, 1))
            special_areas['reefer'].append((row, num_bays, num_bays))
        
        # Add dangerous goods area in middle columns (using proportional values)
        dangerous_start = max(1, int(num_bays * 0.7))
        dangerous_end = min(num_bays, dangerous_start + 2)
        for row in env.terminal.storage_row_names:
            special_areas['dangerous'].append((row, dangerous_start, dangerous_end))
        
        # Add trailer and swap body areas with proportional values
        first_row = env.terminal.storage_row_names[0]
        trailer_start = max(1, int(num_bays * 0.2))
        trailer_end = max(trailer_start + 1, int(num_bays * 0.4))
        special_areas['trailer'].append((first_row, trailer_start, trailer_end))
        
        swap_start = max(trailer_end + 1, int(num_bays * 0.5))
        swap_end = max(swap_start + 1, int(num_bays * 0.7))
        special_areas['swap_body'].append((first_row, swap_start, swap_end))
        
        # Import directly instead of using a method that might not exist
        from simulation.deprecated_components.GPUStorageYard import GPUStorageYard
        
        return GPUStorageYard(
            num_rows=env.terminal.num_storage_rows,
            num_bays=env.terminal.num_storage_slots_per_row,
            max_tier_height=5,
            row_names=env.terminal.storage_row_names,
            special_areas=special_areas,
            device=self.device
        )
    
    def create_agent(self, env):
        """Create GPU-accelerated agent for the current environment."""
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
        
        # Create agent with the specified device
        agent = OptimizedTerminalAgent(
            state_dim, 
            action_dims,
            hidden_dims=[256, 256],
            head_dims=[128],
            device=self.device
        )
        return agent
    
    def _flatten_state(self, state):
        """Flatten state dictionary to vector for the agent, handling both tensor and numpy inputs."""
        # Check if input is None
        if state is None:
            return None
            
        # Helper function to process state components
        def process_component(component):
            # Handle tensor inputs - convert to numpy
            if isinstance(component, torch.Tensor):
                return component.detach().cpu().numpy().flatten()
            # Handle numpy arrays
            elif isinstance(component, np.ndarray):
                return component.flatten()
            # Handle other types if needed
            else:
                # Convert to numpy array if it's a list or other sequence
                try:
                    return np.array(component).flatten()
                except:
                    print(f"Warning: Could not convert component to numpy array: {type(component)}")
                    return np.array([])
        
        # Process all state components
        try:
            crane_positions = process_component(state['crane_positions'])
            crane_available_times = process_component(state['crane_available_times'])
            terminal_truck_available_times = process_component(state['terminal_truck_available_times'])
            current_time = process_component(state['current_time'])
            yard_state = process_component(state['yard_state'])
            parking_status = process_component(state['parking_status'])
            rail_status = process_component(state['rail_status'])
            queue_sizes = process_component(state['queue_sizes'])
            
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
        except Exception as e:
            print(f"Error flattening state: {e}")
            # Return a default state vector of zeros as fallback
            return np.zeros(500)  # Adjust size as needed
    
    def get_gpu_memory_usage(self):
        """Get current GPU memory usage in MB."""
        if self.device == 'cuda' and torch.cuda.is_available() and self.memory_tracking:
            return torch.cuda.memory_allocated() / 1e6  # MB
        return 0
    
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
        self.logger.logger.info(f"Creating GPU environment for stage 1 ({self.curriculum_stages[0]['name']})...")
        env, _ = self.create_environment(self.curriculum_stages[self.current_stage])
        agent = self.create_agent(env)
        
        # Load checkpoint if resuming
        if resume_checkpoint:
            self.logger.logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
            agent.load(resume_checkpoint)
        
        self.logger.logger.info(f"Starting GPU-accelerated curriculum training with {len(self.curriculum_stages)} stages on device: {self.device}")
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
                        episode_reward, episode_steps, mean_loss, action_counts, avg_step_time = self.train_episode(env, agent)
                        
                        # Track metrics
                        stage_rewards.append(episode_reward)
                        
                        # Get GPU memory usage
                        gpu_memory = self.get_gpu_memory_usage()
                        
                        # Log episode
                        self.logger.log_episode(
                            episode_count, 
                            episode_reward,
                            episode_steps,
                            mean_loss,
                            self.current_stage + 1,
                            action_counts,
                            avg_step_time,
                            gpu_memory
                        )
                        
                        # Update progress bar
                        avg_reward = np.mean(stage_rewards[-10:]) if len(stage_rewards) >= 10 else np.mean(stage_rewards)
                        progress_dict = {
                            'reward': f"{episode_reward:.2f}",
                            'avg10': f"{avg_reward:.2f}",
                            'steps': episode_steps
                        }
                        if avg_step_time is not None:
                            progress_dict['step_ms'] = f"{avg_step_time:.1f}"
                        if gpu_memory > 0:
                            progress_dict['GPU_MB'] = f"{gpu_memory:.1f}"
                        stage_pbar.set_postfix(progress_dict)
                        stage_pbar.update(1)
                        
                        # Save checkpoint periodically
                        if episode_count % 20 == 0:
                            self.save_checkpoint(agent, episode_count)
                        
                        # Check if we reached the target reward
                        if avg_reward >= stage_config['target_reward'] and stage_episodes >= 10:
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
                    self.logger.logger.info("Curriculum training complete!")
                    break
                    
                # Clear replay buffer before starting the next stage to avoid dimension mismatches
                self.logger.logger.info("Clearing replay buffer for new stage...")
                agent.replay_buffer.clear()
                
                # Clean up GPU memory between stages
                if self.device == 'cuda' and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    self.logger.logger.info(f"GPU memory cleared. Current usage: {self.get_gpu_memory_usage():.1f} MB")
            
            # Save final metrics and model
            training_time = time.time() - self.logger.start_time
            self.logger.log_training_complete(episode_count, training_time)
            self.logger.save_metrics(self.results_dir)
            self.save_checkpoint(agent, episode_count, "final")
            
            # Print performance stats from GPU environment
            if hasattr(env, 'print_performance_stats'):
                self.logger.logger.info("\nGPU Environment Performance Statistics:")
                env.print_performance_stats()
            
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
        
        finally:
            # Clean up GPU memory
            if self.device == 'cuda' and torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        return agent, self.logger.metrics
    
    def train_episode(self, env, agent):
        """Train for a single episode with GPU optimization."""
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        losses = []
        done = False
        truncated = False
        
        # Calculate maximum simulation time as 90% of env's maximum
        max_simulation_time = env.max_simulation_time * 0.9
        
        # Action tracking
        action_counts = {'crane': 0, 'truck_parking': 0, 'terminal_truck': 0}
        
        # Time tracking
        step_times = []
        consecutive_wait_count = 0
        last_stack_check_time = env.current_simulation_time
        
        # Main training loop 
        while not done and not truncated and env.current_simulation_time < max_simulation_time:
            # Check for pre-marshalling needs
            needs_premarshalling = agent._evaluate_need_for_premarshalling(env)
            
            # Select action based on current state
            action_masks = state['action_mask']
            
            # Get action from agent
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
                    needs_premarshalling = agent._evaluate_need_for_premarshalling(env)
                    
                    # If no pre-marshalling needed and no vehicles, advance time more aggressively
                    if not needs_premarshalling and env.truck_queue.size() == 0 and env.train_queue.size() == 0:
                        # More aggressive time jump - but still max 1 hour
                        advance_time = min(60 * 60, 5 * 60 * consecutive_wait_count)  # Increase time jumps but cap at 1 hour
                    else:
                        # Reset consecutive wait counter if we found something to do
                        consecutive_wait_count = 0
                        advance_time = 5 * 60  # 5 minutes
                else:
                    # Small time jumps between stack checks
                    advance_time = 5 * 60  # 5 minutes
                
                # Execute a wait action and force time advancement
                try:
                    start_time = time.time()
                    
                    # Use environment's create_wait_action to ensure valid indices
                    if hasattr(env, 'create_wait_action'):
                        wait_action = env.create_wait_action()
                    else:
                        # Fallback with safer indices
                        wait_action = {
                            'action_type': 0, 
                            'crane_movement': np.array([0, 0, 0]), 
                            'truck_parking': np.array([0, 0]),
                            'terminal_truck': np.array([0, 0, 0])
                        }
                    
                    next_state, reward, done, truncated, info = env.step(wait_action)
                    step_time = (time.time() - start_time) * 1000  # Convert to ms
                    step_times.append(step_time)
                except Exception as e:
                    self.logger.logger.warning(f"Error executing wait action: {e}")
                    # If the wait action fails, just advance time manually
                    env.current_simulation_time += advance_time
                    env.current_simulation_datetime += timedelta(seconds=advance_time)
                    env._process_vehicle_arrivals(advance_time)
                    env._process_vehicle_departures()
                    
                    # Get observation
                    next_state = env._get_observation()
                    done = env.current_simulation_time >= env.max_simulation_time
                    truncated = False
                    reward = 0
                
                episode_steps += 1
                
                # Update state
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
            
            # Process tensor action by converting to appropriate format
            if isinstance(action['crane_movement'], torch.Tensor):
                action['crane_movement'] = action['crane_movement'].cpu().numpy()
            
            # Validate action indices before executing
            try:
                # Execute action in environment and track time
                start_time = time.time()
                next_state, reward, done, truncated, info = env.step(action)
                step_time = (time.time() - start_time) * 1000  # Convert to ms
                step_times.append(step_time)
                
                episode_reward += reward
                episode_steps += 1
            except Exception as e:
                # Handle the error - try to select a new valid action instead
                self.logger.logger.warning(f"Invalid action: {e}. Using wait action instead.")
                
                # Use environment's create_wait_action method if available
                if hasattr(env, 'create_wait_action'):
                    wait_action = env.create_wait_action()
                else:
                    # Fallback with safer indices
                    wait_action = {
                        'action_type': 0, 
                        'crane_movement': np.array([0, 0, 0]), 
                        'truck_parking': np.array([0, 0]),
                        'terminal_truck': np.array([0, 0, 0])
                    }
                
                try:
                    # Track execution time
                    start_time = time.time()
                    next_state, reward, done, truncated, info = env.step(wait_action)
                    step_time = (time.time() - start_time) * 1000  # Convert to ms
                    step_times.append(step_time)
                    
                    episode_reward += reward  # Usually 0 for wait actions
                    episode_steps += 1
                except Exception as e2:
                    # If the wait action also fails, just advance time manually
                    self.logger.logger.warning(f"Wait action also failed: {e2}. Advancing time manually.")
                    env.current_simulation_time += 60  # Advance 1 minute
                    env.current_simulation_datetime += timedelta(seconds=60)
                    env._process_vehicle_arrivals(60)
                    env._process_vehicle_departures()
                    
                    # Get observation
                    next_state = env._get_observation()
                    done = env.current_simulation_time >= env.max_simulation_time
                    truncated = False
                    reward = 0
            
            # Store experience and update networks
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
            
            # Update networks
            if len(agent.replay_buffer) > agent.batch_size:
                loss = agent.update()
                if loss is not None:
                    losses.append(loss)
            
            # Update state
            state = next_state
        
        # Calculate mean loss
        mean_loss = np.mean(losses) if losses else None
        
        # Calculate average step time
        avg_step_time = np.mean(step_times) if step_times else None
        
        # If using env step times, also check its performance stats
        if hasattr(env, 'step_times') and env.step_times:
            env_avg_step_time = np.mean(env.step_times) * 1000  # Convert to ms
            if avg_step_time is None or len(env.step_times) > len(step_times):
                avg_step_time = env_avg_step_time
        
        return episode_reward, episode_steps, mean_loss, action_counts, avg_step_time
    
    def save_checkpoint(self, agent, episode, suffix=None):
        """Save agent checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.logger.experiment_name}_ep{episode}"
        if suffix:
            filename += f"_{suffix}"
        filename += f"_{timestamp}.pt"
        filepath = os.path.join(self.checkpoints_dir, filename)
        agent.save(filepath)
        self.logger.logger.info(f"Checkpoint saved to {filepath}")
        
        # Log memory usage after saving
        if self.device == 'cuda' and torch.cuda.is_available() and self.memory_tracking:
            memory_usage = torch.cuda.memory_allocated() / 1e6  # MB
            self.logger.logger.info(f"GPU memory after saving: {memory_usage:.1f} MB")
    
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
        
        self.logger.logger.info(f"\nEvaluating agent for {num_episodes} episodes on {self.device}...")
        eval_pbar = tqdm(total=num_episodes, desc="Evaluating")
        
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            truncated = False
            
            while not done and not truncated and episode_steps < 2000:  # Limit eval episodes
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
                    
                    # Get observation
                    next_state = env._get_observation()
                    reward = 0
                else:
                    # Process tensor action by converting to appropriate format
                    if isinstance(action['crane_movement'], torch.Tensor):
                        action['crane_movement'] = action['crane_movement'].cpu().numpy()
                    
                    # Take action in environment
                    next_state, reward, done, truncated, _ = env.step(action)
                    episode_reward += reward
                    
                    # Update action count
                    if action_type == 0:
                        action_counts['crane'] += 1
                    elif action_type == 1:
                        action_counts['truck_parking'] += 1
                    elif action_type == 2:
                        action_counts['terminal_truck'] += 1
                
                episode_steps += 1
                
                # Render if requested
                if render:
                    env.render()
                    time.sleep(0.1)
                
                # Update state
                state = next_state
            
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
            'action_counts': action_counts
        }
        
        # Log evaluation results
        self.logger.logger.info("\nEvaluation Results:")
        self.logger.logger.info(f"  Mean Reward: {evaluation_metrics['mean_reward']:.2f} ± {evaluation_metrics['std_reward']:.2f}")
        self.logger.logger.info(f"  Mean Steps: {evaluation_metrics['mean_steps']:.2f}")
        self.logger.logger.info(f"  Mean Simulation Days: {evaluation_metrics['mean_days']:.2f}")
        
        # Log action distribution
        action_str = " | ".join([f"{key}: {count}" for key, count in action_counts.items()])
        self.logger.logger.info(f"  Action counts: {action_str}")
        
        # Print environment performance stats if available
        if hasattr(env, 'print_performance_stats'):
            env.print_performance_stats()
        
        return evaluation_metrics
    
