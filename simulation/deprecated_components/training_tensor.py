# enhanced_training.py with optimized environment

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
from simulation.deprecated_components.dual_head_q_tensored import OptimizedTerminalAgent

# Import the new optimized environment (adjust path as needed)
from simulation.deprecated_components.optimized_environment import OptimizedTerminalEnvironment, TerminalTrainingWrapper
from simulation.TerminalConfig import TerminalConfig

class TrainingLogger:
    """Logger for training with file output and console display."""
    
    def __init__(self, log_dir='logs', experiment_name=None, quiet_console=True):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (auto-generated if None)
            quiet_console: If True, only show warnings, errors and tqdm on console
        """
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate experiment name if not provided
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"terminal_training_{timestamp}"
        
        self.experiment_name = experiment_name
        self.log_path = os.path.join(log_dir, f"{experiment_name}.log")
        self.quiet_console = quiet_console
        
        # Configure logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.INFO)  # Main logger level stays at INFO
        
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
        
        # Initialize metrics tracking
        self.metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'curriculum_stages': [],
            'action_types': {'crane': 0, 'truck_parking': 0, 'terminal_truck': 0},
            'times': [],
            'env_step_times': []  # Added for environment performance tracking
        }
        
        self.start_time = time.time()
        
        # Log start message (will be in file but may be suppressed on console)
        self.logger.info(f"Started training experiment: {experiment_name}")
        self.logger.info(f"Log file: {self.log_path}")
        
        # Important messages should use warning level to always show on console
        if quiet_console:
            self.logger.warning(f"Running in quiet mode - only warnings and errors will show on console")
            self.logger.warning(f"All logs are still saved to: {self.log_path}")
    
    def log_episode(self, episode, reward, steps, loss, stage, action_counts=None, env_step_time=None):
        """
        Log episode metrics.
        
        Args:
            episode: Episode number
            reward: Episode total reward
            steps: Number of steps in episode
            loss: Mean loss for episode
            stage: Curriculum stage
            action_counts: Dictionary of action type counts
            env_step_time: Average environment step time (in ms)
        """
        # Record metrics
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_lengths'].append(steps)
        self.metrics['losses'].append(loss if loss is not None else 0)
        self.metrics['curriculum_stages'].append(stage)
        self.metrics['times'].append(time.time() - self.start_time)
        if env_step_time is not None:
            self.metrics['env_step_times'].append(env_step_time)
        
        # Update action counts if provided
        if action_counts:
            for action_type, count in action_counts.items():
                if action_type in self.metrics['action_types']:
                    self.metrics['action_types'][action_type] += count
        
        # Calculate averages for logging
        avg_reward = np.mean(self.metrics['episode_rewards'][-10:]) if len(self.metrics['episode_rewards']) >= 10 else np.mean(self.metrics['episode_rewards'])
        avg_steps = np.mean(self.metrics['episode_lengths'][-10:]) if len(self.metrics['episode_lengths']) >= 10 else np.mean(self.metrics['episode_lengths'])
        
        # Format loss string properly outside the f-string
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        
        # Log episode summary
        log_str = f"Episode {episode} | Stage {stage} | Reward: {reward:.2f} | Avg(10): {avg_reward:.2f} | Steps: {steps} | Loss: {loss_str}"
        if env_step_time is not None:
            log_str += f" | Env Step: {env_step_time:.2f}ms"
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
        """Log completion of training."""
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
            
        # Log environment performance statistics if available
        if self.metrics['env_step_times']:
            avg_step_time = np.mean(self.metrics['env_step_times'])
            min_step_time = np.min(self.metrics['env_step_times'])
            max_step_time = np.max(self.metrics['env_step_times'])
            self.logger.info(f"Environment performance: Avg step time: {avg_step_time:.2f}ms, Min: {min_step_time:.2f}ms, Max: {max_step_time:.2f}ms")
    
    def save_metrics(self, results_dir='results'):
        """
        Save training metrics to files.
        
        Args:
            results_dir: Directory to save results
        """
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
        
        # Add env step times if available
        if self.metrics['env_step_times']:
            # Pad with None for episodes without step time data
            padded_step_times = self.metrics['env_step_times'] + [None] * (len(self.metrics['episode_rewards']) - len(self.metrics['env_step_times']))
            results_df['env_step_time'] = padded_step_times[:len(results_df)]
            
        results_df.to_csv(results_file, index=False)
        self.logger.info(f"Training metrics saved to {results_file}")
        
        # Save action counts
        action_counts_file = os.path.join(results_dir, f"{self.experiment_name}_actions_{timestamp}.json")
        with open(action_counts_file, 'w') as f:
            json.dump(self.metrics['action_types'], f, indent=2)
        
        # Create and save plots
        self._save_training_plots(results_dir, timestamp)
    
    def _save_training_plots(self, results_dir, timestamp):
        """Create and save plots of training metrics."""
        # Determine number of plots (add env step times if available)
        n_plots = 3 if self.metrics['env_step_times'] else 2
        
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
        if self.metrics['env_step_times']:
            axs[0, 2].plot(self.metrics['env_step_times'])
            axs[0, 2].set_title('Environment Step Times')
            axs[0, 2].set_xlabel('Episode')
            axs[0, 2].set_ylabel('Time (ms)')
            axs[0, 2].grid(True)
        
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

class CurriculumTrainer:
    """
    Curriculum learning trainer for container terminal agents.
    Progressively increases environment complexity as agents improve.
    """
    
    def __init__(self, 
                 base_config_path=None, 
                 checkpoints_dir='checkpoints',
                 results_dir='results',
                 log_dir='logs',
                 experiment_name=None,
                 device='cuda',  # Default to CPU for optimized environment
                 quiet_console=True,
                 use_optimized_env=True):  # Added flag to toggle optimized environment
        """
        Initialize the curriculum trainer.
        
        Args:
            base_config_path: Path to base configuration file
            checkpoints_dir: Directory to save model checkpoints
            results_dir: Directory to save training results
            log_dir: Directory to save training logs
            experiment_name: Name for this experiment
            device: Computation device (CPU/GPU)
            quiet_console: Whether to suppress console output
            use_optimized_env: Whether to use the optimized environment
        """
        self.base_config_path = base_config_path
        self.checkpoints_dir = checkpoints_dir
        self.results_dir = results_dir
        self.device = device
        self.use_optimized_env = use_optimized_env
        
        # Create directories if they don't exist
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = TrainingLogger(log_dir, experiment_name, quiet_console=quiet_console)
        
        # Log environment type
        env_type = "Optimized CPU" if use_optimized_env else "Original Tensor-based"
        self.logger.logger.info(f"Using {env_type} environment implementation")
        
        # Curriculum stages configuration
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
        """Create environment with configuration for current curriculum stage."""
        # Load base configuration
        terminal_config = TerminalConfig(self.base_config_path)
        
        # Create environment using the wrapper with current stage parameters
        if self.use_optimized_env:
            # Use the optimized environment implementation
            env = OptimizedTerminalEnvironment(
                terminal_config=terminal_config,
                max_simulation_time=stage_config['max_simulation_time'],
                num_cranes=2,  # Fixed for consistency
                num_terminal_trucks=stage_config['num_terminal_trucks']
            )
            
            # Enable performance logging
            env.log_performance = True
        else:
            # Use the original environment implementation
            # Import here to avoid import error if not available
            from simulation.deprecated_components.EnhancedTerminalEnvironment import EnhancedTerminalEnvironment
            
            env = EnhancedTerminalEnvironment(
                terminal_config=terminal_config,
                max_simulation_time=stage_config['max_simulation_time'],
                num_cranes=2,  # Fixed for consistency
                num_terminal_trucks=stage_config['num_terminal_trucks'],
                device=self.device,  # Pass device parameter for tensor-based operations
            )
        
        # Modify the terminal layout with our curriculum parameters
        env.terminal = self._create_custom_terminal(
            stage_config['num_railtracks'],
            stage_config['num_railslots_per_track'],
            stage_config['num_storage_rows']
        )
        
        # Re-initialize components that depend on terminal layout
        env._setup_position_mappings()
        
        # Create appropriate storage yard based on environment type
        if self.use_optimized_env:
            env.storage_yard = self._create_custom_storage_yard_optimized(env)
        else:
            env.storage_yard = self._create_custom_storage_yard_tensor(env)
        
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
        
    def _create_custom_storage_yard_optimized(self, env):
        """Create an optimized storage yard with special areas."""
        # This method creates a CPU-optimized storage yard for the optimized environment
        
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
        
        # Use FastStorageYard from the optimized environment
        return env._create_optimized_storage_yard()
    
    def _create_custom_storage_yard_tensor(self, env):
        """Create a tensor-based storage yard with special areas."""
        # This method creates a PyTorch-based storage yard for the original environment
        from simulation.deprecated_components.TensorStorageYard import TensorStorageYard
        
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
        
        # Use TensorStorageYard for better performance with tensor operations
        return TensorStorageYard(
            num_rows=env.terminal.num_storage_rows,
            num_bays=env.terminal.num_storage_slots_per_row,
            max_tier_height=4,
            row_names=env.terminal.storage_row_names,
            special_areas=special_areas,
            device=self.device  # Pass the device parameter
        )
    
    def create_agent(self, env):
        """Create agent for the current environment."""
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
        # Check if input is None (used in terminal states)
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
    
    def train(self, total_episodes=None, resume_checkpoint=None):
        """
        Train agents through curriculum stages.
        
        Args:
            total_episodes: Total episodes to train (overrides per-stage limits)
            resume_checkpoint: Path to checkpoint for resuming training
            
        Returns:
            Trained agent and training metrics
        """
        # Initialize tracking variables
        episode_count = 0
        
        # Initialize agent with first stage environment
        self.logger.logger.info(f"Creating environment for stage 1 ({self.curriculum_stages[0]['name']})...")
        env, _ = self.create_environment(self.curriculum_stages[self.current_stage])
        agent = self.create_agent(env)
        
        # Load checkpoint if resuming
        if resume_checkpoint:
            self.logger.logger.info(f"Resuming training from checkpoint: {resume_checkpoint}")
            agent.load(resume_checkpoint)
        
        self.logger.logger.info(f"Starting curriculum training with {len(self.curriculum_stages)} stages on device: {self.device}")
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
                        
                        # Log episode
                        self.logger.log_episode(
                            episode_count, 
                            episode_reward,
                            episode_steps,
                            mean_loss,
                            self.current_stage + 1,
                            action_counts,
                            avg_step_time
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
            
            # Save final metrics and model
            training_time = time.time() - self.logger.start_time
            self.logger.log_training_complete(episode_count, training_time)
            self.logger.save_metrics(self.results_dir)
            self.save_checkpoint(agent, episode_count, "final")
            
            # Print performance stats from optimized environment
            if self.use_optimized_env and hasattr(env, 'print_performance_stats'):
                self.logger.logger.info("\nEnvironment Performance Statistics:")
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
        
        return agent, self.logger.metrics
    
    def train_episode(self, env, agent: OptimizedTerminalAgent):
        """Train for a single episode with focus on pre-marshalling during downtime."""
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
            # Use different methods based on environment type
            if self.use_optimized_env:
                # For optimized environment
                needs_premarshalling = False  # Default value
                if hasattr(agent, '_evaluate_need_for_premarshalling'):
                    needs_premarshalling = agent._evaluate_need_for_premarshalling(env)
            else:
                # For original environment
                needs_premarshalling = hasattr(env, 'evaluate_need_for_premarshalling') and env.evaluate_need_for_premarshalling()
            
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
                    if self.use_optimized_env:
                        if hasattr(agent, '_evaluate_need_for_premarshalling'):
                            needs_premarshalling = agent._evaluate_need_for_premarshalling(env)
                    else:
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
                    
                    # Get observation using the appropriate method
                    if self.use_optimized_env:
                        next_state = env._get_observation()
                    else:
                        next_state = env._get_observation_tensor()
                        
                    done = env.current_simulation_time >= env.max_simulation_time
                    truncated = False
                
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
            if action_type == 0:  # Crane Movement
                # Convert tensors to numpy arrays to ensure compatibility
                if isinstance(action['crane_movement'], torch.Tensor):
                    crane_idx, source_idx, dest_idx = action['crane_movement'].detach().cpu().numpy()
                else:
                    crane_idx, source_idx, dest_idx = action['crane_movement']
                
                # Ensure indices are within valid range
                n_cranes = len(env.cranes)
                n_positions = len(env.position_to_idx)
                
                if crane_idx >= n_cranes or source_idx >= n_positions or dest_idx >= n_positions:
                    # Invalid indices, create a wait action instead
                    action = {
                        'action_type': 0, 
                        'crane_movement': np.array([0, 0, 0]),
                        'truck_parking': np.array([0, 0]),
                        'terminal_truck': np.array([0, 0, 0])
                    }
            
            # Validate action indices before executing
            try:
                # Execute action in environment and track time
                start_time = time.time()
                next_state, reward, done, truncated, info = env.step(action)
                step_time = (time.time() - start_time) * 1000  # Convert to ms
                step_times.append(step_time)
                
                episode_reward += reward
                episode_steps += 1
            except (KeyError, ValueError, IndexError) as e:
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
                    
                    # Get observation using the appropriate method
                    if self.use_optimized_env:
                        next_state = env._get_observation()
                    else:
                        next_state = env._get_observation_tensor()
                        
                    done = env.current_simulation_time >= env.max_simulation_time
                    truncated = False
            
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
        
        # If using the optimized environment, also check its performance stats
        if self.use_optimized_env and hasattr(env, 'step_times') and env.step_times:
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
    
    def evaluate(self, agent, num_episodes=5, render=False):
        """
        Evaluate trained agent on the full environment.
        
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
        
        self.logger.logger.info(f"\nEvaluating agent for {num_episodes} episodes...")
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
                    
                    # Get observation using the appropriate method
                    if self.use_optimized_env:
                        next_state = env._get_observation()
                    else:
                        next_state = env._get_observation_tensor()
                else:
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
        if self.use_optimized_env and hasattr(env, 'print_performance_stats'):
            env.print_performance_stats()
        
        return evaluation_metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train terminal agents with optimized environment')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=None, help='Total number of episodes')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Checkpoints directory')
    parser.add_argument('--results', type=str, default='results', help='Results directory')
    parser.add_argument('--logs', type=str, default='logs', help='Logs directory')
    parser.add_argument('--device', type=str, default='cpu', help='Computation device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    parser.add_argument('--render', action='store_true', help='Render during evaluation')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--use-original', action='store_true', help='Use original tensor-based environment')
    
    return parser.parse_args()

def main():
    """Main function for training terminal agents."""
    args = parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Generate experiment name if not provided
    if args.name is None:
        env_type = "original" if args.use_original else "optimized"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"{env_type}_terminal_training_{timestamp}"
    
    # Create curriculum trainer
    trainer = CurriculumTrainer(
        base_config_path=args.config,
        checkpoints_dir=args.checkpoints,
        results_dir=args.results,
        log_dir=args.logs,
        experiment_name=args.name,
        device=args.device,
        use_optimized_env=not args.use_original
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