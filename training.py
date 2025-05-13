# training.py

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
# from simulation.terminal_env import TerminalEnvironment
from simulation.EnhancedTerminalEnvironment import EnhancedTerminalEnvironment as TerminalEnvironment
from simulation.config import TerminalConfig

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
            'times': []
        }
        
        self.start_time = time.time()
        
        # Log start message (will be in file but may be suppressed on console)
        self.logger.info(f"Started training experiment: {experiment_name}")
        self.logger.info(f"Log file: {self.log_path}")
        
        # Important messages should use warning level to always show on console
        if quiet_console:
            self.logger.warning(f"Running in quiet mode - only warnings and errors will show on console")
            self.logger.warning(f"All logs are still saved to: {self.log_path}")
    
    def log_episode(self, episode, reward, steps, loss, stage, action_counts=None):
        """
        Log episode metrics.
        
        Args:
            episode: Episode number
            reward: Episode total reward
            steps: Number of steps in episode
            loss: Mean loss for episode
            stage: Curriculum stage
            action_counts: Dictionary of action type counts
        """
        # Record metrics
        self.metrics['episode_rewards'].append(reward)
        self.metrics['episode_lengths'].append(steps)
        self.metrics['losses'].append(loss if loss is not None else 0)
        self.metrics['curriculum_stages'].append(stage)
        self.metrics['times'].append(time.time() - self.start_time)
        
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
        self.logger.info(f"Episode {episode} | Stage {stage} | Reward: {reward:.2f} | Avg(10): {avg_reward:.2f} | Steps: {steps} | Loss: {loss_str}")
        
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
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
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
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(results_dir, f"{self.experiment_name}_plots_{timestamp}.png")
        plt.savefig(plot_file)
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
                 device='cuda' if torch.cuda.is_available() else 'cpu',
                 quiet_console=True):
        """
        Initialize the curriculum trainer.
        
        Args:
            base_config_path: Path to base configuration file
            checkpoints_dir: Directory to save model checkpoints
            results_dir: Directory to save training results
            log_dir: Directory to save training logs
            experiment_name: Name for this experiment
            device: Computation device (CPU/GPU)
        """
        self.base_config_path = base_config_path
        self.checkpoints_dir = checkpoints_dir
        self.results_dir = results_dir
        self.device = device
        
        # Create directories if they don't exist
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize logger
        self.logger = TrainingLogger(log_dir, experiment_name, quiet_console=quiet_console)
        
        # Curriculum stages configuration
        self.curriculum_stages = [
            # Stage 1: Basic terminal operations (small terminal)
            {
                'name': 'Basic Operations',
                'num_railtracks': 2,
                'num_railslots_per_track': 10,
                'num_storage_rows': 3,
                'num_terminal_trucks': 1,
                'max_trucks_per_day': 5,
                'max_trains_per_day': 2,
                'target_reward': 50,
                'max_episodes': 100,
                'max_simulation_time': 86400 * 365  # days
            }
            # },
            # # Stage 2: Intermediate terminal (medium size)
            # {
            #     'name': 'Intermediate Terminal',
            #     'num_railtracks': 3,
            #     'num_railslots_per_track': 15,
            #     'num_storage_rows': 4,
            #     'num_terminal_trucks': 2,
            #     'max_trucks_per_day': 10,
            #     'max_trains_per_day': 3,
            #     'target_reward': 100,
            #     'max_episodes': 150,
            #     'max_simulation_time': 86400 * 5  # 5 days
            # },
            # # Stage 3: Advanced terminal (full size)
            # {
            #     'name': 'Advanced Terminal',
            #     'num_railtracks': 6,
            #     'num_railslots_per_track': 29,
            #     'num_storage_rows': 5,
            #     'num_terminal_trucks': 3,
            #     'max_trucks_per_day': 20,
            #     'max_trains_per_day': 5,
            #     'target_reward': 200,
            #     'max_episodes': 200,
            #     'max_simulation_time': 86400 * 10  # 10 days
            # }
        ]
        
        # Initialize current stage
        self.current_stage = 0
    
    def create_environment(self, stage_config):
        """Create environment with configuration for current curriculum stage."""
        # Load base configuration
        terminal_config = TerminalConfig(self.base_config_path)
        
        # Create environment with current stage parameters
        # env = TerminalEnvironment(
        #     terminal_config=terminal_config,
        #     max_simulation_time=stage_config['max_simulation_time'],
        #     num_cranes=2,  # Fixed for consistency
        #     num_terminal_trucks=stage_config['num_terminal_trucks']
        # )
        env = TerminalEnvironment(
            terminal_config=terminal_config,
            max_simulation_time=stage_config['max_simulation_time'],
            num_cranes=2,  # Fixed for consistency
            num_terminal_trucks=stage_config['num_terminal_trucks'],
            device=self.device,
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
        """Create a storage yard matching the terminal layout with special areas."""
        from simulation.terminal_components.Storage_Yard import StorageYard
        
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
        
        return StorageYard(
            num_rows=env.terminal.num_storage_rows,
            num_bays=env.terminal.num_storage_slots_per_row,
            max_tier_height=4,
            row_names=env.terminal.storage_row_names,
            special_areas=special_areas
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
        
        # Create agent
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
                return np.array(component).flatten()
        
        # Process all state components
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
        
        self.logger.logger.info(f"Starting curriculum training with {len(self.curriculum_stages)} stages")
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
                        episode_reward, episode_steps, mean_loss, action_counts = self.train_episode(env, agent)
                        
                        # Track metrics
                        stage_rewards.append(episode_reward)
                        
                        # Log episode
                        self.logger.log_episode(
                            episode_count, 
                            episode_reward,
                            episode_steps,
                            mean_loss,
                            self.current_stage + 1,
                            action_counts
                        )
                        
                        # Update progress bar
                        avg_reward = np.mean(stage_rewards[-10:]) if len(stage_rewards) >= 10 else np.mean(stage_rewards)
                        stage_pbar.set_postfix({
                            'reward': f"{episode_reward:.2f}",
                            'avg10': f"{avg_reward:.2f}",
                            'steps': episode_steps
                        })
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
    
    def train_episode(self, env: TerminalEnvironment, agent: OptimizedTerminalAgent):
        """Train for a single episode with focus on pre-marshalling during downtime."""
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        losses = []
        done = False
        
        # Calculate maximum simulation time as 90% of env's maximum
        max_simulation_time = env.max_simulation_time * 0.9
        
        # Action tracking
        action_counts = {'crane': 0, 'truck_parking': 0, 'terminal_truck': 0}
        
        # Time advancement tracking
        consecutive_wait_count = 0
        last_stack_check_time = env.current_simulation_time
        
        # Main training loop 
        while not done and env.current_simulation_time < max_simulation_time:
            # Check for pre-marshalling needs
            needs_premarshalling = hasattr(env, 'evaluate_need_for_premarshalling') and env.evaluate_need_for_premarshalling()
            
            # Select action based on current state
            action_masks = state['action_mask']
            
            # If pre-marshalling is needed, force agent to prioritize yard operations
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
                wait_action = {
                    'action_type': 0, 
                    'crane_movement': np.array([0, 0, 0]), 
                    'truck_parking': np.array([0, 0]),
                    'terminal_truck': np.array([0, 0, 0])  # Add this line
                }
                next_state, reward, done, truncated, info = env.step(wait_action)
                
                # Force time advancement
                env.current_simulation_time += advance_time
                env.current_simulation_datetime += timedelta(seconds=advance_time)
                
                # Process vehicles
                env._process_vehicle_arrivals(advance_time)
                env._process_vehicle_departures()
                
                # # Force vehicle arrivals if we've been waiting too long
                # if consecutive_wait_count > 20:
                #     self._force_vehicle_arrivals(env)
                
                episode_steps += 1
                
                # Update state
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
            
            # Execute action in environment
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
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
        
        return episode_reward, episode_steps, mean_loss, action_counts
    
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
        
        return evaluation_metrics

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train terminal agents with curriculum learning')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=None, help='Total number of episodes')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Checkpoints directory')
    parser.add_argument('--results', type=str, default='results', help='Results directory')
    parser.add_argument('--logs', type=str, default='logs', help='Logs directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                      help='Computation device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    parser.add_argument('--render', action='store_true', help='Render during evaluation')
    parser.add_argument('--resume', type=str, default=None, help='Resume training from checkpoint')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.name = f"terminal_training_{timestamp}"
    
    # Create curriculum trainer
    trainer = CurriculumTrainer(
        base_config_path=args.config,
        checkpoints_dir=args.checkpoints,
        results_dir=args.results,
        log_dir=args.logs,
        experiment_name=args.name,
        device=args.device
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