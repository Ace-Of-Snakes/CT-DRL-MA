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

# Add project root to path to ensure imports work correctly
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import our modules
from simulation.agents.dual_head_q import OptimizedTerminalAgent
from simulation.terminal_env import TerminalEnvironment
from simulation.config import TerminalConfig

def action_to_tuple(action, action_type):
    """Convert action dictionary to tuple based on action type."""
    if action_type == 0 or action_type == 1:  # Pre-Marshalling or Crane Movement
        return tuple(action['crane_movement'])
    elif action_type == 2:  # Truck Parking
        return tuple(action['truck_parking'])
    return None
def evaluate_need_for_premarshalling(env):
    """Determine if pre-marshalling is needed based on yard state."""
    # Get all stacks in yard
    yard = env.storage_yard
    
    # Count problematic stacks (higher priority below lower)
    problem_stacks = 0
    total_stacks = 0
    
    for row in yard.row_names:
        for bay in range(1, yard.num_bays + 1):
            position = f"{row}{bay}"
            containers = yard.get_containers_at_position(position)
            
            if len(containers) > 1:
                total_stacks += 1
                # Check if priorities are ordered correctly (higher on top)
                priorities = [containers[tier].priority for tier in sorted(containers.keys())]
                
                # Check if priorities are in descending order
                if not all(priorities[i] >= priorities[i+1] for i in range(len(priorities)-1)):
                    problem_stacks += 1
    
    # Only allow pre-marshalling if enough problematic stacks exist
    return problem_stacks > 0 and problem_stacks / max(1, total_stacks) > 0.2

def calculate_yard_state_score(env):
    """Calculate a score for the current yard state (higher is better)."""
    yard = env.storage_yard
    score = 0
    
    # Component 1: Priority-based stacking score
    for row in yard.row_names:
        for bay in range(1, yard.num_bays + 1):
            position = f"{row}{bay}"
            containers = yard.get_containers_at_position(position)
            
            if len(containers) > 1:
                # Check if priorities are ordered correctly (higher on top)
                tiers = sorted(containers.keys())
                priorities = [containers[tier].priority for tier in tiers]
                
                # Score each container's position
                for i in range(len(priorities) - 1):
                    if priorities[i] <= priorities[i+1]:  # Lower priority below higher
                        score += 1
                    else:
                        score -= 2  # Penalty for incorrect order
    
    # Component 2: Distribution score - avoid having all containers in a few stacks
    occupied_positions = 0
    total_containers = 0
    
    for row in yard.row_names:
        for bay in range(1, yard.num_bays + 1):
            position = f"{row}{bay}"
            containers = yard.get_containers_at_position(position)
            if containers:
                occupied_positions += 1
                total_containers += len(containers)
    
    # Higher score for better distribution
    if occupied_positions > 0:
        avg_height = total_containers / occupied_positions
        if avg_height <= 2:  # Ideal height
            score += 50
        else:
            score -= (avg_height - 2) * 10  # Penalty for tall stacks
    
    return score

def calculate_premarshalling_reward(env, source_position, dest_position, container):
    """Calculate specialized reward for pre-marshalling actions."""
    # Get the yard state score before the move
    before_score = calculate_yard_state_score(env)
    
    # Temporarily make the move
    source_containers = env.storage_yard.get_containers_at_position(source_position)
    source_tier = max(source_containers.keys())
    container = source_containers[source_tier]
    
    # Remove from source
    env.storage_yard.remove_container(source_position, source_tier)
    
    # Add to destination
    dest_containers = env.storage_yard.get_containers_at_position(dest_position)
    dest_tier = max(dest_containers.keys()) + 1 if dest_containers else 1
    env.storage_yard.add_container(dest_position, container, dest_tier)
    
    # Calculate score after move
    after_score = calculate_yard_state_score(env)
    
    # Undo the move
    env.storage_yard.remove_container(dest_position, dest_tier)
    env.storage_yard.add_container(source_position, container, source_tier)
    
    # Calculate reward based on yard state improvement
    improvement = after_score - before_score
    
    if improvement > 0:
        # Positive reward for yard improvement
        reward = 2.0 + improvement * 0.5
    else:
        # Penalty for non-improving moves
        reward = -4.0 + improvement * 0.5
    
    return reward

class CurriculumTrainer:
    """
    Curriculum learning trainer for container terminal agents.
    Progressively increases environment complexity as agents improve.
    """
    
    def __init__(self, 
                 base_config_path=None, 
                 checkpoints_dir='checkpoints',
                 results_dir='results',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the curriculum trainer.
        
        Args:
            base_config_path: Path to base configuration file
            checkpoints_dir: Directory to save model checkpoints
            results_dir: Directory to save training results
            device: Computation device (CPU/GPU)
        """
        self.base_config_path = base_config_path
        self.checkpoints_dir = checkpoints_dir
        self.results_dir = results_dir
        self.device = device
        
        # Create directories if they don't exist
        os.makedirs(checkpoints_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize metrics tracking
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'curriculum_stages': []
        }
        
        # Curriculum stages configuration - simpler with longer episodes
        self.curriculum_stages = [
            # Stage 1: Minimal terminal (tiny playground)
            {
                'num_railtracks': 2,
                'num_railslots_per_track': 10,  # Just 2 slots
                'num_storage_rows': 5,
                'max_trucks_per_day': 50,       # Only one truck
                'max_trains_per_day': 10,
                'initial_containers': 100,       # Start with few containers
                'target_reward': 100,
                'max_episodes': 100,
                'max_simulation_time': 86400 * 365  #seconds in a day * number of days
            }
            # # Stage 2: Simple terminal (slightly larger)
            # {
            #     'num_railtracks': 1,
            #     'num_railslots_per_track': 10,
            #     'num_storage_rows': 2,
            #     'max_trucks_per_day': 3,
            #     'max_trains_per_day': 1,
            #     'target_reward': -20,  
            #     'max_episodes': 200,
            #     'max_simulation_time': 86400 * 5  # 5 days
            # },
            # # Stage 3: Medium terminal 
            # {
            #     'num_railtracks': 2,
            #     'num_railslots_per_track': 15,
            #     'num_storage_rows': 3,
            #     'max_trucks_per_day': 5,
            #     'max_trains_per_day': 2,
            #     'target_reward': -10,
            #     'max_episodes': 200,
            #     'max_simulation_time': 86400 * 10  # 10 days
            # },
            # # Stage 4: Full terminal
            # {
            #     'num_railtracks': 6,
            #     'num_railslots_per_track': 29,
            #     'num_storage_rows': 5,
            #     'max_trucks_per_day': 10,
            #     'max_trains_per_day': 4,
            #     'target_reward': -5,
            #     'max_episodes': 300,
            #     'max_simulation_time': 86400 * 30  # 30 days
            # }
        ]
        
        # Initialize current stage
        self.current_stage = 0
    
    def create_environment(self, stage_config):
        """
        Create environment with configuration for current curriculum stage.
        
        Args:
            stage_config: Configuration for the current stage
            
        Returns:
            Configured TerminalEnvironment
        """
        # Load base configuration
        terminal_config = TerminalConfig(self.base_config_path)
        
        # Create environment with current stage parameters
        # The TerminalEnvironment creates its own ContainerTerminal, so we need to monkey patch
        # its _create_terminal method to use our desired parameters
        env = TerminalEnvironment(
            terminal_config=terminal_config,
            max_simulation_time=stage_config['max_simulation_time']
        )
        
        # Modify the terminal layout with our curriculum parameters
        env.terminal = self._create_custom_terminal(
            stage_config['num_railtracks'],
            stage_config['num_railslots_per_track'],
            stage_config['num_storage_rows']
        )
        
        # We need to re-initialize components that depend on terminal layout
        env._setup_position_mappings()
        env.storage_yard = self._create_custom_storage_yard(env)
        
        # Set vehicle limits
        env.set_vehicle_limits(
            max_trucks=stage_config['max_trucks_per_day'],
            max_trains=stage_config['max_trains_per_day']
        )
        
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
        """Create a storage yard matching the terminal layout."""
        from simulation.terminal_components.Storage_Yard import StorageYard
        
        # Define special areas for different container types
        special_areas = {
            'reefer': [],
            'dangerous': []
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
        
        return StorageYard(
            num_rows=env.terminal.num_storage_rows,
            num_bays=env.terminal.num_storage_slots_per_row,
            max_tier_height=4,
            row_names=env.terminal.storage_row_names,
            special_areas=special_areas
        )
    
    def create_agent(self, env):
        """
        Create agent system for the current environment.
        
        Args:
            env: Current environment
            
        Returns:
            Configured TerminalAgentSystem
        """
        # Get the state dimension
        sample_obs, _ = env.reset()
        flat_state = self._flatten_state(sample_obs)
        state_dim = len(flat_state)
        
        # Get action dimensions
        action_dims = {
            'crane_movement': env.action_space['crane_movement'].nvec,
            'truck_parking': env.action_space['truck_parking'].nvec
        }
        
        # Create agent system
        agent = OptimizedTerminalAgent(
            state_dim, 
            action_dims,
            hidden_dims=[256, 256],
            head_dims=[128]
        )
        return agent
    
    def _flatten_state(self, state):
        """Flatten state dictionary to vector."""
        # Extract relevant features and flatten
        crane_positions = state['crane_positions'].flatten()
        crane_available_times = state['crane_available_times'].flatten()
        current_time = state['current_time'].flatten()
        yard_state = state['yard_state'].flatten()
        parking_status = state['parking_status'].flatten()
        rail_status = state['rail_status'].flatten()
        queue_sizes = state['queue_sizes'].flatten()
        
        # Concatenate all features
        flat_state = np.concatenate([
            crane_positions, 
            crane_available_times, 
            current_time, 
            yard_state, 
            parking_status, 
            rail_status, 
            queue_sizes
        ])
        
        return flat_state
    
    def train(self, total_episodes=None):
        """
        Train agents through curriculum stages.
        
        Args:
            total_episodes: Total episodes to train (overrides per-stage limits)
            
        Returns:
            Trained agent and training metrics
        """
        # Initialize tracking variables
        episode_count = 0
        self.training_metrics = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'curriculum_stages': []
        }
        
        # Initialize agent with first stage environment
        #print(f"Creating environment for stage 1...")
        env, _ = self.create_environment(self.curriculum_stages[self.current_stage])
        agent = self.create_agent(env)
        
        #print(f"Starting curriculum training with {len(self.curriculum_stages)} stages")
        stage_rewards = []
        
        try:
            # Train through curriculum stages
            while self.current_stage < len(self.curriculum_stages):
                stage_config = self.curriculum_stages[self.current_stage]
                #print(f"\nStarting Stage {self.current_stage + 1}")
                #print(f"Configuration: {stage_config}")
                
                # Create environment for current stage
                env, _ = self.create_environment(stage_config)
                
                # Reset stage tracking
                stage_rewards = []
                stage_episodes = 0
                max_stage_episodes = stage_config['max_episodes']
                
                # Set max episodes if provided
                if total_episodes is not None:
                    max_stage_episodes = min(max_stage_episodes, total_episodes - episode_count)
                
                # Train for this stage
                for episode in range(max_stage_episodes):
                    # Track episode
                    episode_count += 1
                    stage_episodes += 1
                    
                    #print(f"\n--- Episode {episode_count} (Stage {self.current_stage + 1}, Episode {stage_episodes}) ---")
                    # Run episode
                    try:
                        episode_reward, episode_steps, mean_loss = self.train_episode(env, agent)
                        
                        # Track metrics
                        stage_rewards.append(episode_reward)
                        self.training_metrics['episode_rewards'].append(episode_reward)
                        self.training_metrics['episode_lengths'].append(episode_steps)
                        self.training_metrics['losses'].append(mean_loss if mean_loss else 0)
                        self.training_metrics['curriculum_stages'].append(self.current_stage)
                        
                        # Display progress
                        avg_reward = np.mean(stage_rewards[-10:]) if len(stage_rewards) >= 10 else np.mean(stage_rewards)
                        #print(f"Episode {episode_count} Summary:")
                        #print(f"  Total Reward: {episode_reward:.2f}")
                        #print(f"  Avg Reward (last 10): {avg_reward:.2f}")
                        #print(f"  Steps: {episode_steps}")
                        #print(f"  Loss: {mean_loss:.4f}" if mean_loss is not None else "  Loss: N/A")
                        
                        # Save checkpoint less frequently
                        if episode_count % 20 == 0:
                            self.save_checkpoint(agent, episode_count)
                        
                        # Check if we reached the target reward
                        if avg_reward >= stage_config['target_reward'] and stage_episodes >= 10:
                            #print(f"Target reward of {stage_config['target_reward']} reached!")
                            # Save checkpoint before advancing
                            self.save_checkpoint(agent, episode_count, f"stage_{self.current_stage + 1}_complete")
                            break
                    
                    except Exception as e:
                        #print(f"Error during episode: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue with next episode
                
                # Advance to next stage
                self.current_stage += 1
                if self.current_stage >= len(self.curriculum_stages):
                    #print("Curriculum training complete!")
                    break
                    
                # Clear replay buffer before starting the next stage to avoid dimension mismatches
                #print("Clearing replay buffer for new stage...")
                agent.replay_buffer.clear()
            
            # Save final metrics and model
            self.save_results()
            self.save_checkpoint(agent, episode_count, "final")
            
        except KeyboardInterrupt:
            #print("\nTraining interrupted by user!")
            # Save checkpoint and results so far
            self.save_results()
            self.save_checkpoint(agent, episode_count, "interrupted")
        except Exception as e:
            #print(f"Unexpected error during training: {e}")
            import traceback
            traceback.print_exc()
            # Save checkpoint and results so far
            self.save_results()
            self.save_checkpoint(agent, episode_count, "error")
        
        return agent, self.training_metrics
    
    def train_episode(self, env, agent: OptimizedTerminalAgent):
        """
        Train for a single episode.
        
        Args:
            env: Terminal environment
            agent: Agent system
            
        Returns:
            Tuple of (episode_reward, episode_steps, mean_loss)
        """
        # Reset environment
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        losses = []
        done = False
        
        # Create progress bar for this episode
        sim_days = env.max_simulation_time / 86400  # Convert to days
        pbar = tqdm(total=int(sim_days), desc=f"Day Progress", unit="days", leave=False)
        last_day = 0
        
        # Tracking time when no actions are available
        waiting_time = 0
        
        # Add timeouts and stuck detection
        max_steps = 5000  # Maximum steps per episode - increased for longer episodes
        no_progress_counter = 0
        consecutive_wait_count = 0  # Count consecutive waits
        last_progress_time = env.current_simulation_time
        
        while not done and episode_steps < max_steps:
            # Periodically output progress
            if episode_steps % 100 == 0:
                print(f"Step {episode_steps}: Day {env.current_simulation_time/86400:.2f}, Total reward: {episode_reward:.2f}")
                
            # Select action based on current state with exploration
            action_masks = state['action_mask']
            
            # Debug: Check if any actions are valid
            if 'crane_movement' in action_masks:
                crane_valid = action_masks['crane_movement'].sum() > 0
            else:
                crane_valid = False
                
            if 'truck_parking' in action_masks:
                truck_valid = action_masks['truck_parking'].sum() > 0
            else:
                truck_valid = False
                
            #print(f"Valid actions available: Crane={crane_valid}, Truck={truck_valid}")
            
            action, action_type, flat_state = agent.select_action(state, action_masks, env)
            
            # Handle case where no actions are available
            if action is None:
                #print("No valid actions - waiting")
                consecutive_wait_count += 1
                
                # Use longer time increments for waiting when stuck for a while
                if consecutive_wait_count > 10:
                    advance_time = 3600  # 1 hour
                else:
                    advance_time = 300  # 5 minutes
                
                # For extreme cases, make huge jumps
                if consecutive_wait_count > 50:
                    advance_time = 6 * 3600  # 6 hours
                
                # Create a wait action - this is just a placeholder action that will be ignored
                wait_action = {'action_type': 0, 'crane_movement': np.array([0, 0, 0]), 'truck_parking': np.array([0, 0])}
                next_state, reward, done, truncated, info = env.step(wait_action)
                
                # Force the environment time to advance
                env.current_simulation_time += advance_time
                env.current_simulation_datetime += timedelta(seconds=advance_time)
                
                # Manually trigger vehicle arrivals
                if hasattr(env, '_process_vehicle_arrivals'):
                    env._process_vehicle_arrivals(advance_time)
                if hasattr(env, '_process_vehicle_departures'):
                    env._process_vehicle_departures()
                
                # Force truck/train arrivals if stuck for too long
                if consecutive_wait_count > 20:
                    #print("Forcing vehicle arrivals due to inactivity...")
                    try:
                        # Try to add a truck
                        from simulation.terminal_components.Truck import Truck
                        from simulation.terminal_components.Container import ContainerFactory
                        
                        # Find an empty parking spot
                        empty_spots = [spot for spot in env.parking_spots 
                                    if spot not in env.trucks_in_terminal]
                        
                        if empty_spots:
                            spot = empty_spots[0]
                            truck = Truck()
                            # Add a container to the truck
                            truck.add_container(ContainerFactory.create_random())
                            env.trucks_in_terminal[spot] = truck
                            #print(f"Added truck with container to spot {spot}")
                        
                        # Try to add a train
                        from simulation.terminal_components.Train import Train
                        
                        empty_tracks = [track for track in env.terminal.track_names 
                                    if track not in env.trains_in_terminal]
                        
                        if empty_tracks:
                            track = empty_tracks[0]
                            train = Train(num_wagons=3)
                            # Add containers to train
                            for _ in range(2):
                                train.add_container(ContainerFactory.create_random())
                            env.trains_in_terminal[track] = train
                            #print(f"Added train with containers to track {track}")
                    except Exception as e:
                        print(f"Error forcing vehicles: {e}")

                
                waiting_time += advance_time
                episode_steps += 1
                
                # Update current state - need to regenerate after manually changing environment
                next_state = env._get_observation()
                state = next_state
                
                # Update progress bar for waiting time
                current_day = int(env.current_simulation_time / 86400)
                if current_day > last_day:
                    pbar.update(current_day - last_day)
                    last_day = current_day
                
                # Check for progress
                if env.current_simulation_time > last_progress_time + 3600:  # 1 hour of progress
                    last_progress_time = env.current_simulation_time
                    no_progress_counter = 0
                else:
                    no_progress_counter += 1
                
                # Break if stuck (but with a higher threshold)
                if no_progress_counter >= 500:
                    #print(f"Breaking episode due to lack of progress after {no_progress_counter} steps")
                    break
                
                continue
            else:
                consecutive_wait_count = 0  # Reset consecutive wait counter when taking an action
            
            # Take action in environment
            next_state, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            episode_steps += 1
            
            # Track progress for stuck detection
            if reward > -3:  # Consider non-terrible rewards as progress
                no_progress_counter = 0
                last_progress_time = env.current_simulation_time
            else:
                no_progress_counter += 1
            
            # Break if stuck in negative reward loop
            if no_progress_counter >= 200:
                #print(f"Breaking episode due to lack of progress after {no_progress_counter} steps with negative rewards")
                break
            
            # Store experience in buffer
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
            
            # Update progress bar
            current_day = int(env.current_simulation_time / 86400)
            if current_day > last_day:
                pbar.update(current_day - last_day)
                last_day = current_day
        
        # Close progress bar
        pbar.close()
        
        # Calculate mean loss
        mean_loss = np.mean(losses) if losses else None
        
        #print(f"Episode complete - Total time waiting: {waiting_time/3600:.2f} hours")
        #print(f"Final simulation time: {env.current_simulation_time/86400:.2f} days")
        #print(f"Total steps: {episode_steps}, Total reward: {episode_reward:.2f}")
        
        # We may not have completed all days but record what we did accomplish
        current_day = int(env.current_simulation_time / 86400)
        #print(f"Completed {current_day} of {int(sim_days)} days ({current_day/max(1, int(sim_days))*100:.1f}%)")
        
        return episode_reward, episode_steps, mean_loss
    
    def save_checkpoint(self, agent, episode, suffix=None):
        """Save agent checkpoint."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"agent_checkpoint_ep{episode}"
        if suffix:
            filename += f"_{suffix}"
        filename += f"_{timestamp}.pt"
        filepath = os.path.join(self.checkpoints_dir, filename)
        agent.save(filepath)
        #print(f"Checkpoint saved to {filepath}")
    
    def save_results(self):
        """Save training metrics."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(self.results_dir, f"training_results_{timestamp}.csv")
        
        # Convert metrics to DataFrame
        results_df = pd.DataFrame({
            'episode': range(1, len(self.training_metrics['episode_rewards']) + 1),
            'reward': self.training_metrics['episode_rewards'],
            'length': self.training_metrics['episode_lengths'],
            'loss': self.training_metrics['losses'],
            'stage': self.training_metrics['curriculum_stages']
        })
        
        # Save to CSV
        results_df.to_csv(results_file, index=False)
        #print(f"Training results saved to {results_file}")
        
        # Create and save plots
        self.plot_training_results(timestamp)
    
    def plot_training_results(self, timestamp):
        """Create and save plots of training metrics."""
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot rewards
        axs[0, 0].plot(self.training_metrics['episode_rewards'])
        axs[0, 0].set_title('Episode Rewards')
        axs[0, 0].set_xlabel('Episode')
        axs[0, 0].set_ylabel('Reward')
        axs[0, 0].grid(True)
        
        # Plot smoothed rewards
        if len(self.training_metrics['episode_rewards']) > 10:
            smoothed_rewards = pd.Series(self.training_metrics['episode_rewards']).rolling(10).mean()
            axs[0, 1].plot(smoothed_rewards)
            axs[0, 1].set_title('Smoothed Episode Rewards (10-ep window)')
            axs[0, 1].set_xlabel('Episode')
            axs[0, 1].set_ylabel('Reward')
            axs[0, 1].grid(True)
        
        # Plot episode lengths
        axs[1, 0].plot(self.training_metrics['episode_lengths'])
        axs[1, 0].set_title('Episode Lengths')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('Steps')
        axs[1, 0].grid(True)
        
        # Plot losses
        if any(self.training_metrics['losses']):
            axs[1, 1].plot(self.training_metrics['losses'])
            axs[1, 1].set_title('Training Loss')
            axs[1, 1].set_xlabel('Episode')
            axs[1, 1].set_ylabel('Loss')
            axs[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.results_dir, f"training_plots_{timestamp}.png")
        plt.savefig(plot_file)
        plt.close()
        #print(f"Training plots saved to {plot_file}")
    
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
        
        # Run evaluation episodes
        eval_rewards = []
        eval_steps = []
        eval_times = []
        
        #print(f"\nEvaluating agent for {num_episodes} episodes...")
        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False
            
            while not done:
                # Select action with no exploration
                action_masks = state['action_mask']
                action, _, _ = agent.select_action(state, action_masks, env, epsilon=0)
                
                # Handle case where no actions are available
                if action is None:
                    # Wait until next available time
                    wait_action = {'action_type': 0, 'crane_movement': np.array([0, 0, 0]), 'truck_parking': np.array([0, 0])}
                    next_state, reward, done, truncated, _ = env.step(wait_action)
                else:
                    # Take action in environment
                    next_state, reward, done, truncated, _ = env.step(action)
                
                episode_reward += reward
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
            
            #print(f"Evaluation Episode {episode + 1}/{num_episodes}")
            #print(f"  Reward: {episode_reward:.2f}, Steps: {episode_steps}")
            #print(f"  Simulation time: {eval_times[-1]:.2f} days")
        
        # Calculate evaluation metrics
        evaluation_metrics = {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'mean_steps': np.mean(eval_steps),
            'mean_days': np.mean(eval_times)
        }
        
        #print("\nEvaluation Results:")
        #print(f"  Mean Reward: {evaluation_metrics['mean_reward']:.2f} ± {evaluation_metrics['std_reward']:.2f}")
        #print(f"  Mean Steps: {evaluation_metrics['mean_steps']:.2f}")
        #print(f"  Mean Simulation Days: {evaluation_metrics['mean_days']:.2f}")
        
        return evaluation_metrics

def main():
    """Main function for training terminal agents."""
    parser = argparse.ArgumentParser(description='Train terminal agents with curriculum learning')
    parser.add_argument('--config', type=str, default=None, help='Path to configuration file')
    parser.add_argument('--episodes', type=int, default=None, help='Total number of episodes')
    parser.add_argument('--checkpoints', type=str, default='checkpoints', help='Checkpoints directory')
    parser.add_argument('--results', type=str, default='results', help='Results directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', 
                      help='Computation device (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate after training')
    parser.add_argument('--render', action='store_true', help='Render during evaluation')
    
    args = parser.parse_args()
    
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
    
    # Create curriculum trainer
    trainer = CurriculumTrainer(
        base_config_path=args.config,
        checkpoints_dir=args.checkpoints,
        results_dir=args.results,
        device=args.device
    )
    
    # Train agents
    #print(f"Starting training with device: {args.device}")
    start_time = time.time()
    
    agent, metrics = trainer.train(total_episodes=args.episodes)
    
    training_time = time.time() - start_time
    #print(f"Training completed in {training_time:.2f} seconds")
    
    # Evaluate if requested
    if args.evaluate:
        #print("\nStarting evaluation...")
        trainer.evaluate(agent, num_episodes=5, render=args.render)
    
    #print("Done!")

if __name__ == "__main__":
    main()