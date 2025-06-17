# agents/base_agent.py
import torch
import torch.nn as nn
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
from collections import deque
import random

class MoveEmbedder:
    """Creates size-invariant embeddings for moves."""
    
    def __init__(self, n_rows: int, n_bays: int, n_tiers: int):
        self.n_rows = n_rows
        self.n_bays = n_bays
        self.n_tiers = n_tiers
        
        # Move type to index mapping
        self.move_types = {
            'wait': 0,
            'train_to_truck': 1,
            'truck_to_train': 2,
            'yard_to_train': 3,
            'train_to_yard': 4,
            'yard_to_truck': 5,
            'truck_to_yard': 6,
            'yard_to_yard': 7,
            'yard_to_stack': 8,
            'from_yard': 9,
            'to_yard': 10
        }
        
    def embed_move(self, move_data: Dict, 
                   container_info: Optional[Dict] = None,
                   terminal_state: Optional[Dict] = None) -> np.ndarray:
        """Create normalized embedding for a single move."""
        features = []
        
        # 1. Move type one-hot encoding (11 dimensions)
        move_type = move_data.get('move_type', 'wait')
        move_type_vec = np.zeros(len(self.move_types))
        move_type_idx = self.move_types.get(move_type, 0)
        move_type_vec[move_type_idx] = 1.0
        features.extend(move_type_vec)
        
        # 2. Source position features (normalized, 4 dimensions)
        src_features = self._encode_position(move_data.get('source_pos'), 
                                           move_data.get('source_type'))
        features.extend(src_features)
        
        # 3. Destination position features (normalized, 4 dimensions)
        dst_features = self._encode_position(move_data.get('dest_pos'),
                                           move_data.get('dest_type'))
        features.extend(dst_features)
        
        # 4. Priority and urgency (2 dimensions)
        priority = move_data.get('priority', 5.0) / 10.0  # Normalize to [0,1]
        urgency = move_data.get('urgency', 0.0) / 30.0   # Normalize urgency score
        features.extend([priority, urgency])
        
        # 5. Container properties (if available, 5 dimensions)
        if container_info:
            features.extend([
                container_info.get('days_waiting', 0) / 10.0,
                1.0 if container_info.get('is_export', False) else 0.0,
                1.0 if container_info.get('has_pickup', False) else 0.0,
                1.0 if container_info.get('is_reefer', False) else 0.0,
                1.0 if container_info.get('is_dangerous', False) else 0.0
            ])
        else:
            features.extend([0.0] * 5)
        
        # 6. Terminal state context (if available, 4 dimensions)
        if terminal_state:
            features.extend([
                terminal_state.get('train_urgency', 0.0),  # Trains departing soon
                terminal_state.get('truck_waiting', 0.0),  # Trucks waiting
                terminal_state.get('yard_congestion', 0.0),  # Yard fullness
                terminal_state.get('time_pressure', 0.0)  # Time of day pressure
            ])
        else:
            features.extend([0.0] * 4)
        
        # Total: 30 dimensions
        return np.array(features, dtype=np.float32)
    
    def _encode_position(self, position: Any, pos_type: str) -> List[float]:
        """Encode position to normalized features."""
        if pos_type == 'yard':
            # Handle list of coordinates (multi-slot containers)
            if isinstance(position, list) and len(position) > 0:
                # Use the first coordinate as the primary position
                first_coord = position[0]
                if len(first_coord) >= 4:
                    row, bay, split, tier = first_coord[:4]
                else:
                    row, bay, tier = first_coord[:3]
                    split = 0
            # Handle single coordinate
            elif isinstance(position, tuple):
                if len(position) >= 4:
                    row, bay, split, tier = position[:4]
                else:
                    row, bay, tier = position[:3]
                    split = 0
            else:
                # Fallback for unexpected format
                return [0.0, 0.0, 0.0, 0.0]
            
            # Normalize the position
            return [
                row / max(1, self.n_rows - 1),
                bay / max(1, self.n_bays - 1),
                tier / max(1, self.n_tiers - 1),
                0.0  # Yard indicator
            ]
        elif pos_type == 'train':
            return [0.5, 0.5, 0.0, 1.0]  # Train indicator
        elif pos_type == 'truck':
            return [0.5, 0.5, 0.0, 2.0]  # Truck indicator
        else:
            return [0.0, 0.0, 0.0, 3.0]  # Unknown/stack


class BaseTransferableAgent(ABC):
    """Base class for size-invariant terminal agents."""
    
    def __init__(self, state_dim: int, device: str = 'cuda'):
        self.state_dim = state_dim
        self.device = device
        self.training = True
        
        # Experience replay
        self.memory = deque(maxlen=100000)
        
        # Training stats
        self.steps = 0
        self.episodes = 0
        
    def store_experience(self, state, action, reward, next_state, done, info):
        """Store experience with additional move information."""
        # Ensure info contains move_list for proper Q-value calculation
        if 'available_moves' in info and 'move_list' not in info:
            info['move_list'] = list(info['available_moves'].keys())
        
        self.memory.append((state, action, reward, next_state, done, info))

    @abstractmethod
    def select_action(self, state: torch.Tensor, 
                     available_moves: Dict[str, Dict],
                     move_embeddings: torch.Tensor,
                     epsilon: float = 0.0) -> Tuple[int, Dict]:
        """Select action given state and available moves."""
        pass
    
    @abstractmethod
    def compute_loss(self, batch: List) -> torch.Tensor:
        """Compute training loss."""
        pass
    
    @abstractmethod
    def update(self, batch_size: int = 32) -> Dict[str, float]:
        """Perform training update."""
        pass
    
    def store_transition(self, state, action, reward, next_state, done, info):
        """Store experience in replay buffer."""
        self.memory.append((state, action, reward, next_state, done, info))
    
    def train_mode(self):
        """Set to training mode."""
        self.training = True
    
    def eval_mode(self):
        """Set to evaluation mode."""
        self.training = False