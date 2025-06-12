# agents/hierarchical_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from simulation.agents.base_agent import BaseTransferableAgent

class AttentionMoveSelector(nn.Module):
    """Attention mechanism for selecting moves."""
    
    def __init__(self, state_dim: int, move_embed_dim: int = 30, hidden_dim: int = 256):
        super().__init__()
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Move encoder
        self.move_encoder = nn.Sequential(
            nn.Linear(move_embed_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, state: torch.Tensor, move_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: [batch_size, state_dim]
            move_embeddings: [batch_size, n_moves, move_embed_dim]
        
        Returns:
            scores: [batch_size, n_moves]
        """
        batch_size, n_moves, _ = move_embeddings.shape
        
        # Encode state
        state_encoded = self.state_encoder(state)  # [batch, hidden//2]
        state_encoded = state_encoded.unsqueeze(1).repeat(1, n_moves, 1)  # [batch, n_moves, hidden//2]
        
        # Encode moves
        moves_encoded = self.move_encoder(move_embeddings)  # [batch, n_moves, hidden//2]
        
        # Cross-attention: state attends to moves
        attn_out, _ = self.cross_attention(
            state_encoded,
            moves_encoded, 
            moves_encoded
        )  # [batch, n_moves, hidden//2]
        
        # Combine with residual
        combined = attn_out + moves_encoded
        
        # Score each move
        scores = self.output_proj(combined).squeeze(-1)  # [batch, n_moves]
        
        return scores


class HierarchicalAttentionAgent(BaseTransferableAgent):
    """Hierarchical agent that first selects move type, then specific move."""
    
    def __init__(self, state_dim: int, 
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 device: str = 'cuda'):
        super().__init__(state_dim, device)
        
        self.gamma = gamma
        
        # Networks
        self.move_type_selector = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 11)  # Number of move types
        ).to(device)
        
        self.move_selector = AttentionMoveSelector(
            state_dim=state_dim + 11,  # State + move type probs
            move_embed_dim=30
        ).to(device)
        
        # Value network (critic)
        self.value_network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        ).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.AdamW(
            list(self.move_type_selector.parameters()) + 
            list(self.move_selector.parameters()),
            lr=learning_rate,
            weight_decay=1e-5
        )
        self.critic_optimizer = optim.AdamW(
            self.value_network.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
        # PPO specific
        self.clip_param = 0.2
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.01
        
    def select_action(self, state: torch.Tensor,
                     available_moves: Dict[str, Dict],
                     move_embeddings: torch.Tensor,
                     epsilon: float = 0.0) -> Tuple[int, Dict]:
        """Select action using hierarchical policy."""
        with torch.no_grad():
            if self.training and np.random.random() < epsilon:
                # Random exploration
                move_list = list(available_moves.items())
                if move_list:
                    idx = np.random.randint(len(move_list))
                    return idx, {'move_id': move_list[idx][0], 
                               'move_data': move_list[idx][1]}
                return 0, {'move_id': 'wait', 'move_data': {'move_type': 'wait'}}
            
            # Ensure state is on device
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            # Group moves by type
            moves_by_type = self._group_moves_by_type(available_moves)
            
            # Select move type
            type_logits = self.move_type_selector(state)
            type_probs = F.softmax(type_logits, dim=-1)
            
            # Mask unavailable types
            type_mask = torch.zeros_like(type_probs)
            for move_type_idx, moves in moves_by_type.items():
                if moves:
                    type_mask[0, move_type_idx] = 1.0
            
            masked_probs = type_probs * type_mask
            if masked_probs.sum() > 0:
                masked_probs = masked_probs / masked_probs.sum()
            else:
                # No moves available
                return 0, {'move_id': 'wait', 'move_data': {'move_type': 'wait'}}
            
            # Sample move type
            move_type_idx = torch.multinomial(masked_probs, 1).item()
            
            # Get moves of selected type
            selected_moves = moves_by_type[move_type_idx]
            if not selected_moves:
                return 0, {'move_id': 'wait', 'move_data': {'move_type': 'wait'}}
            
            # Create embeddings for selected type moves
            selected_embeddings = []
            move_indices = []
            for move_id, move_data in selected_moves:
                idx = list(available_moves.keys()).index(move_id)
                move_indices.append(idx)
                selected_embeddings.append(move_embeddings[idx])
            
            selected_embeddings = torch.stack(selected_embeddings).unsqueeze(0)
            
            # Select specific move
            augmented_state = torch.cat([state, type_probs], dim=-1)
            move_scores = self.move_selector(augmented_state, selected_embeddings)
            move_probs = F.softmax(move_scores, dim=-1)
            
            # Sample move
            selected_idx = torch.multinomial(move_probs, 1).item()
            final_move_idx = move_indices[selected_idx]
            move_id, move_data = selected_moves[selected_idx]
            
            return final_move_idx, {'move_id': move_id, 'move_data': move_data,
                                   'type_probs': type_probs.cpu().numpy(),
                                   'move_probs': move_probs.cpu().numpy()}
    
    def compute_loss(self, batch: List) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss."""
        states, actions, rewards, next_states, dones, old_log_probs, returns, advantages = batch
        
        # Current policy evaluation
        type_logits = self.move_type_selector(states)
        type_probs = F.softmax(type_logits, dim=-1)
        
        # Value predictions
        values = self.value_network(states).squeeze(-1)
        
        # Compute policy loss (simplified for demonstration)
        # In full implementation, would need to reconstruct exact action probabilities
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Entropy bonus
        entropy = -(type_probs * type_probs.log()).sum(-1).mean()
        
        # Total loss
        total_loss = value_loss - self.entropy_coef * entropy
        
        return total_loss, {
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'total_loss': total_loss.item()
        }
        
    def update(self, batch_size: int = 32) -> Dict[str, float]:
        """Perform training update."""
        if len(self.memory) < batch_size:
            return {}
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        
        # Prepare batch tensors - FIXED: Convert to numpy array first
        states = np.array([e[0] for e in batch])
        states = torch.from_numpy(states).float().to(self.device)
        
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        
        next_states = np.array([e[3] for e in batch])
        next_states = torch.from_numpy(next_states).float().to(self.device)
        
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        # Compute returns and advantages (simplified)
        with torch.no_grad():
            next_values = self.value_network(next_states).squeeze(-1)
            returns = rewards + self.gamma * next_values * (1 - dones)
            
            values = self.value_network(states).squeeze(-1)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Placeholder for old log probs (would be computed during collection)
        old_log_probs = torch.zeros_like(rewards)
        
        # Compute loss
        loss, metrics = self.compute_loss([
            states, actions, rewards, next_states, dones,
            old_log_probs, returns, advantages
        ])
        
        # Update networks
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.move_type_selector.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.move_selector.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        self.steps += 1
        
        return metrics
    
    def _group_moves_by_type(self, available_moves: Dict[str, Dict]) -> Dict[int, List]:
        """Group moves by their type index."""
        move_types = {
            'wait': 0, 'train_to_truck': 1, 'truck_to_train': 2,
            'yard_to_train': 3, 'train_to_yard': 4, 'yard_to_truck': 5,
            'truck_to_yard': 6, 'yard_to_yard': 7, 'yard_to_stack': 8,
            'from_yard': 9, 'to_yard': 10
        }
        
        grouped = {i: [] for i in range(11)}
        
        for move_id, move_data in available_moves.items():
            move_type = move_data.get('move_type', 'wait')
            type_idx = move_types.get(move_type, 0)
            grouped[type_idx].append((move_id, move_data))
        
        return grouped
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'move_type_selector': self.move_type_selector.state_dict(),
            'move_selector': self.move_selector.state_dict(),
            'value_network': self.value_network.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.move_type_selector.load_state_dict(checkpoint['move_type_selector'])
        self.move_selector.load_state_dict(checkpoint['move_selector'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.steps = checkpoint.get('steps', 0)
        self.episodes = checkpoint.get('episodes', 0)