# agents/gnn_agent.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.data import Data, Batch
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from simulation.agents.base_agent import BaseTransferableAgent

class TerminalGraphBuilder:
    """Builds graph representation of terminal state."""
    
    def __init__(self, n_rows: int, n_bays: int, n_tiers: int):
        self.n_rows = n_rows
        self.n_bays = n_bays  
        self.n_tiers = n_tiers
        
    def build_graph(self, terminal_state: Dict, available_moves: Dict[str, Dict]) -> Data:
        """Build graph from terminal state and moves."""
        # Node features: positions in terminal
        node_features = []
        node_positions = {}
        node_idx = 0
        
        # Add yard positions as nodes
        for row in range(self.n_rows):
            for bay in range(self.n_bays):
                for tier in range(self.n_tiers):
                    features = [
                        row / self.n_rows,
                        bay / self.n_bays,
                        tier / self.n_tiers,
                        1.0,  # Yard node
                        0.0, 0.0  # Padding
                    ]
                    node_features.append(features)
                    node_positions[f"yard_{row}_{bay}_{tier}"] = node_idx
                    node_idx += 1
        
        # Add vehicle nodes (simplified)
        for i in range(10):  # Max 10 trains
            features = [0.5, float(i)/10, 0.0, 0.0, 1.0, 0.0]
            node_features.append(features)
            node_positions[f"train_{i}"] = node_idx
            node_idx += 1
            
        for i in range(20):  # Max 20 trucks
            features = [0.5, float(i)/20, 0.0, 0.0, 0.0, 1.0]
            node_features.append(features)
            node_positions[f"truck_{i}"] = node_idx
            node_idx += 1
        
        # Build edges from moves
        edge_index = []
        edge_attr = []
        
        for move_id, move_data in available_moves.items():
            # Simplified edge construction
            # In real implementation, would map positions to node indices
            src_idx = np.random.randint(0, len(node_features))
            dst_idx = np.random.randint(0, len(node_features))
            
            edge_index.append([src_idx, dst_idx])
            
            # Edge features: move properties
            edge_features = [
                move_data.get('priority', 5.0) / 10.0,
                1.0 if move_data.get('move_type') == 'yard_to_train' else 0.0,
                1.0 if move_data.get('move_type') == 'train_to_yard' else 0.0,
                1.0 if move_data.get('move_type') == 'yard_to_yard' else 0.0,
            ]
            edge_attr.append(edge_features)
        
        # Convert to tensors
        x = torch.FloatTensor(node_features)
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        edge_attr = torch.FloatTensor(edge_attr)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class GraphPolicyNetwork(nn.Module):
    """GNN-based policy network."""
    
    def __init__(self, node_features: int = 6, edge_features: int = 4, 
                 hidden_dim: int = 128):
        super().__init__()
        
        # Node embedding
        self.node_encoder = nn.Sequential(
            nn.Linear(node_features, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Edge embedding
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_features, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Graph attention layers
        self.gat1 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.gat2 = GATConv(hidden_dim, hidden_dim, heads=4, concat=False)
        self.gat3 = GATConv(hidden_dim, hidden_dim, heads=2, concat=False)
        
        # Edge scorer
        self.edge_scorer = nn.Sequential(
            nn.Linear(hidden_dim * 2 + hidden_dim // 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )
        
    def forward(self, data: Data) -> torch.Tensor:
        """Forward pass returning edge scores."""
        # Encode nodes
        x = self.node_encoder(data.x)
        
        # Message passing
        x = F.relu(self.gat1(x, data.edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = F.relu(self.gat2(x, data.edge_index))
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.gat3(x, data.edge_index)
        
        # Score edges
        edge_scores = []
        edge_features = self.edge_encoder(data.edge_attr)
        
        for i, (src, dst) in enumerate(data.edge_index.t()):
            # Combine source, destination, and edge features
            combined = torch.cat([
                x[src],
                x[dst],
                edge_features[i]
            ])
            score = self.edge_scorer(combined)
            edge_scores.append(score)
        
        return torch.cat(edge_scores)


class GNNAgent(BaseTransferableAgent):
    """Graph Neural Network based agent."""
    
    def __init__(self, state_dim: int,
                 n_rows: int, n_bays: int, n_tiers: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 device: str = 'cuda'):
        super().__init__(state_dim, device)
        
        self.gamma = gamma
        self.graph_builder = TerminalGraphBuilder(n_rows, n_bays, n_tiers)
        
        # Networks
        self.policy_network = GraphPolicyNetwork().to(device)
        
        # Value network (uses state vector)
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
        self.policy_optimizer = optim.AdamW(
            self.policy_network.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        self.value_optimizer = optim.AdamW(
            self.value_network.parameters(),
            lr=learning_rate,
            weight_decay=1e-5
        )
        
    def select_action(self, state: torch.Tensor,
                     available_moves: Dict[str, Dict],
                     move_embeddings: torch.Tensor,
                     epsilon: float = 0.0,
                     terminal_state: Optional[Dict] = None) -> Tuple[int, Dict]:
        """Select action using graph policy."""
        with torch.no_grad():
            if self.training and np.random.random() < epsilon:
                # Random exploration
                move_list = list(available_moves.items())
                if move_list:
                    idx = np.random.randint(len(move_list))
                    return idx, {'move_id': move_list[idx][0],
                               'move_data': move_list[idx][1]}
                return 0, {'move_id': 'wait', 'move_data': {'move_type': 'wait'}}
            
            # Build graph
            if terminal_state is None:
                terminal_state = {}
            graph = self.graph_builder.build_graph(terminal_state, available_moves)
            graph = graph.to(self.device)
            
            # Get edge scores
            edge_scores = self.policy_network(graph)
            
            # Apply softmax to get probabilities
            edge_probs = F.softmax(edge_scores, dim=0)
            
            # Sample edge (move)
            if len(edge_probs) > 0:
                selected_edge = torch.multinomial(edge_probs, 1).item()
                
                # Map edge to move index
                move_list = list(available_moves.items())
                if selected_edge < len(move_list):
                    move_id, move_data = move_list[selected_edge]
                    return selected_edge, {'move_id': move_id,
                                         'move_data': move_data,
                                         'edge_probs': edge_probs.cpu().numpy()}
            
            return 0, {'move_id': 'wait', 'move_data': {'move_type': 'wait'}}
    
    def compute_loss(self, batch: List) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute A2C loss."""
        states, actions, rewards, next_states, dones, advantages, returns = batch
        
        # Value predictions
        values = self.value_network(states).squeeze(-1)
        
        # Value loss
        value_loss = F.mse_loss(values, returns)
        
        # Policy loss would require graph reconstruction
        # Simplified for demonstration
        policy_loss = torch.tensor(0.0).to(self.device)
        
        # Total loss
        total_loss = value_loss + policy_loss
        
        return total_loss, {
            'value_loss': value_loss.item(),
            'policy_loss': policy_loss.item(),
            'total_loss': total_loss.item()
        }
    
    def update(self, batch_size: int = 32) -> Dict[str, float]:
        """Perform training update."""
        if len(self.memory) < batch_size:
            return {}
        
        # Sample batch
        batch = random.sample(self.memory, batch_size)
        
        # Prepare batch tensors
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        # Compute returns and advantages
        with torch.no_grad():
            next_values = self.value_network(next_states).squeeze(-1)
            returns = rewards + self.gamma * next_values * (1 - dones)
            
            values = self.value_network(states).squeeze(-1)
            advantages = returns - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute loss
        loss, metrics = self.compute_loss([
            states, actions, rewards, next_states, dones,
            advantages, returns
        ])
        
        # Update networks
        self.policy_optimizer.zero_grad()
        self.value_optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), 1.0)
        
        self.policy_optimizer.step()
        self.value_optimizer.step()
        
        self.steps += 1
        
        return metrics
    
    def save(self, path: str):
        """Save model weights."""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'value_network': self.value_network.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes
        }, path)
    
    def load(self, path: str):
        """Load model weights."""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint['policy_network'])
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.steps = checkpoint.get('steps', 0)
        self.episodes = checkpoint.get('episodes', 0)