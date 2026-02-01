# multihead_dqn/state_utils.py
"""State encoding utilities at split resolution."""
import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass


@dataclass
class ChannelSpec:
    """Specification for state tensor channels."""
    # Core channels (always present)
    OCCUPANCY: int = 0          # 1.0 if position occupied
    CONTAINER_START: int = 1    # 1.0 at leftmost split of container
    
    # Container type (categorical encoding)
    CONTAINER_TYPE: int = 2     # 0=empty, 0.25=regular, 0.5=reefer, 0.75=DG, 1.0=swap
    
    # State channels
    ACCESSIBLE: int = 3         # 1.0 if container is accessible (top of stack)
    DEPARTURE_URGENCY: int = 4  # normalized days until departure (0=urgent, 1=far)
    
    # Relational channels
    BLOCKS_URGENT: int = 5      # 1.0 if blocking something departing sooner
    
    # Heat/context channels
    TRAIN_HEAT: int = 6         # proximity to train anchors
    TRUCK_HEAT: int = 7         # proximity to truck activity
    
    @classmethod
    def num_channels(cls) -> int:
        return 8


class SplitLevelEncoder:
    """
    Encodes yard state at split resolution.
    Output shape: (C, R, S, T) where S = n_bays * split_factor
    """
    
    def __init__(
        self,
        n_rows: int,
        n_bays: int,
        n_tiers: int,
        split_factor: int
    ):
        self.n_rows = n_rows
        self.n_bays = n_bays
        self.n_tiers = n_tiers
        self.split_factor = split_factor
        self.total_splits = n_bays * split_factor
        self.channels = ChannelSpec()
    
    def encode(
        self,
        containers: Dict,  # container_id -> ContainerRecord
        container_length_map: Dict[float, int],  # length_ft -> n_splits
        current_time,  # datetime
        train_anchors: Optional[Dict[str, int]] = None,  # train_id -> anchor_bay
        truck_bays: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Encode yard state to tensor.
        
        Args:
            containers: Dict mapping container_id to ContainerRecord
            container_length_map: Maps container length to split count
            current_time: Current simulation time
            train_anchors: Train anchor bay positions
            truck_bays: Active truck bay positions
            
        Returns:
            State tensor (C, R, S, T)
        """
        C = ChannelSpec.num_channels()
        R, S, T = self.n_rows, self.total_splits, self.n_tiers
        
        tensor = np.zeros((C, R, S, T), dtype=np.float32)
        
        # Track blocking relationships for BLOCKS_URGENT channel
        position_urgency: Dict[Tuple[int, int, int], float] = {}  # (r, s, t) -> urgency
        
        # First pass: encode container properties
        for cid, rec in containers.items():
            r = rec.placement.row
            bay = rec.placement.bay
            tier = rec.placement.tier
            start_split = rec.placement.start_split
            
            # Compute absolute split range
            abs_start = bay * self.split_factor + start_split
            n_splits = rec.n_splits
            abs_end = abs_start + n_splits
            
            # Clamp to bounds
            abs_end = min(abs_end, self.total_splits)
            
            c = rec.container
            
            # Occupancy - all splits
            tensor[ChannelSpec.OCCUPANCY, r, abs_start:abs_end, tier] = 1.0
            
            # Start marker - only first split
            tensor[ChannelSpec.CONTAINER_START, r, abs_start, tier] = 1.0
            
            # Container type (categorical)
            type_val = self._encode_container_type(c)
            tensor[ChannelSpec.CONTAINER_TYPE, r, abs_start:abs_end, tier] = type_val
            
            # Accessibility
            if rec.is_accessible:
                tensor[ChannelSpec.ACCESSIBLE, r, abs_start:abs_end, tier] = 1.0
            
            # Departure urgency
            urgency = self._compute_urgency(c, current_time)
            tensor[ChannelSpec.DEPARTURE_URGENCY, r, abs_start:abs_end, tier] = urgency
            
            # Store for blocking analysis
            position_urgency[(r, abs_start, tier)] = urgency
        
        # Second pass: compute blocking relationships
        self._encode_blocking(tensor, containers, position_urgency)
        
        # Encode heat maps
        if train_anchors:
            self._encode_train_heat(tensor, train_anchors)
        if truck_bays:
            self._encode_truck_heat(tensor, truck_bays)
        
        return tensor
    
    def _encode_container_type(self, container) -> float:
        """Categorical encoding of container type."""
        if hasattr(container, 'is_swap_body') and container.is_swap_body:
            return 1.0
        if hasattr(container, 'is_trailer') and container.is_trailer:
            return 1.0
        
        goods_type = getattr(container, 'goods_type', 'Regular')
        if goods_type == 'Reefer':
            return 0.5
        if goods_type == 'DangerousGoods':
            return 0.75
        return 0.25  # Regular
    
    def _compute_urgency(self, container, current_time) -> float:
        """Compute normalized departure urgency (0=urgent, 1=far)."""
        try:
            if hasattr(container, 'days_until_departure'):
                days = container.days_until_departure(current_time)
            elif hasattr(container, 'departure_date') and container.departure_date:
                delta = container.departure_date - current_time
                days = delta.total_seconds() / 86400.0
            else:
                days = 30.0  # Default
            
            # Normalize: 0 days = 0.0 (urgent), 30+ days = 1.0 (not urgent)
            return min(1.0, max(0.0, days / 30.0))
        except Exception:
            return 0.5  # Unknown
    
    def _encode_blocking(
        self,
        tensor: np.ndarray,
        containers: Dict,
        position_urgency: Dict
    ):
        """Encode blocking relationships channel."""
        for cid, rec in containers.items():
            r = rec.placement.row
            bay = rec.placement.bay
            tier = rec.placement.tier
            abs_start = bay * self.split_factor + rec.placement.start_split
            abs_end = abs_start + rec.n_splits
            
            # Check if blocking something below
            if tier > 0:
                below_tier = tier - 1
                # Check position directly below (same start)
                below_key = (r, abs_start, below_tier)
                
                if below_key in position_urgency:
                    my_urgency = position_urgency.get((r, abs_start, tier), 0.5)
                    below_urgency = position_urgency[below_key]
                    
                    # If I'm less urgent than what's below, I'm blocking
                    if my_urgency > below_urgency:
                        blocking_severity = my_urgency - below_urgency
                        tensor[ChannelSpec.BLOCKS_URGENT, r, abs_start:abs_end, tier] = blocking_severity
    
    def _encode_train_heat(self, tensor: np.ndarray, train_anchors: Dict[str, int]):
        """Encode proximity to train anchor bays."""
        if not train_anchors:
            return
        
        R, S, T = self.n_rows, self.total_splits, self.n_tiers
        heat = np.zeros(S, dtype=np.float32)
        
        for train_id, anchor_bay in train_anchors.items():
            anchor_split = anchor_bay * self.split_factor + self.split_factor // 2
            anchor_split = min(max(0, anchor_split), S - 1)
            
            # Gaussian-like heat around anchor
            for s in range(S):
                dist = abs(s - anchor_split) / self.split_factor
                heat[s] += np.exp(-0.5 * (dist / 3.0) ** 2)  # σ = 3 bays
        
        # Normalize
        if heat.max() > 0:
            heat /= heat.max()
        
        # Broadcast to all rows and tiers
        tensor[ChannelSpec.TRAIN_HEAT, :, :, :] = heat[None, :, None]
    
    def _encode_truck_heat(self, tensor: np.ndarray, truck_bays: List[int]):
        """Encode proximity to active trucks."""
        if not truck_bays:
            return
        
        R, S, T = self.n_rows, self.total_splits, self.n_tiers
        heat = np.zeros(S, dtype=np.float32)
        
        for bay in truck_bays:
            center_split = bay * self.split_factor + self.split_factor // 2
            center_split = min(max(0, center_split), S - 1)
            
            for s in range(S):
                dist = abs(s - center_split) / self.split_factor
                heat[s] += np.exp(-0.5 * (dist / 2.0) ** 2)  # Tighter spread
        
        if heat.max() > 0:
            heat /= heat.max()
        
        tensor[ChannelSpec.TRUCK_HEAT, :, :, :] = heat[None, :, None]
    
    def get_occupancy_mask(self, tensor: np.ndarray) -> np.ndarray:
        """Extract occupancy mask from state tensor."""
        return tensor[ChannelSpec.OCCUPANCY] > 0.5
    
    def get_validity_mask(
        self,
        occupancy_mask: np.ndarray,
        n_splits_needed: int,
        goods_mask: Optional[np.ndarray] = None,
        support_mask: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Compute valid placement mask for a container.
        
        Args:
            occupancy_mask: (R, S, T) current occupancy
            n_splits_needed: Number of splits the container needs
            goods_mask: (R, S, T) where container type is allowed
            support_mask: (R, S, T) where support exists (for tier > 0)
            
        Returns:
            validity_mask: (R, S, T) where placement is valid
        """
        R, S, T = occupancy_mask.shape
        
        # Start with unoccupied positions
        unoccupied = ~occupancy_mask
        
        # Check contiguous space using rolling sum
        valid = np.zeros((R, S, T), dtype=bool)
        
        for tier in range(T):
            for row in range(R):
                # Rolling sum to check contiguous availability
                cumsum = np.cumsum(unoccupied[row, :, tier].astype(int))
                cumsum = np.insert(cumsum, 0, 0)
                
                for start in range(S - n_splits_needed + 1):
                    end = start + n_splits_needed
                    available = cumsum[end] - cumsum[start]
                    
                    if available == n_splits_needed:
                        # All splits in range are free
                        # Mark start position as valid
                        valid[row, start, tier] = True
        
        # Apply goods mask if provided
        if goods_mask is not None:
            valid &= goods_mask
        
        # Apply support mask if provided (for tier > 0)
        if support_mask is not None:
            for tier in range(1, T):
                valid[:, :, tier] &= support_mask[:, :, tier]
        
        return valid