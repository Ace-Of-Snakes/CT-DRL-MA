# simulation/terminal_components/storage/tensor_converter.py

import numpy as np
import torch
from typing import Optional, Dict, List
from simulation.terminal_components.storage.StorageYard import OptimizedStorageYard
from simulation.terminal_components.storage.storage_constants import *

class YardTensorConverter:
    """Converts yard state to tensor representations for CNN/DRL models."""
    
    # Channel definitions
    CHANNEL_OCCUPANCY = 0
    CHANNEL_CONTAINER_20FT = 1
    CHANNEL_CONTAINER_22FT = 2
    CHANNEL_CONTAINER_23FT = 3
    CHANNEL_CONTAINER_26FT = 4
    CHANNEL_CONTAINER_40FT = 5
    CHANNEL_CONTAINER_45FT = 6
    CHANNEL_EXCLUSIVE = 7  # Trailers/Swap Bodies
    CHANNEL_GOODS_REGULAR = 8
    CHANNEL_GOODS_REEFER = 9
    CHANNEL_GOODS_DANGEROUS = 10
    CHANNEL_ACCESSIBILITY = 11
    CHANNEL_STACK_HEIGHT = 12
    CHANNEL_SPECIAL_REEFER = 13
    CHANNEL_SPECIAL_DG = 14
    CHANNEL_SPECIAL_GROUND = 15
    
    TOTAL_CHANNELS = 16
    
    def __init__(self, yard: OptimizedStorageYard, device: str = 'cpu'):
        """
        Initialize converter.
        
        Args:
            yard: Storage yard instance
            device: Torch device ('cpu' or 'cuda')
        """
        self.yard = yard
        self.device = device
        
        # Pre-allocate tensors for efficiency
        self.tensor_shape = (
            self.TOTAL_CHANNELS,
            yard.n_rows,
            yard.n_bays * YARD_SPLIT_FACTOR,  # Full resolution
            yard.n_tiers
        )
        
    def to_tensor(self, normalize: bool = True, resolution: str = 'full') -> torch.Tensor:
        """
        Convert yard state to 4D tensor.
        
        Args:
            normalize: Whether to normalize values to [0, 1]
            resolution: 'full' for subslot resolution, 'bay' for bay-level
            
        Returns:
            Tensor of shape (C, H, W, D) where:
            - C: channels
            - H: rows
            - W: bays * split_factor (or just bays if resolution='bay')
            - D: tiers
        """
        if resolution == 'full':
            return self._full_resolution_tensor(normalize)
        else:
            return self._bay_resolution_tensor(normalize)
    
    def _full_resolution_tensor(self, normalize: bool) -> torch.Tensor:
        """Create full subslot resolution tensor."""
        # Initialize tensor
        tensor = np.zeros(self.tensor_shape, dtype=np.float32)
        
        # Channel 0: Occupancy
        for row in range(self.yard.n_rows):
            for tier in range(self.yard.n_tiers):
                for bay in range(self.yard.n_bays):
                    bits = self.yard._get_occupancy_bits(row, tier, bay)
                    for split in range(YARD_SPLIT_FACTOR):
                        if bits & (1 << split):
                            tensor[self.CHANNEL_OCCUPANCY, row, bay * YARD_SPLIT_FACTOR + split, tier] = 1.0
        
        # Channels 1-7: Container types (one-hot)
        # Channels 8-10: Goods types (one-hot)
        for container_id, (container, position) in self.yard.containers.items():
            positions = position.get_subslot_positions(self.yard.mapper)
            
            for row, bay, tier, split in positions:
                idx = bay * YARD_SPLIT_FACTOR + split
                
                # Container type channels
                if container.container_type == "20ft" or container.container_type == "TEU":
                    tensor[self.CHANNEL_CONTAINER_20FT, row, idx, tier] = 1.0
                elif container.container_type == "22ft":
                    tensor[self.CHANNEL_CONTAINER_22FT, row, idx, tier] = 1.0
                elif container.container_type == "23ft":
                    tensor[self.CHANNEL_CONTAINER_23FT, row, idx, tier] = 1.0
                elif container.container_type == "26ft":
                    tensor[self.CHANNEL_CONTAINER_26FT, row, idx, tier] = 1.0
                elif container.container_type == "40ft" or container.container_type == "FEU":
                    tensor[self.CHANNEL_CONTAINER_40FT, row, idx, tier] = 1.0
                elif container.container_type == "45ft":
                    tensor[self.CHANNEL_CONTAINER_45FT, row, idx, tier] = 1.0
                elif container.container_type in EXCLUSIVE_TYPES:
                    tensor[self.CHANNEL_EXCLUSIVE, row, idx, tier] = 1.0
                
                # Goods type channels
                if container.goods_type == GOODS_REGULAR:
                    tensor[self.CHANNEL_GOODS_REGULAR, row, idx, tier] = 1.0
                elif container.goods_type == GOODS_REEFER:
                    tensor[self.CHANNEL_GOODS_REEFER, row, idx, tier] = 1.0
                elif container.goods_type == GOODS_DANGEROUS:
                    tensor[self.CHANNEL_GOODS_DANGEROUS, row, idx, tier] = 1.0
        
        # Channel 11: Accessibility (1 if accessible, 0 if blocked)
        for row in range(self.yard.n_rows):
            for bay in range(self.yard.n_bays):
                for tier in range(self.yard.n_tiers):
                    if self.yard._is_position_accessible(row, bay, tier):
                        for split in range(YARD_SPLIT_FACTOR):
                            idx = bay * YARD_SPLIT_FACTOR + split
                            tensor[self.CHANNEL_ACCESSIBILITY, row, idx, tier] = 1.0
        
        # Channel 12: Stack height (normalized)
        for row in range(self.yard.n_rows):
            for bay in range(self.yard.n_bays):
                # Find highest occupied tier
                height = 0
                for tier in range(self.yard.n_tiers):
                    if self.yard._get_occupancy_bits(row, tier, bay) != 0:
                        height = tier + 1
                
                # Set normalized height for all subslots in bay
                normalized_height = height / self.yard.n_tiers if normalize else height
                for split in range(YARD_SPLIT_FACTOR):
                    idx = bay * YARD_SPLIT_FACTOR + split
                    for tier in range(self.yard.n_tiers):
                        tensor[self.CHANNEL_STACK_HEIGHT, row, idx, tier] = normalized_height
        
        # Channels 13-15: Special area masks
        for row in range(self.yard.n_rows):
            for bay in range(self.yard.n_bays):
                for split in range(YARD_SPLIT_FACTOR):
                    idx = bay * YARD_SPLIT_FACTOR + split
                    
                    if self.yard.reefer_mask[row, bay]:
                        for tier in range(self.yard.n_tiers):
                            tensor[self.CHANNEL_SPECIAL_REEFER, row, idx, tier] = 1.0
                    
                    if self.yard.dg_mask[row, bay]:
                        for tier in range(self.yard.n_tiers):
                            tensor[self.CHANNEL_SPECIAL_DG, row, idx, tier] = 1.0
                    
                    if self.yard.ground_only_mask[row, bay]:
                        tensor[self.CHANNEL_SPECIAL_GROUND, row, idx, 0] = 1.0
        
        return torch.from_numpy(tensor).to(self.device)
    
    def _bay_resolution_tensor(self, normalize: bool) -> torch.Tensor:
        """Create bay-level resolution tensor (aggregated)."""
        shape = (self.TOTAL_CHANNELS, self.yard.n_rows, self.yard.n_bays, self.yard.n_tiers)
        tensor = np.zeros(shape, dtype=np.float32)
        
        # Aggregate occupancy by bay
        for row in range(self.yard.n_rows):
            for tier in range(self.yard.n_tiers):
                for bay in range(self.yard.n_bays):
                    bits = self.yard._get_occupancy_bits(row, tier, bay)
                    occupancy_rate = bin(bits).count('1') / YARD_SPLIT_FACTOR
                    tensor[self.CHANNEL_OCCUPANCY, row, bay, tier] = occupancy_rate
        
        # Aggregate container types by bay
        for container_id, (container, position) in self.yard.containers.items():
            row, bay, tier = position.row, position.bay, position.tier
            
            # Container type channels (use max to handle multiple containers in bay)
            if container.container_type in ["20ft", "22ft", "TEU"]:
                tensor[self.CHANNEL_CONTAINER_20FT, row, bay, tier] = 1.0
            elif container.container_type == "23ft":
                tensor[self.CHANNEL_CONTAINER_23FT, row, bay, tier] = 1.0
            elif container.container_type == "26ft":
                tensor[self.CHANNEL_CONTAINER_26FT, row, bay, tier] = 1.0
            elif container.container_type in ["40ft", "FEU"]:
                tensor[self.CHANNEL_CONTAINER_40FT, row, bay, tier] = 1.0
            elif container.container_type == "45ft":
                tensor[self.CHANNEL_CONTAINER_45FT, row, bay, tier] = 1.0
            elif container.container_type in EXCLUSIVE_TYPES:
                tensor[self.CHANNEL_EXCLUSIVE, row, bay, tier] = 1.0
            
            # Goods type channels
            if container.goods_type == GOODS_REGULAR:
                tensor[self.CHANNEL_GOODS_REGULAR, row, bay, tier] = 1.0
            elif container.goods_type == GOODS_REEFER:
                tensor[self.CHANNEL_GOODS_REEFER, row, bay, tier] = 1.0
            elif container.goods_type == GOODS_DANGEROUS:
                tensor[self.CHANNEL_GOODS_DANGEROUS, row, bay, tier] = 1.0
        
        # Accessibility and height channels
        for row in range(self.yard.n_rows):
            for bay in range(self.yard.n_bays):
                # Accessibility
                for tier in range(self.yard.n_tiers):
                    if self.yard._is_position_accessible(row, bay, tier):
                        tensor[self.CHANNEL_ACCESSIBILITY, row, bay, tier] = 1.0
                
                # Stack height
                height = 0
                for tier in range(self.yard.n_tiers):
                    if self.yard._get_occupancy_bits(row, tier, bay) != 0:
                        height = tier + 1
                
                normalized_height = height / self.yard.n_tiers if normalize else height
                for tier in range(self.yard.n_tiers):
                    tensor[self.CHANNEL_STACK_HEIGHT, row, bay, tier] = normalized_height
        
        # Special area masks
        for row in range(self.yard.n_rows):
            for bay in range(self.yard.n_bays):
                if self.yard.reefer_mask[row, bay]:
                    for tier in range(self.yard.n_tiers):
                        tensor[self.CHANNEL_SPECIAL_REEFER, row, bay, tier] = 1.0
                
                if self.yard.dg_mask[row, bay]:
                    for tier in range(self.yard.n_tiers):
                        tensor[self.CHANNEL_SPECIAL_DG, row, bay, tier] = 1.0
                
                if self.yard.ground_only_mask[row, bay]:
                    tensor[self.CHANNEL_SPECIAL_GROUND, row, bay, 0] = 1.0
        
        return torch.from_numpy(tensor).to(self.device)
    
    def get_action_mask(self) -> torch.Tensor:
        """
        Get mask of valid actions for DRL.
        
        Returns:
            Boolean tensor indicating which positions can accept containers
        """
        mask_shape = (self.yard.n_rows, self.yard.n_bays * YARD_SPLIT_FACTOR, self.yard.n_tiers)
        mask = np.zeros(mask_shape, dtype=bool)
        
        # Mark all free and accessible positions
        for row in range(self.yard.n_rows):
            for tier in range(self.yard.n_tiers):
                for bay in range(self.yard.n_bays):
                    bits = self.yard._get_occupancy_bits(row, tier, bay)
                    for split in range(YARD_SPLIT_FACTOR):
                        idx = bay * YARD_SPLIT_FACTOR + split
                        if not (bits & (1 << split)):  # Position is free
                            # Check if position is valid for placement
                            if tier == 0 or self.yard._get_occupancy_bits(row, tier - 1, bay) != 0:
                                mask[row, idx, tier] = True
        
        return torch.from_numpy(mask).to(self.device)
    
    def get_compact_state(self) -> torch.Tensor:
        """
        Get compact state representation for simple DRL models.
        
        Returns:
            1D tensor with aggregated features
        """
        features = []
        
        # Total occupancy rate
        total_positions = self.yard.n_rows * self.yard.n_bays * self.yard.n_tiers * YARD_SPLIT_FACTOR
        occupied_count = sum(1 for _ in self.yard.containers)
        features.append(occupied_count / total_positions)
        
        # Average stack height
        total_height = 0
        for row in range(self.yard.n_rows):
            for bay in range(self.yard.n_bays):
                for tier in range(self.yard.n_tiers):
                    if self.yard._get_occupancy_bits(row, tier, bay) != 0:
                        total_height += tier + 1
                        break
        
        avg_height = total_height / (self.yard.n_rows * self.yard.n_bays)
        features.append(avg_height / self.yard.n_tiers)
        
        # Accessibility rate
        accessible_count = len(self.yard.get_moveable_containers())
        features.append(accessible_count / max(1, len(self.yard.containers)))
        
        # Container type distribution (normalized counts)
        type_counts = {
            "small": 0,  # 20-26ft
            "standard": 0,  # 40ft
            "large": 0,  # 45ft
            "exclusive": 0  # Trailers/Swap bodies
        }
        
        for _, (container, _) in self.yard.containers.items():
            if container.container_type in ["20ft", "22ft", "23ft", "26ft", "TEU"]:
                type_counts["small"] += 1
            elif container.container_type in ["40ft", "FEU"]:
                type_counts["standard"] += 1
            elif container.container_type == "45ft":
                type_counts["large"] += 1
            elif container.container_type in EXCLUSIVE_TYPES:
                type_counts["exclusive"] += 1
        
        total_containers = max(1, sum(type_counts.values()))
        for count in type_counts.values():
            features.append(count / total_containers)
        
        return torch.tensor(features, dtype=torch.float32).to(self.device)