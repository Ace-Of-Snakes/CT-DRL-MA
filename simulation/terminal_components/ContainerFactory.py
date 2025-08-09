# DEPENDANCIES
from datetime import datetime, timedelta
import numpy as np
import random

# CUSTOM DEPENDANCIES
from simulation.terminal_components.Container import *

# ==================== CONTAINER FACTORY CONSTANTS ====================
# Probability distributions
HIGH_CUBE_PROBABILITY = 0.3
REEFER_PROBABILITY = 0.0066
DANGEROUS_GOODS_PROBABILITIES = {
    "TWEU": 0.0134,
    "FEU": 0.0023,
    "Swap Body": 0.0152,
    "FFEU": 0.0,
    "THEU": 0.2204,
    "Trailer": 0.0726
}

# Container type weights for random generation
CONTAINER_TYPE_WEIGHTS = {
    "FEU": 0.532,
    "Swap Body": 0.256,
    "TWEU": 0.180,
    "Trailer": 0.032,
    "THEU": 0.014,
    "FFEU": 0.011
}

class ContainerFactory:
    """Factory class for creating containers."""
    
    @staticmethod
    def create_container(container_id: str,
                        container_type: str,
                        direction: str = "Import",
                        goods_type: str = "Regular",
                        is_high_cube: bool = False,
                        arrival_date: datetime = None,
                        departure_date: datetime = None,
                        **kwargs) -> Container:
        """Create a container with specified properties."""
        # Determine special properties based on type
        is_stackable = container_type != "Trailer"
        stack_compatibility = "none" if container_type in SPECIAL_CONTAINER_TYPES else "size"
        
        # Trailers and swap bodies are typically export
        if container_type in SPECIAL_CONTAINER_TYPES:
            direction = "Export"
        
        # Special stacking for dangerous/reefer goods
        if goods_type in ONLY_SELF_STACKABLE:
            stack_compatibility = "self"
        
        return Container(
            container_id=container_id,
            direction=direction,
            container_type=container_type,
            goods_type=goods_type,
            is_high_cube=is_high_cube,
            is_stackable=is_stackable,
            stack_compatibility=stack_compatibility,
            arrival_date=arrival_date or datetime.now(),
            departure_date=departure_date,
            **kwargs
        )
    
    @staticmethod
    def create_random(container_id: str = None, config=None) -> Container:
        """Create a random container based on probability distributions."""
        container_id = container_id or f"CONT{random.randint(100000, 999999)}"
        
        # Select container type
        if config and hasattr(config, 'get_container_type_probabilities'):
            container_type, goods_type, is_high_cube = ContainerFactory._from_config(config)
        else:
            container_type, goods_type, is_high_cube = ContainerFactory._from_defaults()
        
        # Generate random stay duration
        arrival_date = datetime.now() + timedelta(days=random.randint(-5, 2))
        departure_date = arrival_date + timedelta(days=random.randint(3, 15))
        
        return ContainerFactory.create_container(
            container_id=container_id,
            container_type=container_type,
            direction=random.choice(list(VALID_DIRECTIONS)),
            goods_type=goods_type,
            is_high_cube=is_high_cube,
            arrival_date=arrival_date,
            departure_date=departure_date
        )
    
    @staticmethod
    def _from_config(config) -> tuple:
        """Generate container properties from config."""
        probs = config.get_container_type_probabilities()
        length_probs = probs["length"]
        
        # Select length type
        length_types = list(length_probs.keys())
        weights = [length_probs[lt]["probability"] for lt in length_types]
        selected_length = random.choices(length_types, weights=weights)[0]
        
        # Handle special types
        if selected_length == "trailer":
            return "Trailer", "Regular", False
        elif selected_length == "swap body":
            return "Swap Body", "Regular", False
        
        # Standard containers
        length_props = length_probs[selected_length]
        is_high_cube = random.random() < length_props.get("probability_high_cube", 0)
        
        # Determine goods type
        rand = random.random()
        reefer_prob = length_props.get("probability_reefer", 0)
        dg_prob = length_props.get("probability_dangerous_goods", 0)
        
        if rand < reefer_prob:
            goods_type = "Reefer"
        elif rand < (reefer_prob + dg_prob):
            goods_type = "Dangerous"
        else:
            goods_type = "Regular"
        
        # Map length to container type
        container_type = {"20": "TWEU", "30": "THEU", "40": "FEU"}.get(selected_length, "TWEU")
        
        return container_type, goods_type, is_high_cube
    
    @staticmethod
    def _from_defaults() -> tuple:
        """Generate container properties from default probabilities."""
        # Select container type
        types = list(CONTAINER_TYPE_WEIGHTS.keys())
        weights = list(CONTAINER_TYPE_WEIGHTS.values())
        container_type = random.choices(types, weights=weights)[0]
        
        # Special containers
        if container_type in SPECIAL_CONTAINER_TYPES:
            is_dangerous = random.random() < DANGEROUS_GOODS_PROBABILITIES.get(container_type, 0.01)
            goods_type = "Dangerous" if is_dangerous else "Regular"
            return container_type, goods_type, False
        
        # Standard containers
        is_high_cube = random.random() < HIGH_CUBE_PROBABILITY
        is_dangerous = random.random() < DANGEROUS_GOODS_PROBABILITIES.get(container_type, 0.01)
        is_reefer = random.random() < REEFER_PROBABILITY
        
        if is_dangerous:
            goods_type = "Dangerous"
        elif is_reefer:
            goods_type = "Reefer"
        else:
            goods_type = "Regular"
        
        return container_type, goods_type, is_high_cube