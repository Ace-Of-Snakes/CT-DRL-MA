from datetime import datetime, timedelta
import random
import numpy as np

class Container:
    """
    Container class representing shipping containers in a terminal.
    
    Attributes:
        container_id (str): Unique identifier for the container
        direction (str): 'Import' (train to customer) or 'Export' (truck to customer)
        container_type (str): Physical type - 'TWEU', 'THEU', 'FEU', 'FFEU', 'Trailer', or 'Swap Body'
        goods_type (str): Contents type - 'Regular', 'Reefer', or 'Dangerous'
        is_high_cube (bool): Whether the container is a high cube variant (taller)
        is_stackable (bool): Whether the container can be stacked
        stack_compatibility (str): How it stacks with others - 'self', 'size', or 'none'
        arrival_date (datetime): When the container arrived at the terminal
        departure_date (datetime): When the container is scheduled to leave
        weight (float): Weight of the container in kg
        height (float): Height in meters
        length (float): Length in meters
        width (float): Width in meters
        priority (int): Priority for handling (lower is higher priority)
    """
    
    def __init__(self, 
                 container_id, 
                 direction, 
                 container_type, 
                 goods_type="Regular",
                 is_high_cube=False,
                 is_stackable=True,
                 stack_compatibility="size",
                 arrival_date=None,
                 departure_date=None,
                 weight=None,
                 height=None,
                 length=None,
                 width=None,
                 priority=None):
        
        # Basic identification
        self.container_id = container_id
        
        # Direction (Import or Export)
        if direction not in ["Import", "Export"]:
            raise ValueError("Direction must be either 'Import' or 'Export'")
        self.direction = direction
        
        # Container physical type 
        valid_types = ["TWEU", "THEU", "FEU", "FFEU", "Trailer", "Swap Body"]
        if container_type not in valid_types:
            raise ValueError(f"Container type must be one of {valid_types}")
        self.container_type = container_type
        self.is_high_cube = is_high_cube
        
        # Set dimensions based on container type if not explicitly provided
        if height is None:
            self.height = 2.89 if is_high_cube else 2.59
        else:
            self.height = height
            
        if length is None:
            length_map = {
                "TWEU": 6.06,      # 20 feet
                "THEU": 9.14,      # 30 feet
                "FEU": 12.19,      # 40 feet
                "FFEU": 13.72,     # 45 feet
                "Swap Body": 7.45, # typical swap body length
                "Trailer": 12.19   # default for trailers
            }
            self.length = length_map.get(container_type, 12.19)
        else:
            self.length = length
            
        if width is None:
            self.width = 2.55 if container_type in ["Trailer", "Swap Body"] else 2.44
        else:
            self.width = width
        
        # Goods type (Regular, Reefer, Dangerous)
        valid_goods = ["Regular", "Reefer", "Dangerous"]
        if goods_type not in valid_goods:
            raise ValueError(f"Goods type must be one of {valid_goods}")
        self.goods_type = goods_type
        
        # Stackability properties
        self.is_stackable = is_stackable
        
        # Update stack compatibility options
        valid_stack_compatibility = ["self", "size", "none"]
        if stack_compatibility not in valid_stack_compatibility:
            raise ValueError(f"Stack compatibility must be one of {valid_stack_compatibility}")
        self.stack_compatibility = stack_compatibility
        
        # Special case: reefers and dangerous goods have more restrictive stacking
        if goods_type in ["Reefer", "Dangerous"]:
            self.stack_compatibility = "self"
        
        # Timing and scheduling
        self.arrival_date = arrival_date or datetime.now()
        self.departure_date = departure_date
        
        # Weight for stacking rules
        if weight is None:
            # Default weights based on container type
            weight_map = {
                "TWEU": 20000,    # 20 tonnes typical for TWEU
                "THEU": 25000,    # 25 tonnes typical for THEU
                "FEU": 30000,     # 30 tonnes typical for FEU
                "FFEU": 32000,    # 32 tonnes typical for FFEU
                "Trailer": 15000, # Lighter than standard containers
                "Swap Body": 12000 # Lighter than standard containers
            }
            self.weight = weight_map.get(container_type, 20000)
        else:
            self.weight = weight
        
        # Priority (can be calculated dynamically)
        self.priority = priority
        self.update_priority()
    
    def update_priority(self):
        """Calculate container priority based on time in terminal and departure date."""
        if not self.priority:
            # Base priority starts at 100
            priority = 100
            
            # Non-stackable containers get priority
            if not self.is_stackable:
                priority -= 40
                
            # Adjust by container type
            type_priority = {
                "Trailer": 30,
                "Swap Body": 20
            }
            priority -= type_priority.get(self.container_type, 0)
                
            # Adjust by goods type
            goods_priority = {
                "Dangerous": 15,
                "Reefer": 10
            }
            priority -= goods_priority.get(self.goods_type, 0)
                
            # Adjust for residence time
            days_in_terminal = self.days_in_terminal(datetime.now())
            if days_in_terminal > 8:
                priority -= 25
            elif days_in_terminal > 5:
                priority -= 10
                
            # Adjust for departure date
            if self.departure_date:
                days_until_departure = self.days_until_departure(datetime.now())
                if days_until_departure <= 1:
                    priority -= 50
                elif days_until_departure <= 3:
                    priority -= 25
            
            self.priority = max(1, priority)
    
    def days_in_terminal(self, current_date):
        """Calculate how many days the container has been in the terminal."""
        if self.arrival_date and current_date:
            return max(0, (current_date - self.arrival_date).days)
        return 0
    
    def days_until_departure(self, current_date):
        """Calculate how many days until the container needs to leave."""
        if self.departure_date and current_date:
            return max(0, (self.departure_date - current_date).days)
        return float('inf')
    
    def is_overdue(self, current_date):
        """Check if the container has been in the terminal for more than 10 days."""
        return self.days_in_terminal(current_date) > 10
    
    def can_stack_with(self, other_container):
        """
        Determine if this container can be stacked on top of another container.
        
        Args:
            other_container: The container below this one
            
        Returns:
            bool: True if this container can be stacked on the other container
        """
        # If either container is not stackable, they can't stack
        if not self.is_stackable or not other_container.is_stackable:
            return False
        
        # Check stack compatibility
        if self.stack_compatibility == "none" or other_container.stack_compatibility == "none":
            return False
            
        # Dangerous goods and Reefers have special stacking rules
        if self.goods_type in ["Dangerous", "Reefer"]:
            # Must stack on same type containers or same size regular containers
            if other_container.goods_type != self.goods_type:
                # Can only stack on same size regular containers
                if other_container.goods_type != "Regular" or other_container.container_type != self.container_type:
                    return False
        
        # Self compatibility: must be same type and goods
        if self.stack_compatibility == "self" or other_container.stack_compatibility == "self":
            if self.container_type != other_container.container_type or self.goods_type != other_container.goods_type:
                return False
        
        # Size compatibility: must be same size (but can be different goods)
        if self.stack_compatibility == "size":
            # Check if container types have the same size
            if self.container_type != other_container.container_type:
                return False
        
        # Check weight rules (heavier containers should be on bottom)
        if self.weight > other_container.weight:
            return False
        
        # Check departure timing (containers leaving earlier should be on top)
        if self.departure_date and other_container.departure_date:
            if self.departure_date > other_container.departure_date:
                return False
        
        # If all checks pass, containers can stack
        return True
    
    def can_be_stacked_on(self, container_below):
        """Check if this container can be safely stacked on another container."""
        # First check the regular stacking compatibility
        if not self.can_stack_with(container_below):
            return False
        
        # Safety weight check - container above should not be more than 25% heavier
        weight_ratio = self.weight / container_below.weight if container_below.weight > 0 else float('inf')
        if weight_ratio > 1.25:
            return False
        
        return True
        
    def __str__(self):
        return f"Container {self.container_id} ({self.container_type}): {self.direction}, {self.goods_type}"
    
    def __repr__(self):
        return f"Container(id={self.container_id}, type={self.container_type}, goods={self.goods_type})"


class ContainerFactory:
    """Factory class to easily create different types of containers."""
    
    @staticmethod
    def sample_container_weight(config=None):
        """Sample a container weight from the KDE distribution."""
        # If config with KDE model is available, use it
        if (config and hasattr(config, 'sample_from_kde') and 
            'container_weight' in config.kde_models):
            # Sample from KDE
            weight = config.sample_from_kde('container_weight', n_samples=1, 
                                        min_val=1000, max_val=31000)[0]
            return weight
        
        # Fallback: return a random weight
        return random.uniform(1000, 31000)
    
    @staticmethod
    def create_container(container_id, container_type, direction="Import", 
                      goods_type="Regular", is_high_cube=False,
                      arrival_date=None, departure_date=None, 
                      weight=None, config=None, **kwargs):
        """
        Unified method to create a container of any type.
        
        Args:
            container_id: Container identifier
            container_type: Type of container ('TWEU', 'THEU', 'FEU', 'FFEU', 'Trailer', 'Swap Body')
            direction: 'Import' or 'Export'
            goods_type: 'Regular', 'Reefer', or 'Dangerous'
            is_high_cube: Whether it's a high cube container
            arrival_date: When the container arrived
            departure_date: When the container will depart
            weight: Container weight in kg
            config: Configuration object for sampling weights
            **kwargs: Additional arguments to pass to Container constructor
            
        Returns:
            Container object
        """
        # Sample weight if not provided
        if weight is None:
            weight = ContainerFactory.sample_container_weight(config)
        
        # Special cases based on container type
        is_stackable = True
        stack_compatibility = "size"  # Default
        
        # Special case for trailers and swap bodies
        if container_type == "Trailer":
            is_stackable = False
            stack_compatibility = "none"
            direction = "Export"  # Trailers are typically for export
        elif container_type == "Swap Body":
            is_stackable = True  # Can be stacked but only as a single unit
            stack_compatibility = "none"
            direction = "Export"  # Swap bodies are typically for export
        
        # Special case for reefer and dangerous goods
        if goods_type in ["Reefer", "Dangerous"]:
            stack_compatibility = "self"  # Only stack with same goods type
        
        # Create the container
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
            weight=weight,
            **kwargs
        )
    
    @staticmethod
    def create_random(container_id=None, config=None):
        """
        Create a random container based on probability distributions.
        
        Args:
            container_id: Optional container ID
            config: TerminalConfig object with probability distributions
            
        Returns:
            A Container object
        """
        # Generate random container ID if not provided
        container_id = container_id or f"CONT{random.randint(100000, 999999)}"
        
        # Random container direction
        direction = random.choice(["Import", "Export"])
        
        # Get container probabilities from config
        if config:
            # Use configuration probabilities
            probs = config.get_container_type_probabilities()
            
            # Select container length/type based on probabilities
            length_probs = probs["length"]
            length_types = list(length_probs.keys())
            length_weights = [length_probs[lt]["probability"] for lt in length_types]
            
            selected_length = random.choices(length_types, weights=length_weights)[0]
            
            # Sample weight
            weight = ContainerFactory.sample_container_weight(config)
            
            # Handle trailer and swap body separately
            if selected_length in ["trailer", "swap body"]:
                return ContainerFactory.create_container(
                    container_id=container_id,
                    container_type="Trailer" if selected_length == "trailer" else "Swap Body",
                    goods_type="Regular", 
                    arrival_date=datetime.now(),
                    weight=weight,
                    config=config
                )
            
            # For standard containers, get properties from config
            length_type_props = length_probs[selected_length]
            
            # Check for high cube
            is_high_cube = random.random() < length_type_props.get("probability_high_cube", 0)
            
            # Determine goods type (reefer, dangerous, or regular)
            goods_type_rand = random.random()
            reefer_prob = length_type_props.get("probability_reefer", 0)
            dg_prob = length_type_props.get("probability_dangerous_goods", 0)
            
            if goods_type_rand < reefer_prob:
                goods_type = "Reefer"
            elif goods_type_rand < (reefer_prob + dg_prob):
                goods_type = "Dangerous"
            else:
                goods_type = "Regular"
                
            # Map length to container type
            length_to_type = {
                "20": "TWEU",
                "30": "THEU",
                "40": "FEU"
            }
            
            container_type = length_to_type.get(selected_length, "TWEU")
            
        else:
            # Fallback to legacy probabilities
            container_types = ["FEU", "Swap Body", "TWEU", "Trailer", "THEU", "FFEU"]
            container_type_weights = [0.532, 0.256, 0.180, 0.032, 0.014, 0.011]
            container_type = random.choices(container_types, weights=container_type_weights)[0]
            
            # Dangerous goods probability depends on container type
            dg_probabilities = {
                "TWEU": 0.0134, "FEU": 0.0023, "Swap Body": 0.0152,
                "FFEU": 0.0, "THEU": 0.2204, "Trailer": 0.0726
            }
            
            # Determine if this is a dangerous goods container
            is_dangerous = random.random() < dg_probabilities[container_type]
            
            # Reefer probability (only for standard containers)
            is_reefer = False
            if container_type in ["TWEU", "THEU", "FEU", "FFEU"]:
                is_reefer = random.random() < 0.0066
            
            # Determine goods type
            if is_dangerous:
                goods_type = "Dangerous"
            elif is_reefer:
                goods_type = "Reefer"
            else:
                goods_type = "Regular"
            
            # 30% chance of high cube for standard containers
            is_high_cube = False
            if container_type in ["TWEU", "THEU", "FEU", "FFEU"]:
                is_high_cube = random.random() < 0.3
            
            # Random dates
            today = datetime.now()
            arrival_offset = random.randint(-5, 2)  # -5 to 2 days from today
            arrival_date = today + timedelta(days=arrival_offset)
            
            stay_duration = random.randint(3, 15)  # 3 to 15 days stay
            departure_date = arrival_date + timedelta(days=stay_duration)
            
            # Sample weight
            weight = ContainerFactory.sample_container_weight(None)
        
        # Create the container with all determined properties
        return ContainerFactory.create_container(
            container_id=container_id,
            container_type=container_type,
            direction=direction,
            goods_type=goods_type,
            is_high_cube=is_high_cube,
            arrival_date=arrival_date if 'arrival_date' in locals() else datetime.now(),
            departure_date=departure_date if 'departure_date' in locals() else None,
            weight=weight
        )