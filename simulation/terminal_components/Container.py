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
                              TWEU = Twenty-foot (20')
                              THEU = Thirty-foot (30')
                              FEU = Forty-foot (40')
                              FFEU = Forty-five-foot (45')
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
                 stack_compatibility="size",  # "self", "size", or "none"
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
            if is_high_cube:
                self.height = 2.89  # High cube height
            else:
                self.height = 2.59  # Standard height
        else:
            self.height = height
            
        if length is None:
            if container_type == "TWEU":
                self.length = 6.06  # meters (20 feet)
            elif container_type == "THEU":
                self.length = 9.14  # meters (30 feet)
            elif container_type == "FEU":
                self.length = 12.19  # meters (40 feet)
            elif container_type == "FFEU":
                self.length = 13.72  # meters (45 feet)
            elif container_type == "Swap Body":
                self.length = 7.45  # typical swap body length (Wechselbrücke)
            else:
                self.length = 12.19  # Default for trailers
        else:
            self.length = length
            
        if width is None:
            if container_type in ["Trailer", "Swap Body"]:
                self.width = 2.55  # Slightly wider than standard containers
            else:
                self.width = 2.44  # meters (standard width)
        else:
            self.width = width
        
        # Goods type (Regular, Reefer, Dangerous)
        valid_goods = ["Regular", "Reefer", "Dangerous"]
        if goods_type not in valid_goods:
            raise ValueError(f"Goods type must be one of {valid_goods}")
        self.goods_type = goods_type
        
        # Stackability properties
        self.is_stackable = is_stackable
        
        # Update stack compatibility options: "self" (only with identical containers),
        # "size" (only with same size containers), or "none" (not stackable)
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
            if container_type == "TWEU":
                self.weight = 20000  # 20 tonnes typical for TWEU
            elif container_type == "THEU":
                self.weight = 25000  # 25 tonnes typical for THEU
            elif container_type == "FEU":
                self.weight = 30000  # 30 tonnes typical for FEU
            elif container_type == "FFEU":
                self.weight = 32000  # 32 tonnes typical for FFEU
            elif container_type == "Trailer":
                self.weight = 15000  # Lighter than standard containers
            elif container_type == "Swap Body":
                self.weight = 12000  # Lighter than standard containers (Wechselbrücke)
        else:
            self.weight = weight
        
        # Priority (can be calculated dynamically)
        self.priority = priority
        self.update_priority()
    
    def update_priority(self):
        """Calculate container priority based on time in terminal and departure date."""
        if not self.priority:
            # Default priority calculation based on diagram specifications
            # Lower number = higher priority
            
            # Base priority factors
            priority = 100  # Default starting value
            
            # Non-stackable containers should be prioritized for removal (golden move)
            if not self.is_stackable:
                priority -= 40
                
            # Adjust priority for special container types
            if self.container_type == "Trailer":
                priority -= 30  # Highest priority for trailers (they take up most space)
            elif self.container_type == "Swap Body":
                priority -= 20  # High priority for swap bodies
                
            # Special goods types get priority
            if self.goods_type == "Dangerous":
                priority -= 15  # Dangerous goods need special handling
            elif self.goods_type == "Reefer":
                priority -= 10  # Reefers need electricity connection
                
            # Adjust for residence time
            days_in_terminal = self.days_in_terminal(datetime.now())
            if days_in_terminal > 8:
                # Approaching 10-day limit, high priority
                priority -= 25
            elif days_in_terminal > 5:
                # Medium residence time
                priority -= 10
                
            # Adjust for departure date if known
            if self.departure_date:
                days_until_departure = self.days_until_departure(datetime.now())
                if days_until_departure <= 1:
                    # Leaving very soon, highest priority
                    priority -= 50
                elif days_until_departure <= 3:
                    # Leaving soon, high priority
                    priority -= 25
            
            self.priority = max(1, priority)  # Ensure priority is at least 1
    
    def days_in_terminal(self, current_date):
        """Calculate how many days the container has been in the terminal."""
        if self.arrival_date and current_date:
            return max(0, (current_date - self.arrival_date).days)
        return 0
    
    def days_until_departure(self, current_date):
        """Calculate how many days until the container needs to leave."""
        if self.departure_date and current_date:
            return max(0, (self.departure_date - current_date).days)
        return float('inf')  # If no departure date set
    
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
        """
        Check if this container can be safely stacked on another container.
        
        Args:
            container_below: The container that would be below this one
            
        Returns:
            Boolean indicating if stacking is safe
        """
        # First check the regular stacking compatibility
        if not self.can_stack_with(container_below):
            return False
        
        # Weight check - heavier containers cannot be stacked on lighter ones
        # Use a ratio threshold - container below should not be less than 80% of the weight above
        # This means the container above should not be more than 25% heavier than the one below
        # This is a conservative threshold for safety
        weight_ratio = self.weight / container_below.weight if container_below.weight > 0 else float('inf')
        
        # If the container above is more than 25% heavier than the one below, it's unsafe
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
        """
        Sample a container weight from the KDE distribution.
        
        Args:
            config: TerminalConfig object (optional)
            
        Returns:
            Sampled weight in kg
        """
        # If config with KDE model is available, use it
        if (config and hasattr(config, 'sample_from_kde') and 
            'container_weight' in config.kde_models):
            # Sample from KDE
            weight = config.sample_from_kde('container_weight', n_samples=1, 
                                        min_val=1000, max_val=31000)[0]
            return weight
        
        # Fallback: return a random weight between 1000-31000 kg
        return random.uniform(1000, 31000)
    
    @staticmethod
    def create_tweu(container_id, direction, goods_type="Regular", is_high_cube=False, 
                  arrival_date=None, departure_date=None, weight=None, config=None):
        """Create a TWEU container (Twenty-foot Equivalent Unit)."""
        # Define stack compatibility based on goods type
        stack_compatibility = "size"  # Can only stack with same size containers
        
        # Sample weight if not provided
        if weight is None:
            weight = ContainerFactory.sample_container_weight(config)
            
        return Container(
            container_id=container_id,
            direction=direction,
            container_type="TWEU",
            goods_type=goods_type,
            is_high_cube=is_high_cube,
            is_stackable=True,
            stack_compatibility=stack_compatibility,
            arrival_date=arrival_date,
            departure_date=departure_date,
            weight=weight
        )
    
    @staticmethod
    def create_theu(container_id, direction, goods_type="Regular", is_high_cube=False,
                   arrival_date=None, departure_date=None, weight=None, config=None):
        """Create a THEU container (Thirty-foot Equivalent Unit)."""
        stack_compatibility = "size"  # Can only stack with same size containers
        
        # Sample weight if not provided
        if weight is None:
            weight = ContainerFactory.sample_container_weight(config)
            
        return Container(
            container_id=container_id,
            direction=direction,
            container_type="THEU",
            goods_type=goods_type,
            is_high_cube=is_high_cube,
            is_stackable=True,
            stack_compatibility=stack_compatibility,
            arrival_date=arrival_date,
            departure_date=departure_date,
            weight=weight
        )
    
    @staticmethod
    def create_feu(container_id, direction, goods_type="Regular", is_high_cube=False,
                  arrival_date=None, departure_date=None, weight=None, config=None):
        """Create a FEU container (Forty-foot Equivalent Unit)."""
        stack_compatibility = "size"  # Can only stack with same size containers
        
        # Sample weight if not provided
        if weight is None:
            weight = ContainerFactory.sample_container_weight(config)
            
        return Container(
            container_id=container_id,
            direction=direction,
            container_type="FEU",
            goods_type=goods_type,
            is_high_cube=is_high_cube,
            is_stackable=True,
            stack_compatibility=stack_compatibility,
            arrival_date=arrival_date,
            departure_date=departure_date,
            weight=weight
        )
    
    @staticmethod
    def create_ffeu(container_id, direction, goods_type="Regular", is_high_cube=False,
                   arrival_date=None, departure_date=None, weight=None, config=None):
        """Create a FFEU container (Forty-five-foot Equivalent Unit)."""
        stack_compatibility = "size"  # Can only stack with same size containers
        
        # Sample weight if not provided
        if weight is None:
            weight = ContainerFactory.sample_container_weight(config)
            
        return Container(
            container_id=container_id,
            direction=direction,
            container_type="FFEU",
            goods_type=goods_type,
            is_high_cube=is_high_cube,
            is_stackable=True,
            stack_compatibility=stack_compatibility,
            arrival_date=arrival_date,
            departure_date=departure_date,
            weight=weight
        )
    
    @staticmethod
    def create_trailer(container_id, goods_type="Regular", arrival_date=None, 
                      departure_date=None, config=None):
        """Create a Trailer container (non-stackable, Row E storage only)."""
        # Sample weight if needed
        weight = ContainerFactory.sample_container_weight(config)
        
        return Container(
            container_id=container_id,
            direction="Export",  # Trailers are typically for export
            container_type="Trailer",
            goods_type=goods_type,
            is_stackable=False,
            stack_compatibility="none",
            arrival_date=arrival_date,
            departure_date=departure_date,
            height=4.0,  # Approximate height
            width=2.55,  # Slightly wider than standard containers
            weight=weight
        )
    
    @staticmethod
    def create_swap_body(container_id, goods_type="Regular", arrival_date=None, 
                        departure_date=None, config=None):
        """Create a Swap Body container (stackable only as single unit)."""
        # Sample weight if needed
        weight = ContainerFactory.sample_container_weight(config)
        
        return Container(
            container_id=container_id,
            direction="Export",  # Swap bodies are typically for export
            container_type="Swap Body",
            goods_type=goods_type,
            is_stackable=True,  # Can be stacked but only as a single unit
            stack_compatibility="none",
            arrival_date=arrival_date,
            departure_date=departure_date,
            height=2.67,  # Typical height
            length=7.45,  # Typical length
            width=2.55,   # Typical width
            weight=weight
        )
    
    @staticmethod
    def create_reefer(container_id, container_type="TWEU", direction="Import",
                     arrival_date=None, departure_date=None, weight=None, config=None):
        """Create a Reefer container (temperature-controlled, needs electricity)."""
        # Validate container type
        if container_type not in ["TWEU", "THEU", "FEU", "FFEU"]:
            raise ValueError("Container type for reefer must be TWEU, THEU, FEU, or FFEU")
        
        # Sample weight if not provided
        if weight is None:
            weight = ContainerFactory.sample_container_weight(config)
            
        return Container(
            container_id=container_id,
            direction=direction,
            container_type=container_type,
            goods_type="Reefer",
            is_stackable=True,
            stack_compatibility="self",  # Reefers only stack with other reefers or same size containers
            arrival_date=arrival_date,
            departure_date=departure_date,
            weight=weight
        )
    
    @staticmethod
    def create_dangerous(container_id, container_type="TWEU", direction="Import",
                        arrival_date=None, departure_date=None, weight=None, config=None):
        """Create a Dangerous Goods container (needs special storage bays)."""
        # Validate container type
        if container_type not in ["TWEU", "THEU", "FEU", "FFEU", "Trailer", "Swap Body"]:
            raise ValueError("Container type must be a valid type")
        
        # Sample weight if not provided
        if weight is None:
            weight = ContainerFactory.sample_container_weight(config)
            
        # In our data, trailers have higher percentage of dangerous goods
        if container_type == "Trailer":
            is_stackable = False
            stack_compatibility = "none"
        else:
            is_stackable = True
            stack_compatibility = "self"  # Dangerous goods only stack with same type
            
        return Container(
            container_id=container_id,
            direction=direction,
            container_type=container_type,
            goods_type="Dangerous",
            is_stackable=is_stackable,
            stack_compatibility=stack_compatibility,
            arrival_date=arrival_date,
            departure_date=departure_date,
            weight=weight
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
        
        # If no config provided, use legacy implementation
        if config is None:
            return ContainerFactory._create_random_default(container_id, direction)
        
        # Get container probabilities from config
        probs = config.get_container_type_probabilities()
        
        # Select container length/type based on probabilities
        length_probs = probs["length"]
        length_types = list(length_probs.keys())
        length_weights = [length_probs[lt]["probability"] for lt in length_types]
        
        selected_length = random.choices(length_types, weights=length_weights)[0]
        
        # Sample weight
        weight = ContainerFactory.sample_container_weight(config)
        
        # Handle trailer and swap body separately
        if selected_length == "trailer":
            container = ContainerFactory.create_trailer(
                container_id, 
                "Regular", 
                datetime.now(), 
                None,
                config
            )
            return container
        elif selected_length == "swap body":
            container = ContainerFactory.create_swap_body(
                container_id, 
                "Regular", 
                datetime.now(), 
                None,
                config
            )
            return container
        
        # For standard containers, determine high cube, reefer, and dangerous goods status
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
        
        # Create container based on determined attributes
        if goods_type == "Reefer":
            return ContainerFactory.create_reefer(
                container_id, 
                container_type, 
                direction,
                datetime.now(), 
                None, 
                weight,
                config
            )
        elif goods_type == "Dangerous":
            return ContainerFactory.create_dangerous(
                container_id, 
                container_type, 
                direction,
                datetime.now(), 
                None, 
                weight,
                config
            )
        else:
            if container_type == "TWEU":
                return ContainerFactory.create_tweu(
                    container_id, 
                    direction, 
                    "Regular", 
                    is_high_cube,
                    datetime.now(), 
                    None, 
                    weight,
                    config
                )
            elif container_type == "THEU":
                return ContainerFactory.create_theu(
                    container_id, 
                    direction, 
                    "Regular", 
                    is_high_cube,
                    datetime.now(), 
                    None, 
                    weight,
                    config
                )
            elif container_type == "FEU":
                return ContainerFactory.create_feu(
                    container_id, 
                    direction, 
                    "Regular", 
                    is_high_cube,
                    datetime.now(), 
                    None, 
                    weight,
                    config
                )
            else:  # Default fallback
                return ContainerFactory.create_tweu(
                    container_id, 
                    direction, 
                    "Regular", 
                    is_high_cube,
                    datetime.now(), 
                    None, 
                    weight,
                    config
                )

    @staticmethod
    def _create_random_default(container_id, direction):
        """Legacy method for random container creation without config."""
        # Container type distribution based on the provided data
        container_types = ["FEU", "Swap Body", "TWEU", "Trailer", "THEU", "FFEU"]
        container_type_weights = [0.532, 0.256, 0.180, 0.032, 0.014, 0.011]
        container_type = random.choices(container_types, weights=container_type_weights)[0]
        
        # Dangerous goods probability depends on container type
        dg_probabilities = {
            "TWEU": 0.0134,    # 20' Container
            "FEU": 0.0023,     # 40' Container
            "Swap Body": 0.0152, # Wechselbrücke
            "FFEU": 0.0,       # 45' Container
            "THEU": 0.2204,    # 30' Container
            "Trailer": 0.0726  # Trailer
        }
        
        # Determine if this is a dangerous goods container
        is_dangerous = random.random() < dg_probabilities[container_type]
        
        # Reefer probability is 0.66%, but only for standard containers, not trailers or swap bodies
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
        
        # Sample weight using the new method, but without config
        weight = ContainerFactory.sample_container_weight(None)
        
        # Use the appropriate factory method
        if goods_type == "Dangerous":
            return ContainerFactory.create_dangerous(
                container_id, 
                container_type, 
                direction,
                arrival_date, 
                departure_date, 
                weight
            )
        elif goods_type == "Reefer":
            return ContainerFactory.create_reefer(
                container_id, 
                container_type, 
                direction, 
                arrival_date, 
                departure_date, 
                weight
            )
        elif container_type == "Trailer":
            trailer = ContainerFactory.create_trailer(
                container_id, 
                "Regular", 
                arrival_date, 
                departure_date
            )
            # Set weight since create_trailer doesn't have a weight parameter in the original version
            trailer.weight = weight
            return trailer
        elif container_type == "Swap Body":
            swap_body = ContainerFactory.create_swap_body(
                container_id, 
                "Regular", 
                arrival_date, 
                departure_date
            )
            # Set weight since create_swap_body doesn't have a weight parameter in the original version
            swap_body.weight = weight
            return swap_body
        elif container_type == "TWEU":
            return ContainerFactory.create_tweu(
                container_id, 
                direction, 
                "Regular", 
                is_high_cube,
                arrival_date, 
                departure_date, 
                weight
            )
        elif container_type == "THEU":
            return ContainerFactory.create_theu(
                container_id, 
                direction, 
                "Regular", 
                is_high_cube,
                arrival_date, 
                departure_date, 
                weight
            )
        elif container_type == "FEU":
            return ContainerFactory.create_feu(
                container_id, 
                direction, 
                "Regular", 
                is_high_cube,
                arrival_date, 
                departure_date, 
                weight
            )
        else:  # FFEU
            return ContainerFactory.create_ffeu(
                container_id, 
                direction, 
                "Regular", 
                is_high_cube,
                arrival_date, 
                departure_date, 
                weight
            )