import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import time

class ContainerTerminal:
    def __init__(self, 
                layout_order=['rails', 'parking', 'driving_lane', 'yard_storage'],
                num_railtracks=3,
                num_railslots_per_track=10,
                num_storage_rows=6,
                # Ratio parameters
                parking_to_railslot_ratio=1.0,  # Number of parking spots per rail track length
                storage_to_railslot_ratio=2.0,  # Number of storage slots per rail slot length
                # Dimension parameters
                rail_slot_length=20.0,  # in meters
                track_width=3.0,        # in meters
                space_between_tracks=1.5, # in meters
                space_rails_to_parking=5.0, # in meters
                space_driving_to_storage=2.0, # in meters
                parking_width=4.0,     # in meters
                driving_lane_width=8.0, # in meters
                storage_slot_width=10.0 # in meters
                ):
        # Validate inputs
        self.validate_inputs(layout_order, num_railtracks, num_railslots_per_track, num_storage_rows,
                           parking_to_railslot_ratio, storage_to_railslot_ratio,
                           rail_slot_length, track_width, space_between_tracks, space_rails_to_parking,
                           space_driving_to_storage, parking_width, driving_lane_width, storage_slot_width)
        
        self.layout_order = layout_order
        self.num_railtracks = num_railtracks
        self.num_railslots_per_track = num_railslots_per_track
        self.num_storage_rows = num_storage_rows
        
        # Store ratio parameters
        self.parking_to_railslot_ratio = parking_to_railslot_ratio
        self.storage_to_railslot_ratio = storage_to_railslot_ratio
        
        # Derived metrics based on the rules and ratios
        self.num_parking_spots = int(self.num_railslots_per_track * self.parking_to_railslot_ratio)
        self.num_storage_slots_per_row = int(self.num_railslots_per_track * self.storage_to_railslot_ratio)
        
        # Dimensions
        self.rail_slot_length = rail_slot_length
        self.track_width = track_width
        self.space_between_tracks = space_between_tracks
        self.space_rails_to_parking = space_rails_to_parking
        self.space_driving_to_storage = space_driving_to_storage
        self.parking_width = parking_width
        self.driving_lane_width = driving_lane_width
        self.storage_slot_width = storage_slot_width
        
        # Calculate adjusted dimensions based on ratios
        self.parking_slot_length = self.rail_slot_length / self.parking_to_railslot_ratio
        self.storage_slot_length = self.rail_slot_length / self.storage_to_railslot_ratio
        
        # Naming schemes
        self.track_names = [f'T{i+1}' for i in range(self.num_railtracks)]
        self.storage_row_names = [chr(65 + i) for i in range(self.num_storage_rows)]  # A, B, C, ...
        
        # Generate all object names and positions
        self.generate_objects_and_positions()
        
        # Calculate distance matrix
        self.calculate_distance_matrix()
    
    def validate_inputs(self, layout_order, num_railtracks, num_railslots_per_track, num_storage_rows,
                      parking_to_railslot_ratio, storage_to_railslot_ratio,
                      rail_slot_length, track_width, space_between_tracks, space_rails_to_parking,
                      space_driving_to_storage, parking_width, driving_lane_width, storage_slot_width):
        """Validate the input parameters."""
        # Check if all required sections are in the layout
        required_sections = ['rails', 'parking', 'driving_lane', 'yard_storage']
        for section in required_sections:
            assert section in layout_order, f"Required section '{section}' missing from layout_order"
        
        # Check if parking and driving lane are adjacent in the layout
        parking_idx = layout_order.index('parking')
        driving_idx = layout_order.index('driving_lane')
        assert abs(parking_idx - driving_idx) == 1, "Parking and driving lane must be adjacent"
        
        # Check numeric parameters
        assert num_railtracks > 0, "Number of railtracks must be positive"
        assert num_railslots_per_track > 0, "Number of rail slots per track must be positive"
        assert num_storage_rows > 0, "Number of storage rows must be positive"
        
        # Check ratio parameters
        assert parking_to_railslot_ratio > 0, "Parking to rail slot ratio must be positive"
        assert storage_to_railslot_ratio > 0, "Storage to rail slot ratio must be positive"
        
        # Check dimensions
        assert rail_slot_length > 0, "Rail slot length must be positive"
        assert track_width > 0, "Track width must be positive"
        assert space_between_tracks >= 0, "Space between tracks must be non-negative"
        assert space_rails_to_parking >= 0, "Space between rails and parking must be non-negative"
        assert space_driving_to_storage >= 0, "Space between driving lane and storage must be non-negative"
        assert parking_width > 0, "Parking width must be positive"
        assert driving_lane_width > 0, "Driving lane width must be positive"
        assert storage_slot_width > 0, "Storage slot width must be positive"
        
        # Check if we have enough letters for storage rows
        assert num_storage_rows <= 26, "Number of storage rows must be at most 26 (A-Z)"
    
    def generate_objects_and_positions(self):
        """Generate all objects and their positions in the terminal."""
        self.objects = {}
        self.positions = {}
        
        # Calculate total width of each section
        rails_width = self.num_railtracks * self.track_width + (self.num_railtracks - 1) * self.space_between_tracks
        parking_width = self.parking_width
        driving_width = self.driving_lane_width
        storage_width = self.num_storage_rows * self.storage_slot_width
        
        # Calculate starting positions of each section (y-coordinate)
        section_widths = {
            'rails': rails_width,
            'parking': parking_width,
            'driving_lane': driving_width,
            'yard_storage': storage_width
        }
        
        # Calculate section positions dynamically based on layout order
        current_pos = 0
        section_positions = {}
        
        for section in self.layout_order:
            section_positions[section] = current_pos
            current_pos += section_widths[section]
            
            # Add spacing between sections
            next_idx = self.layout_order.index(section) + 1
            if next_idx < len(self.layout_order):
                next_section = self.layout_order[next_idx]
                if section == 'rails' and next_section == 'parking':
                    current_pos += self.space_rails_to_parking
                elif section == 'driving_lane' and next_section == 'yard_storage':
                    current_pos += self.space_driving_to_storage
                elif section == 'parking' and next_section == 'driving_lane':
                    # No space between parking and driving lane as per requirements
                    pass
        
        # Generate rail track slots - starting from bottom to top
        for i, track_name in enumerate(self.track_names):
            # Calculate y-position for this track (from bottom up)
            track_y = section_positions['rails'] + i * (self.track_width + self.space_between_tracks) + self.track_width / 2
            
            for j in range(self.num_railslots_per_track):
                slot_name = f"{track_name.lower()}_{j+1}"  # t1_1, t2_3, etc.
                slot_x = j * self.rail_slot_length + self.rail_slot_length / 2
                self.objects[slot_name] = {'type': 'rail_slot', 'track': track_name, 'index': j+1}
                self.positions[slot_name] = (slot_x, track_y)
        
        # Generate parking spots
        parking_y = section_positions['parking'] + self.parking_width / 2
        for j in range(self.num_parking_spots):
            spot_name = f"p_{j+1}"  # p_1, p_2, etc.
            # Calculate position based on ratio
            if self.parking_to_railslot_ratio == 1.0:
                # Simple case: one parking spot per rail slot
                spot_x = j * self.rail_slot_length + self.rail_slot_length / 2
            else:
                # Position adjusted by ratio
                spot_x = j * (self.rail_slot_length * self.num_railslots_per_track / self.num_parking_spots) + \
                        (self.rail_slot_length * self.num_railslots_per_track / self.num_parking_spots) / 2
            
            self.objects[spot_name] = {'type': 'parking_spot', 'index': j+1}
            self.positions[spot_name] = (spot_x, parking_y)
        
        # Generate yard storage slots - starting from bottom to top (A is lowest)
        for i, row_name in enumerate(self.storage_row_names):
            row_y = section_positions['yard_storage'] + i * self.storage_slot_width + self.storage_slot_width / 2
            for j in range(self.num_storage_slots_per_row):
                slot_name = f"{row_name}{j+1}"  # A1, B2, etc.
                # Position based on storage-to-railslot ratio
                if self.storage_to_railslot_ratio == 2.0:
                    # Two storage slots per rail slot length (default ratio)
                    slot_x = j * (self.rail_slot_length / 2) + (self.rail_slot_length / 4)
                else:
                    # Position adjusted by ratio
                    total_rail_length = self.num_railslots_per_track * self.rail_slot_length
                    storage_slot_spacing = total_rail_length / self.num_storage_slots_per_row
                    slot_x = j * storage_slot_spacing + storage_slot_spacing / 2
                
                self.objects[slot_name] = {'type': 'storage_slot', 'row': row_name, 'index': j+1}
                self.positions[slot_name] = (slot_x, row_y)
    
    def calculate_distance_matrix(self):
        """Calculate the distance matrix between all objects efficiently using vectorization."""
        start_time = time.time()
        print("Calculating distance matrix...")
        
        object_names = list(self.objects.keys())
        n_objects = len(object_names)
        self.distance_matrix = np.zeros((n_objects, n_objects))
        self.object_to_idx = {name: i for i, name in enumerate(object_names)}
        
        # Pre-compute positions as a numpy array for vectorized operations
        positions_array = np.array([self.positions[name] for name in object_names])
        
        # Exploiting symmetry for distance calculation
        for i in tqdm(range(n_objects), desc="Processing objects"):
            # Calculate distances to all other objects at once
            # Vectorized distance calculation using broadcasting
            pos_i = positions_array[i]
            distances = np.sqrt(np.sum((positions_array - pos_i)**2, axis=1))
            
            # Store in distance matrix (and its symmetric counterpart)
            self.distance_matrix[i, :] = distances
        
        end_time = time.time()
        print(f"Distance matrix calculation completed in {end_time - start_time:.2f} seconds")
    
    def get_distance(self, obj1, obj2):
        """Get the distance between two objects."""
        if obj1 not in self.object_to_idx or obj2 not in self.object_to_idx:
            raise ValueError(f"Object not found: {obj1 if obj1 not in self.object_to_idx else obj2}")
            
        idx1 = self.object_to_idx[obj1]
        idx2 = self.object_to_idx[obj2]
        return self.distance_matrix[idx1, idx2]
    
    def visualize(self, figsize=(15, 10), show_labels=True):
        """Visualize the container terminal layout."""
        fig, ax = plt.subplots(figsize=figsize)
        
        # Calculate section positions for proper labeling
        section_positions = {}
        section_heights = {}
        current_y = 0
        
        # First, determine the y-positions and heights of each section
        for section in self.layout_order:
            section_height = self.get_section_width(section)
            section_positions[section] = current_y
            section_heights[section] = section_height
            current_y += section_height
            
            # Add spacing between sections
            next_idx = self.layout_order.index(section) + 1
            if next_idx < len(self.layout_order):
                next_section = self.layout_order[next_idx]
                if section == 'rails' and next_section == 'parking':
                    current_y += self.space_rails_to_parking
                elif section == 'driving_lane' and next_section == 'yard_storage':
                    current_y += self.space_driving_to_storage
        
        # Plot rail slots
        rail_slots = {k: v for k, v in self.objects.items() if v['type'] == 'rail_slot'}
        for slot_name, slot_info in rail_slots.items():
            x, y = self.positions[slot_name]
            track_idx = int(slot_info['track'][1:]) - 1  # Extract track number (T1 -> 0, T2 -> 1)
            color = plt.cm.tab10(track_idx % 10)
            rect = plt.Rectangle((x - self.rail_slot_length/2, y - self.track_width/2), 
                              self.rail_slot_length, self.track_width, 
                              color=color, alpha=0.5)
            ax.add_patch(rect)
            if show_labels:
                ax.text(x, y, slot_name, ha='center', va='center', fontsize=6)
        
        # Plot parking spots
        parking_spots = {k: v for k, v in self.objects.items() if v['type'] == 'parking_spot'}
        for spot_name, spot_info in parking_spots.items():
            x, y = self.positions[spot_name]
            rect = plt.Rectangle((x - self.rail_slot_length/2, y - self.parking_width/2), 
                              self.rail_slot_length, self.parking_width, 
                              color='lightgray', alpha=0.5)
            ax.add_patch(rect)
            if show_labels:
                ax.text(x, y, spot_name, ha='center', va='center', fontsize=6)
        
        # Plot storage slots
        storage_slots = {k: v for k, v in self.objects.items() if v['type'] == 'storage_slot'}
        for slot_name, slot_info in storage_slots.items():
            x, y = self.positions[slot_name]
            row_idx = ord(slot_info['row']) - 65  # Convert A->0, B->1, etc.
            color = plt.cm.Pastel1(row_idx % 9)
            rect = plt.Rectangle((x - self.rail_slot_length/4, y - self.storage_slot_width/2), 
                              self.rail_slot_length/2, self.storage_slot_width, 
                              color=color, alpha=0.5)
            ax.add_patch(rect)
            if show_labels:
                ax.text(x, y, slot_name, ha='center', va='center', fontsize=6)
        
        # Add driving lane
        driving_y_start = section_positions['driving_lane']
        max_x = max(pos[0] for pos in self.positions.values()) + self.rail_slot_length
        rect = plt.Rectangle((0, driving_y_start), 
                          max_x, self.driving_lane_width, 
                          color='darkgray', alpha=0.3)
        ax.add_patch(rect)
        ax.text(max_x / 2, driving_y_start + self.driving_lane_width / 2, 
               'Driving Lane', ha='center', va='center', fontsize=12)
        
        # Set axis limits
        max_y = max(pos[1] for pos in self.positions.values()) + self.storage_slot_width
        ax.set_xlim(-20, max_x + 20)
        ax.set_ylim(-10, max_y + 10)
        
        # Add section labels with proper positioning
        for section in self.layout_order:
            if section == 'driving_lane':
                continue  # Skip, already added above
                
            # Calculate centered position for each section
            y_center = section_positions[section] + section_heights[section] / 2
            
            if section == 'rails':
                # Position label at the left side, vertically centered in the section
                ax.text(-10, y_center, 'RAILS', fontsize=14, ha='right', va='center', 
                      weight='bold', rotation=90)
            elif section == 'parking':
                # Position label at the left side, vertically centered in the section
                ax.text(-10, y_center, 'PARKING', fontsize=14, ha='right', va='center', 
                      weight='bold', rotation=90)
            elif section == 'yard_storage':
                # Position label at the left side, vertically centered in the section
                ax.text(-10, y_center, 'STORAGE YARD', fontsize=14, ha='right', va='center', 
                      weight='bold', rotation=90)
        
        # Add grid lines for better readability
        if not show_labels:
            # Add vertical grid lines at each rail slot boundary
            for i in range(self.num_railslots_per_track + 1):
                x_pos = i * self.rail_slot_length
                ax.axvline(x=x_pos, color='gray', linestyle='--', alpha=0.3)
            
            # Add horizontal grid lines at section boundaries
            for section, y_pos in section_positions.items():
                ax.axhline(y=y_pos, color='gray', linestyle='-', alpha=0.5)
                if section != self.layout_order[-1]:  # Not the last section
                    ax.axhline(y=y_pos + section_heights[section], color='gray', linestyle='-', alpha=0.5)
        
        ax.set_title('Container Terminal Layout', fontsize=16)
        ax.set_xlabel('Length (m)', fontsize=12)
        ax.set_ylabel('Width (m)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def get_section_width(self, section):
        """Helper method to calculate section width."""
        if section == 'rails':
            return self.num_railtracks * self.track_width + (self.num_railtracks - 1) * self.space_between_tracks
        elif section == 'parking':
            return self.parking_width
        elif section == 'driving_lane':
            return self.driving_lane_width
        elif section == 'yard_storage':
            return self.num_storage_rows * self.storage_slot_width
        return 0
    
    def are_points_in_same_span(self, point1, point2):
        """Determine if two points can be reached by the same crane position."""
        pos1 = self.positions[point1]
        pos2 = self.positions[point2]
        
        # Check if points are aligned along the crane's travel direction
        # For our layout, gantry movement is along the y-axis
        # and trolley movement is along the x-axis
        return abs(pos1[1] - pos2[1]) < self.track_width  # If y-positions are close

    def save_distance_matrix(self, filename='distance_matrix.pkl'):
        """Save the distance matrix and related data to a file."""
        data = {
            'distance_matrix': self.distance_matrix,
            'object_to_idx': self.object_to_idx,
            'objects': self.objects,
            'positions': self.positions,
            'layout_order': self.layout_order,
            'num_railtracks': self.num_railtracks,
            'num_railslots_per_track': self.num_railslots_per_track,
            'num_storage_rows': self.num_storage_rows,
            'parking_to_railslot_ratio': self.parking_to_railslot_ratio,
            'storage_to_railslot_ratio': self.storage_to_railslot_ratio,
            'rail_slot_length': self.rail_slot_length,
            'track_width': self.track_width,
            'space_between_tracks': self.space_between_tracks,
            'space_rails_to_parking': self.space_rails_to_parking,
            'space_driving_to_storage': self.space_driving_to_storage,
            'parking_width': self.parking_width,
            'driving_lane_width': self.driving_lane_width,
            'storage_slot_width': self.storage_slot_width
        }
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"Distance matrix saved to {filename}")
    
    def load_distance_matrix(self, filename='distance_matrix.pkl'):
        """Load the distance matrix and related data from a file."""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.distance_matrix = data['distance_matrix']
        self.object_to_idx = data['object_to_idx']
        self.objects = data['objects']
        self.positions = data['positions']
        self.layout_order = data.get('layout_order', ['rails', 'parking', 'driving_lane', 'yard_storage'])
        self.num_railtracks = data.get('num_railtracks', 3)
        self.num_railslots_per_track = data.get('num_railslots_per_track', 10)
        self.num_storage_rows = data.get('num_storage_rows', 6)
        self.parking_to_railslot_ratio = data.get('parking_to_railslot_ratio', 1.0)
        self.storage_to_railslot_ratio = data.get('storage_to_railslot_ratio', 2.0)
        self.rail_slot_length = data.get('rail_slot_length', 20.0)
        self.track_width = data.get('track_width', 3.0)
        self.space_between_tracks = data.get('space_between_tracks', 1.5)
        self.space_rails_to_parking = data.get('space_rails_to_parking', 5.0)
        self.space_driving_to_storage = data.get('space_driving_to_storage', 2.0)
        self.parking_width = data.get('parking_width', 4.0)
        self.driving_lane_width = data.get('driving_lane_width', 8.0)
        self.storage_slot_width = data.get('storage_slot_width', 10.0)
        print(f"Distance matrix loaded from {filename}")
        
    def to_dataframe(self):
        """Convert the distance matrix to a pandas DataFrame for easier analysis."""
        object_names = list(self.object_to_idx.keys())
        df = pd.DataFrame(self.distance_matrix, index=object_names, columns=object_names)
        return df
    
    def export_distance_csv(self, filename='distance_matrix.csv'):
        """Export the distance matrix to a CSV file."""
        df = self.to_dataframe()
        df.to_csv(filename)
        print(f"Distance matrix exported to {filename}")

def main():
    # Create a container terminal matching the user's requirements
    print("Creating container terminal...")
    terminal = ContainerTerminal(
        layout_order=['rails', 'parking', 'driving_lane', 'yard_storage'],
        num_railtracks=6,      
        num_railslots_per_track=29,
        num_storage_rows=5,   
        # Ratio parameters
        parking_to_railslot_ratio=1.0,
        storage_to_railslot_ratio=2.0,
        # Dimension parameters
        rail_slot_length=24.384,
        track_width=2.44,
        space_between_tracks=2.05,
        space_rails_to_parking=1.05,
        space_driving_to_storage=0.26,
        parking_width=4.0,
        driving_lane_width=4.0,
        storage_slot_width=2.5
    )
    
    # Visualize the terminal
    print("Visualizing terminal layout...")
    fig, ax = terminal.visualize(figsize=(30, 15), show_labels=True)  # Increased size, turned off individual labels
    plt.savefig('terminal_layout.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Make sure we calculate the distance matrix before trying to access it
    terminal.calculate_distance_matrix()
    
    # Save the distance matrix
    terminal.save_distance_matrix()
    
    # Export to CSV for easy inspection
    terminal.export_distance_csv()
    
    # Create a simpler visualization without individual labels for better readability
    print("Creating clean visualization...")
    terminal.visualize(figsize=(30, 15), show_labels=True)
    plt.savefig('terminal_layout_clean.png', dpi=300, bbox_inches='tight')
    
    # Get a list of valid objects for sample distances
    rail_slots = [k for k, v in terminal.objects.items() if v['type'] == 'rail_slot']
    storage_slots = [k for k, v in terminal.objects.items() if v['type'] == 'storage_slot']
    parking_spots = [k for k, v in terminal.objects.items() if v['type'] == 'parking_spot']
    
    # Example: Get distance between two objects
    obj1 = rail_slots[0]  # First rail slot (t1_1)
    obj2 = storage_slots[0]  # First storage slot (A1)
    distance = terminal.get_distance(obj1, obj2)
    print(f"Distance between {obj1} and {obj2}: {distance:.2f} meters")
    
    # Calculate more useful distances
    print("\nSample distances:")
    # Ensure all sample objects exist in the terminal
    samples = [
        (rail_slots[0], storage_slots[0]),                # First rail to first storage (t1_1 to A1)
        (rail_slots[-1], storage_slots[-1]),              # Last rail to last storage
        (parking_spots[19], storage_slots[19 + 40]),      # Middle parking to middle storage
        (rail_slots[14], parking_spots[14]),              # Rail to corresponding parking
        (storage_slots[4], storage_slots[4 + 40])         # Adjacent storage slots
    ]
    
    for src, dst in samples:
        distance = terminal.get_distance(src, dst)
        print(f"Distance from {src} to {dst}: {distance:.2f} meters")
    
    # Display some statistics
    print("\nContainer Terminal Statistics:")
    print(f"Total number of objects: {len(terminal.objects)}")
    print(f"Number of rail slots: {sum(1 for obj in terminal.objects.values() if obj['type'] == 'rail_slot')}")
    print(f"Number of parking spots: {sum(1 for obj in terminal.objects.values() if obj['type'] == 'parking_spot')}")
    print(f"Number of storage slots: {sum(1 for obj in terminal.objects.values() if obj['type'] == 'storage_slot')}")
    print(f"Total terminal length: {terminal.num_railslots_per_track * terminal.rail_slot_length} meters")
    print(f"Total terminal width: {max(pos[1] for pos in terminal.positions.values()) + terminal.storage_slot_width/2} meters")
    
    # Display ratio information
    print("\nRatio Information:")
    print(f"Parking to rail slot ratio: {terminal.parking_to_railslot_ratio}")
    print(f"Storage to rail slot ratio: {terminal.storage_to_railslot_ratio}")
    print(f"Number of parking spots: {terminal.num_parking_spots}")
    print(f"Number of storage slots per row: {terminal.num_storage_slots_per_row}")
    print(f"Total storage slots: {terminal.num_storage_rows * terminal.num_storage_slots_per_row}")
    
    # Performance of distance matrix calculation
    start_time = time.time()
    terminal.calculate_distance_matrix()
    end_time = time.time()
    print(f"\nTime to recalculate distance matrix: {end_time - start_time:.3f} seconds")
    
    # Sample of distance matrix size
    print(f"\nDistance matrix shape: {terminal.distance_matrix.shape}")
    df = terminal.to_dataframe()
    print(f"Size of distance dataframe: {df.shape}")
    print("Sample of distance matrix:")
    print(df.iloc[:5, :5])

if __name__ == "__main__":
    main()