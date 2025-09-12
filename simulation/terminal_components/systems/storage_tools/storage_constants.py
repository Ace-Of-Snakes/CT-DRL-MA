# simulation/terminal_components/storage/storage_constants.py

# Physical dimensions
BASE_SLOT_LENGTH_M = 6.096  # 20ft container length in meters
SLOT_WIDTH_M = 2.44  # Standard container width
SLOT_HEIGHT_M = 2.591  # Standard container height

# Split factor for sub-slots (handles 20, 22, 23, 26, 40, 45 ft containers)
# With 8 sub-slots per bay, we can handle: 20ft=3, 22ft=3, 23ft=4, 26ft=4, 40ft=8, 45ft=9
YARD_SPLIT_FACTOR = 8
BAY_LENGTH_M = 12.192  # 40ft in meters

# Container length mappings (in sub-slots)
CONTAINER_SUBSLOTS = {
    "20ft": 3,   # 6.096m
    "22ft": 3,   # 6.706m (rounded to 3 sub-slots)
    "23ft": 4,   # 7.010m
    "26ft": 4,   # 7.925m
    "40ft": 8,   # 12.192m (full bay)
    "45ft": 9,   # 13.716m (cross-bay)
    "FEU": 8,    # Standard 40ft
    "TEU": 3,    # Standard 20ft
    "Trailer": 8,
    "Swap Body": 8
}

# Goods type constants
GOODS_REGULAR = "Regular"
GOODS_REEFER = "Reefer"
GOODS_DANGEROUS = "DangerousGoods"

# Container type groups
EXCLUSIVE_TYPES = {"Trailer", "Swap Body"}
STACKABLE_TYPES = {"20ft", "22ft", "23ft", "26ft", "40ft", "45ft", "FEU", "TEU"}

# Maximum values
MAX_TIER_HEIGHT = 6  # Maximum stacking height
MAX_BAY_OVERFLOW = 1  # 45ft containers can overflow into 1 adjacent bay

# Placement rules
CROSS_BAY_MIN_LENGTH = 9  # Minimum sub-slots for cross-bay placement
FULL_BAY_LENGTH = 8  # Sub-slots for full bay