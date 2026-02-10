"""Configuration for yard storage and container placement."""
import os
from typing import Final, Dict, List, Tuple
import numpy as np
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


class YardPhysicalConfig:
    """Physical yard configuration."""
    
    BAY_LENGTH_FT: Final[int] = 40
    BAY_SPLIT_FACTOR: Final[int] = 20  # Sub-bay divisions
    SUB_BAY_LENGTH_FT: Final[float] = BAY_LENGTH_FT / BAY_SPLIT_FACTOR


class ContainerLengthConfig:
    """Container length configurations loaded from data."""
    
    _lengths_ft: List[int] = None
    _length_to_subbays: Dict[int, int] = None
    
    @classmethod
    def load_from_csv(cls, csv_path: str = None):
        """Load container length data."""
        if cls._lengths_ft is not None:
            return
        if csv_path is None:
            csv_path = os.path.join(_DATA_DIR, "container_data.csv")
        df = pd.read_csv(csv_path)
        cls._lengths_ft = sorted(df["LENGTH_IN_FEET"].unique().tolist())
        cls._length_to_subbays = {
            length: int(np.ceil(length / YardPhysicalConfig.SUB_BAY_LENGTH_FT))
            for length in cls._lengths_ft
        }
    
    @classmethod
    def get_container_lengths(cls) -> List[int]:
        """Get all container lengths in feet."""
        if cls._lengths_ft is None:
            cls.load_from_csv()
        return cls._lengths_ft
    
    @classmethod
    def get_length_to_subbays(cls) -> Dict[int, int]:
        """Get mapping of container length to sub-bay count."""
        if cls._length_to_subbays is None:
            cls.load_from_csv()
        return cls._length_to_subbays


class YardZoneConfig:
    """Configuration for special yard zones."""
    
    # Default special position configuration
    SWAP_BODY_ROW_1BASED: Final[int] = 1
    DG_ROWS_LARGE_YARD: Final[Tuple[int, ...]] = (3, 4, 5)
    DG_ROWS_SMALL_YARD: Final[Tuple[int, ...]] = (2, 3)
    DG_HALFWIDTH_BAYS: Final[int] = 3
    REEFERS_INCLUDE_SB_ROW: Final[bool] = True
    
    @staticmethod
    def generate_special_coordinates(
        n_rows: int,
        n_bays: int,
        sb_row_1based: int = None,
        dg_rows_large: Tuple[int, ...] = None,
        dg_rows_small: Tuple[int, ...] = None,
        dg_halfwidth: int = None,
        reefers_include_sb: bool = None
    ) -> List[Tuple[int, int, str]]:
        """
        Generate special position coordinates for yard.
        
        Returns:
            List of (bay_1based, row_1based, position_type) tuples.
            position_type: 'sb_t' (swap body), 'r' (reefer), 'dg' (dangerous goods)
        """
        # Use defaults if not provided
        sb_row = sb_row_1based or YardZoneConfig.SWAP_BODY_ROW_1BASED
        dg_large = dg_rows_large or YardZoneConfig.DG_ROWS_LARGE_YARD
        dg_small = dg_rows_small or YardZoneConfig.DG_ROWS_SMALL_YARD
        dg_half = dg_halfwidth or YardZoneConfig.DG_HALFWIDTH_BAYS
        reefer_sb = reefers_include_sb if reefers_include_sb is not None else YardZoneConfig.REEFERS_INCLUDE_SB_ROW
        
        coords: List[Tuple[int, int, str]] = []
        
        # Clamp swap body row
        sb_row = max(1, min(n_rows, sb_row))
        center_bay = (n_bays + 1) // 2
        
        # 1. Swap bodies/trailers on designated row
        for bay in range(1, n_bays + 1):
            coords.append((bay, sb_row, "sb_t"))
        
        # 2. Reefers on outer bays
        for row in range(1, n_rows + 1):
            if not reefer_sb and row == sb_row:
                continue
            coords.append((1, row, "r"))
            coords.append((n_bays, row, "r"))
        
        # 3. Dangerous goods in center
        is_large_yard = n_rows >= max(dg_large) if dg_large else False
        dg_rows = [r for r in (dg_large if is_large_yard else dg_small)
                   if 1 <= r <= n_rows and r != sb_row]
        
        for row in dg_rows:
            for delta in range(-dg_half, dg_half + 1):
                bay = center_bay + delta
                if 1 <= bay <= n_bays:
                    coords.append((bay, row, "dg"))
        
        return coords