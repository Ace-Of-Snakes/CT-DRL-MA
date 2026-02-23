# simulation/training/scenarios/5_terminal_truck/__init__.py
"""Tier 5 -- Terminal truck scenarios."""
from .tt_cant_carry_regular import TTCantCarryRegular
from .regular_truck_preferred import RegularTruckPreferred
from .tt_only_option import TTOnlyOption
from .basic_tt_dispatch import BasicTTDispatch

SCENARIOS = [TTCantCarryRegular, RegularTruckPreferred, TTOnlyOption, BasicTTDispatch]
