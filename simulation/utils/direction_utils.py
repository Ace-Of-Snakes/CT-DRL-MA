# simulation/utils/direction_utils.py
"""Centralised direction checking for containers.

Eliminates fragile enum-vs-string comparisons scattered across
env.py, state_encoder.py, hierarchical_moves.py, and terminal_manager.py.
"""


def _direction_str(container) -> str:
    """Extract direction as a plain string, handling enums and raw strings."""
    d = getattr(container, "direction", None)
    if d is None:
        return ""
    return d.value if hasattr(d, "value") else str(d)


def is_import(container) -> bool:
    """True if container direction is Import."""
    return _direction_str(container) == "Import"


def is_export(container) -> bool:
    """True if container direction is Export."""
    return _direction_str(container) == "Export"


def direction_float(container) -> float:
    """Numeric direction for state encoding: 0.0=Import, 1.0=Export, 0.5=unknown."""
    s = _direction_str(container)
    if s == "Import":
        return 0.0
    if s == "Export":
        return 1.0
    return 0.5