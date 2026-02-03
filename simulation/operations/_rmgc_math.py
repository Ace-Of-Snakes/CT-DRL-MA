# simulation/operations/_rmgc_math.py
"""
Numba-optimized crane timing helpers.
"""
from numba import njit


@njit(cache=True, fastmath=True)
def axis_time(dist: float, vmax: float, acc: float) -> float:
    """
    Trapezoidal motion profile time for a single axis (nopython).
    Triangular profile if distance too short to reach vmax.
    """
    if dist <= 0.0:
        return 0.0
    t_acc = vmax / acc
    d_acc = 0.5 * acc * t_acc * t_acc
    if dist <= 2.0 * d_acc:
        # triangular profile
        return 2.0 * (dist / acc) ** 0.5
    # trapezoidal
    return 2.0 * t_acc + (dist - 2.0 * d_acc) / vmax


@njit(cache=True, fastmath=True)
def move_time(x1: float, y1: float, z1: float,
              x2: float, y2: float, z2: float,
              trolley_v: float, trolley_a: float,
              gantry_v: float, gantry_a: float,
              hoist_v: float, hoist_a: float,
              plane_z: float,
              handling_s: float) -> float:
    """
    Critical-path move time across trolley (y), gantry (x), hoist (z) axes,
    including down-to-pick, up-to-plane, in-plane travel (max of x/y),
    lower-to-destination, and a fixed handling time.
    """
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    # Hoist: lower to source z, then raise to plane, then lower to dest z
    hoist_to_pickup   = axis_time(abs(plane_z - z1), hoist_v, hoist_a)
    hoist_from_pickup = axis_time(abs(plane_z - z1), hoist_v, hoist_a)

    # In-plane: critical path between gantry (x) and trolley (y)
    gx = axis_time(dx, gantry_v, gantry_a)
    gy = axis_time(dy, trolley_v, trolley_a)
    plane = gx if gx >= gy else gy

    hoist_to_dest = axis_time(abs(plane_z - z2), hoist_v, hoist_a)
    return hoist_to_pickup + hoist_from_pickup + plane + hoist_to_dest + handling_s