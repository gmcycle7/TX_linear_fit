"""General PRBS pattern generation for SerDes testing.

Supports PRBS7 through PRBS31 with standard ITU-T polynomials.
"""

import numpy as np

# ── Primitive polynomials (0-indexed tap positions for XOR feedback) ──
PRBS_POLYS = {
    7:  ([6, 5],             "x^7+x^6+1"),
    9:  ([8, 4],             "x^9+x^5+1"),
    11: ([10, 8],            "x^11+x^9+1"),
    13: ([12, 11, 1, 0],     "x^13+x^12+x^2+x+1"),
    15: ([14, 13],           "x^15+x^14+1"),
    20: ([19, 2],            "x^20+x^3+1"),
    23: ([22, 17],           "x^23+x^18+1"),
    31: ([30, 27],           "x^31+x^28+1"),
}

SUPPORTED_ORDERS = sorted(PRBS_POLYS.keys())


def generate_prbs(order, length=None, seed=None):
    """Generate a PRBS bit sequence of any supported order.

    Parameters
    ----------
    order : int
        PRBS order (7, 9, 11, 13, 15, 20, 23, or 31).
    length : int or None
        Number of bits. None = one full period (2^order - 1).
    seed : int or None
        Initial LFSR state (nonzero). None = all-ones.

    Returns
    -------
    bits : np.ndarray, dtype int8
    """
    if order not in PRBS_POLYS:
        raise ValueError(f"Unsupported PRBS order {order}. "
                         f"Choose from {SUPPORTED_ORDERS}")

    taps, _ = PRBS_POLYS[order]
    mask = (1 << order) - 1
    period = mask  # 2^order - 1

    if seed is None:
        seed = mask  # all ones
    if seed == 0:
        raise ValueError("LFSR seed must be nonzero")
    seed &= mask

    if length is None:
        length = period

    lfsr = seed
    bits = np.empty(length, dtype=np.int8)
    msb = order - 1

    for i in range(length):
        bits[i] = (lfsr >> msb) & 1
        fb = 0
        for t in taps:
            fb ^= (lfsr >> t) & 1
        lfsr = ((lfsr << 1) | fb) & mask

    return bits


def prbs_period(order):
    """Return the period of a PRBS sequence: 2^order - 1."""
    return (1 << order) - 1


def prbs_info(order):
    """Return (period, polynomial_string) for a PRBS order."""
    _, poly_str = PRBS_POLYS[order]
    return prbs_period(order), poly_str


# ── Backward-compatible aliases ──

def generate_prbs9(length=None, seed=0x1FF):
    return generate_prbs(9, length=length, seed=seed)

PRBS9_PERIOD = 511

def generate_prbsq9(length=None, seed_msb=0x1FF, seed_lsb=0x1AA):
    if length is None:
        length = 511
    msb = generate_prbs(9, length=length, seed=seed_msb)
    lsb = generate_prbs(9, length=length, seed=seed_lsb)
    return msb, lsb

def verify_prbs9(bits):
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < 9:
        return False, None
    seed = 0
    for i in range(9):
        seed = (seed << 1) | int(bits[i])
    if seed == 0:
        return False, None
    ref = generate_prbs(9, length=len(bits), seed=seed)
    valid = np.array_equal(bits, ref)
    return valid, seed if valid else None
