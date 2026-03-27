"""PRBS pattern generation for PAM4 SerDes testing.

Implements PRBS9 (ITU-T O.150, polynomial x^9 + x^5 + 1) and PRBSQ9
(quaternary PRBS using two decorrelated PRBS9 streams).
"""

import numpy as np

PRBS9_POLY_TAPS = (8, 4)  # 0-indexed tap positions for x^9 + x^5 + 1
PRBS9_PERIOD = 2**9 - 1   # 511


def generate_prbs9(length=None, seed=0x1FF):
    """Generate PRBS9 bit sequence using polynomial x^9 + x^5 + 1.

    Parameters
    ----------
    length : int or None
        Number of bits to generate. None produces one full period (511).
    seed : int
        Initial 9-bit LFSR state (must be nonzero).

    Returns
    -------
    bits : np.ndarray, dtype int8
        Binary sequence of 0s and 1s.
    """
    if seed == 0:
        raise ValueError("LFSR seed must be nonzero")
    seed &= 0x1FF
    if length is None:
        length = PRBS9_PERIOD

    lfsr = seed
    bits = np.empty(length, dtype=np.int8)
    tap_hi, tap_lo = PRBS9_POLY_TAPS

    for i in range(length):
        bits[i] = (lfsr >> tap_hi) & 1
        feedback = ((lfsr >> tap_hi) ^ (lfsr >> tap_lo)) & 1
        lfsr = ((lfsr << 1) | feedback) & 0x1FF

    return bits


def generate_prbsq9(length=None, seed_msb=0x1FF, seed_lsb=0x1AA):
    """Generate PRBSQ9 quaternary pattern for PAM4 testing.

    Uses two decorrelated PRBS9 sequences as MSB and LSB streams.
    Default seeds provide maximal decorrelation.

    Parameters
    ----------
    length : int or None
        Number of PAM4 symbols to generate. None for one period (511).
    seed_msb : int
        LFSR seed for MSB stream.
    seed_lsb : int
        LFSR seed for LSB stream.

    Returns
    -------
    msb : np.ndarray, dtype int8
        MSB bit stream.
    lsb : np.ndarray, dtype int8
        LSB bit stream.
    """
    if length is None:
        length = PRBS9_PERIOD
    msb = generate_prbs9(length=length, seed=seed_msb)
    lsb = generate_prbs9(length=length, seed=seed_lsb)
    return msb, lsb


def verify_prbs9(bits):
    """Verify that a bit sequence matches a valid PRBS9 sequence.

    Parameters
    ----------
    bits : array-like
        Binary sequence to verify.

    Returns
    -------
    valid : bool
        True if the sequence matches a PRBS9 pattern.
    seed : int or None
        Detected LFSR seed if valid, None otherwise.
    """
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < 9:
        return False, None

    # Reconstruct seed from first 9 bits
    seed = 0
    for i in range(9):
        seed = (seed << 1) | int(bits[i])

    if seed == 0:
        return False, None

    ref = generate_prbs9(length=len(bits), seed=seed)
    valid = np.array_equal(bits, ref)
    return valid, seed if valid else None
