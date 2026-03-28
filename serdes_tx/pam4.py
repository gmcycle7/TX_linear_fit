"""PAM signal generation for PAM2 / PAM4 / PAM6 / PAM8.

Supports arbitrary PRBS order (7-31) and PAM order (2/4/6/8).
Gray coding is used for power-of-2 PAM orders.
"""

import numpy as np
from .prbs import generate_prbs, SUPPORTED_ORDERS, prbs_period

# =====================================================================
#  General PAM support
# =====================================================================
SUPPORTED_PAM = [2, 4, 6, 8]


def pam_levels(pam_order):
    """Return normalized PAM levels: [-M+1, -M+3, ..., M-3, M-1].

    PAM2: [-1, +1]
    PAM4: [-3, -1, +1, +3]
    PAM6: [-5, -3, -1, +1, +3, +5]
    PAM8: [-7, -5, -3, -1, +1, +3, +5, +7]
    """
    M = int(pam_order)
    return np.array([2 * i - (M - 1) for i in range(M)], dtype=np.float64)


def _gray_encode(n, bits):
    """Binary to Gray code."""
    return n ^ (n >> 1)


def _bits_to_int(bit_group):
    """Convert a group of bits [MSB, ..., LSB] to integer."""
    val = 0
    for b in bit_group:
        val = (val << 1) | int(b)
    return val


def generate_pam_prbs(pam_order=4, prbs_order=9, length=None, seed=None):
    """Generate a PAM signal from a PRBS bit stream.

    Parameters
    ----------
    pam_order : int
        Number of levels: 2, 4, 6, or 8.
    prbs_order : int
        PRBS order: 7, 9, 11, 13, 15, 20, 23, or 31.
    length : int or None
        Number of PAM symbols. None = one PRBS period worth.
    seed : int or None
        PRBS LFSR seed. None = all-ones.

    Returns
    -------
    symbols : np.ndarray, float64
        PAM symbol values.
    """
    M = int(pam_order)
    if M not in SUPPORTED_PAM:
        raise ValueError(f"PAM order {M} not supported. Choose from {SUPPORTED_PAM}")

    levels = pam_levels(M)
    bits_per_sym = int(np.ceil(np.log2(M)))

    if length is None:
        length = prbs_period(prbs_order)

    # Generate enough PRBS bits
    n_bits = length * bits_per_sym * 2  # extra for non-power-of-2 rejection
    bits = generate_prbs(prbs_order, length=n_bits, seed=seed)

    if M in (2, 4, 8):
        # Power-of-2: use Gray coding with decorrelated PRBS streams
        return _pam_pow2(M, prbs_order, length, bits_per_sym, levels, seed)
    else:
        # Non-power-of-2: modular mapping
        return _pam_mod(M, bits, bits_per_sym, length, levels)


def _pam_pow2(M, prbs_order, length, bps, levels, seed):
    """PAM for power-of-2 orders using decorrelated PRBS streams.

    Uses one long PRBS sequence split into offset-separated segments
    for guaranteed decorrelation.
    """
    period = (1 << prbs_order) - 1
    # Generate one long PRBS, take segments at evenly-spaced offsets
    total = length + period  # enough for all offsets
    base_seed = seed if seed is not None else (1 << prbs_order) - 1
    long_prbs = generate_prbs(prbs_order, length=total, seed=base_seed)

    offsets = [i * (period // (bps + 1)) for i in range(bps)]
    streams = [long_prbs[off:off + length] for off in offsets]

    # Combine bits → Gray index → level
    indices = np.zeros(length, dtype=np.int32)
    for b in range(bps):
        indices |= streams[b].astype(np.int32) << (bps - 1 - b)

    # Gray decode: gray_index → natural_index
    natural = np.copy(indices)
    shift = 1
    while shift < M:
        natural ^= (natural >> shift)
        shift <<= 1

    symbols = levels[np.clip(natural, 0, M - 1)]
    return symbols


def _pam_mod(M, bits, bps, length, levels):
    """PAM for non-power-of-2 orders using modular mapping."""
    symbols = np.empty(length, dtype=np.float64)
    idx = 0
    sym_count = 0
    while sym_count < length and idx + bps <= len(bits):
        val = _bits_to_int(bits[idx:idx + bps])
        idx += bps
        if val < M:  # reject values >= M
            symbols[sym_count] = levels[val]
            sym_count += 1
    # If we ran out of bits, fill remaining with random levels
    if sym_count < length:
        rng = np.random.default_rng(42)
        symbols[sym_count:] = rng.choice(levels, size=length - sym_count)
    return symbols


# Normalized PAM4 levels (backward compat)
GRAY_LEVELS = np.array([-3.0, -1.0, 1.0, 3.0])

# ---- backward-compatible imports ----
from .prbs import generate_prbs9, generate_prbsq9


def bits_to_pam4(msb, lsb):
    """Convert MSB/LSB bit streams to PAM4 symbols using Gray coding.

    Parameters
    ----------
    msb, lsb : array-like of int
        Bit streams of equal length.

    Returns
    -------
    symbols : np.ndarray, float64
        PAM4 symbol values from {-3, -1, +1, +3}.
    indices : np.ndarray, int8
        Gray-coded symbol indices (0-3).
    """
    msb = np.asarray(msb, dtype=np.int8)
    lsb = np.asarray(lsb, dtype=np.int8)
    if msb.shape != lsb.shape:
        raise ValueError("MSB and LSB must have the same length")

    # Gray code: index = 2*MSB + (MSB XOR LSB)
    indices = (2 * msb + (msb ^ lsb)).astype(np.int8)
    symbols = GRAY_LEVELS[indices]
    return symbols, indices


def pam4_to_bits(symbols):
    """Slice PAM4 symbols to nearest level and decode to MSB/LSB bits.

    Parameters
    ----------
    symbols : array-like of float
        PAM4 waveform samples (quantized to nearest level).

    Returns
    -------
    msb, lsb : np.ndarray, int8
        Decoded bit streams.
    """
    symbols = np.asarray(symbols, dtype=np.float64)
    indices = np.argmin(
        np.abs(symbols[:, None] - GRAY_LEVELS[None, :]), axis=1
    )
    # Inverse Gray: msb = index >> 1, lsb = msb ^ (index & 1)
    msb = (indices >> 1).astype(np.int8)
    lsb = (msb ^ (indices & 1)).astype(np.int8)
    return msb, lsb


def generate_pam4_prbs9(length=None, seed=0x1FF):
    """Generate PAM4 signal from PRBS9 using consecutive bit pairs.

    Parameters
    ----------
    length : int or None
        Number of PAM4 symbols (uses 2*length bits). None for 511 symbols.
    seed : int
        PRBS9 LFSR seed.

    Returns
    -------
    symbols : np.ndarray
    indices : np.ndarray
    """
    if length is None:
        length = 511
    bits = generate_prbs9(length=2 * length, seed=seed)
    return bits_to_pam4(bits[0::2], bits[1::2])


def generate_pam4_prbsq9(length=None, **kwargs):
    """Generate PAM4 signal from PRBSQ9 (two decorrelated PRBS9).

    Parameters
    ----------
    length : int or None
        Number of PAM4 symbols. None for 511.
    **kwargs
        Passed to generate_prbsq9 (seed_msb, seed_lsb).

    Returns
    -------
    symbols : np.ndarray
    indices : np.ndarray
    """
    msb, lsb = generate_prbsq9(length=length, **kwargs)
    return bits_to_pam4(msb, lsb)


def upsample_pam4(symbols, samples_per_ui, method='zoh'):
    """Create oversampled PAM4 signal.

    Parameters
    ----------
    symbols : np.ndarray
        Symbol-rate PAM4 signal (1 sample/UI).
    samples_per_ui : int
        Oversampling factor (1 to 32).
    method : str
        'zoh'  - zero-order hold (sample-and-hold, sharp transitions).
        'interp' - bandlimited interpolation via polyphase filter.
                   Downsampling at the correct phase recovers the
                   original signal without ISI.

    Returns
    -------
    oversampled : np.ndarray
        Upsampled signal with length = len(symbols) * samples_per_ui.
    """
    from scipy.signal import resample_poly

    samples_per_ui = int(samples_per_ui)
    if samples_per_ui < 1:
        raise ValueError("samples_per_ui must be >= 1")
    if samples_per_ui == 1:
        return symbols.copy()

    if method == 'interp':
        return resample_poly(np.asarray(symbols, dtype=np.float64),
                             samples_per_ui, 1).astype(np.float64)
    else:
        return np.repeat(symbols, samples_per_ui)


def estimate_levels(signal, n_levels=4):
    """Estimate PAM4 voltage levels from a signal using histogram peaks.

    Parameters
    ----------
    signal : np.ndarray
        1 sample/UI PAM4 signal.
    n_levels : int
        Number of expected levels (default 4 for PAM4).

    Returns
    -------
    levels : np.ndarray
        Estimated voltage levels sorted ascending.
    """
    # Use histogram to find level clusters
    n_bins = max(50, len(signal) // 20)
    hist, bin_edges = np.histogram(signal, bins=n_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Smooth histogram
    kernel_size = max(3, n_bins // 20)
    kernel = np.ones(kernel_size) / kernel_size
    hist_smooth = np.convolve(hist, kernel, mode='same')

    # Find peaks: local maxima
    peaks = []
    for i in range(1, len(hist_smooth) - 1):
        if hist_smooth[i] > hist_smooth[i - 1] and hist_smooth[i] > hist_smooth[i + 1]:
            peaks.append((hist_smooth[i], bin_centers[i]))

    # Sort by height and take top n_levels
    peaks.sort(reverse=True)
    if len(peaks) >= n_levels:
        levels = sorted([p[1] for p in peaks[:n_levels]])
    else:
        # Fallback: use quantiles
        quantiles = np.linspace(0, 1, n_levels + 2)[1:-1]
        levels = list(np.quantile(signal, quantiles))

    return np.array(levels)
