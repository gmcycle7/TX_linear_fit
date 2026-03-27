"""PAM4 signal generation and encoding with Gray coding.

Gray-coded mapping:
    MSB  LSB  ->  Index  ->  Level
     0    0   ->    0    ->   -3
     0    1   ->    1    ->   -1
     1    1   ->    2    ->   +1
     1    0   ->    3    ->   +3
"""

import numpy as np
from .prbs import generate_prbs9, generate_prbsq9

# Normalized PAM4 levels
GRAY_LEVELS = np.array([-3.0, -1.0, 1.0, 3.0])


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
