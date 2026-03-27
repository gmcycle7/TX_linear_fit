"""Channel impairment models for SerDes TX simulation.

Includes AWGN, ISI (FIR channel), bandwidth limitation,
and jitter (RJ + DJ).
"""

import numpy as np
from scipy.signal import butter, filtfilt, lfilter, bessel
from scipy.interpolate import interp1d


def add_awgn(signal, snr_db=None, noise_std=None, rng=None):
    """Add white Gaussian noise to signal.

    Specify either snr_db or noise_std.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    snr_db : float or None
        Target SNR in dB (relative to signal RMS).
    noise_std : float or None
        Noise standard deviation (absolute).
    rng : np.random.Generator or int or None
        Random number generator or seed.

    Returns
    -------
    noisy : np.ndarray
        Signal with added noise.
    noise : np.ndarray
        The noise that was added.
    """
    signal = np.asarray(signal, dtype=np.float64)
    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(rng)

    if snr_db is not None:
        sig_rms = np.sqrt(np.mean(signal ** 2))
        if sig_rms < 1e-30:
            raise ValueError("Signal has zero power; cannot set SNR")
        noise_std = sig_rms / (10 ** (snr_db / 20.0))
    elif noise_std is None:
        raise ValueError("Must specify either snr_db or noise_std")

    noise = rng.normal(0, noise_std, size=signal.shape)
    return signal + noise, noise


def apply_isi(signal, channel_fir):
    """Apply ISI via FIR channel convolution.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    channel_fir : array-like
        Channel impulse response (FIR taps).

    Returns
    -------
    output : np.ndarray
        Convolved signal (same length as input, causal truncation).
    """
    channel_fir = np.asarray(channel_fir, dtype=np.float64)
    output = np.convolve(signal, channel_fir, mode='full')
    # Return same length as input (causal)
    return output[:len(signal)]


def apply_bandwidth_limit(signal, samples_per_ui, bw_ratio=0.75,
                          filter_order=4, filter_type='bessel'):
    """Apply bandwidth limitation via low-pass filter.

    Parameters
    ----------
    signal : np.ndarray
        Oversampled signal.
    samples_per_ui : int
        Oversampling factor.
    bw_ratio : float
        3-dB bandwidth as fraction of baud rate (e.g., 0.75 = 75% of f_baud).
    filter_order : int
        Filter order (default 4).
    filter_type : str
        'bessel' (maximally flat group delay) or 'butter' (maximally flat magnitude).

    Returns
    -------
    filtered : np.ndarray
        Bandwidth-limited signal.
    """
    if samples_per_ui < 2:
        raise ValueError("Bandwidth limitation requires oversampled signal (spui >= 2)")

    # Normalized cutoff: wn = f_cutoff / f_nyquist
    # f_nyquist = spui * f_baud / 2
    # f_cutoff = bw_ratio * f_baud
    wn = 2.0 * bw_ratio / samples_per_ui

    if wn >= 1.0:
        return signal.copy()  # cutoff above Nyquist, no filtering
    wn = min(wn, 0.99)  # guard against numerical issues

    if filter_type == 'bessel':
        b, a = bessel(filter_order, wn, btype='low', analog=False)
    else:
        b, a = butter(filter_order, wn, btype='low', analog=False)

    # Zero-phase filtering to avoid group delay artifacts
    # Pad signal to avoid edge transients
    pad_len = min(3 * max(len(b), len(a)), len(signal) - 1)
    if pad_len < 1:
        return lfilter(b, a, signal)

    filtered = filtfilt(b, a, signal, padlen=pad_len)
    return filtered


def add_jitter(signal, samples_per_ui, rj_rms_ui=0.0, dj_amp_ui=0.0,
               dj_freq_ratio=0.1, rng=None):
    """Add random jitter (RJ) and deterministic jitter (DJ) to signal.

    Jitter is modeled as timing perturbation on the signal transitions,
    implemented via interpolation-based resampling.

    Parameters
    ----------
    signal : np.ndarray
        Oversampled signal.
    samples_per_ui : int
        Oversampling factor.
    rj_rms_ui : float
        Random jitter RMS in UI fractions (e.g., 0.01 = 1% of UI).
    dj_amp_ui : float
        Deterministic jitter peak amplitude in UI fractions.
    dj_freq_ratio : float
        DJ frequency as fraction of baud rate.
    rng : np.random.Generator or int or None
        Random number generator or seed.

    Returns
    -------
    jittered : np.ndarray
        Signal with jitter applied.
    jitter_samples : np.ndarray
        Applied jitter in sample units (for diagnostics).
    """
    if samples_per_ui < 2:
        raise ValueError("Jitter injection requires oversampled signal")

    signal = np.asarray(signal, dtype=np.float64)
    n = len(signal)

    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(rng)

    # Time axis in sample units
    t_ideal = np.arange(n, dtype=np.float64)

    # Generate jitter per UI, then interpolate to sample rate
    n_ui = n // samples_per_ui
    jitter_ui = np.zeros(n_ui)

    if rj_rms_ui > 0:
        rj = rng.normal(0, rj_rms_ui, size=n_ui)
        jitter_ui += rj

    if dj_amp_ui > 0:
        t_ui = np.arange(n_ui, dtype=np.float64)
        dj = dj_amp_ui * np.sin(2 * np.pi * dj_freq_ratio * t_ui)
        jitter_ui += dj

    # Convert jitter from UI to samples
    jitter_ui_samples = jitter_ui * samples_per_ui

    # Interpolate jitter to sample rate (smooth between UIs)
    t_ui_centers = np.arange(n_ui) * samples_per_ui + samples_per_ui // 2
    if n_ui >= 2:
        jitter_interp = interp1d(
            t_ui_centers, jitter_ui_samples,
            kind='linear', fill_value='extrapolate'
        )
        jitter_samples = jitter_interp(t_ideal)
    else:
        jitter_samples = np.full(n, jitter_ui_samples[0] if n_ui > 0 else 0.0)

    # Resample signal at jittered times
    t_jittered = t_ideal + jitter_samples
    t_jittered = np.clip(t_jittered, 0, n - 1)

    interp_func = interp1d(t_ideal, signal, kind='cubic',
                           bounds_error=False, fill_value='extrapolate')
    jittered = interp_func(t_jittered)

    return jittered, jitter_samples


def apply_channel(signal, samples_per_ui=1, channel_fir=None,
                  bw_ratio=None, snr_db=None, noise_std=None,
                  rj_rms_ui=0.0, dj_amp_ui=0.0, dj_freq_ratio=0.1,
                  filter_type='bessel', rng=None):
    """Apply a complete channel model to the signal.

    Impairments are applied in order: ISI -> BW limit -> Jitter -> AWGN.

    Parameters
    ----------
    signal : np.ndarray
        Input signal.
    samples_per_ui : int
        Oversampling factor.
    channel_fir : array-like or None
        ISI channel impulse response.
    bw_ratio : float or None
        Bandwidth as fraction of baud rate (None = no BW limit).
    snr_db : float or None
        Target SNR for AWGN (None = no noise).
    noise_std : float or None
        Noise standard deviation (alternative to snr_db).
    rj_rms_ui, dj_amp_ui, dj_freq_ratio : float
        Jitter parameters (see add_jitter).
    filter_type : str
        Low-pass filter type.
    rng : np.random.Generator or int or None
        Random number generator or seed.

    Returns
    -------
    output : np.ndarray
        Impaired signal.
    """
    if rng is None or isinstance(rng, int):
        rng = np.random.default_rng(rng)

    output = np.asarray(signal, dtype=np.float64).copy()

    # 1. ISI
    if channel_fir is not None:
        output = apply_isi(output, channel_fir)

    # 2. Bandwidth limitation
    if bw_ratio is not None and samples_per_ui >= 2:
        output = apply_bandwidth_limit(
            output, samples_per_ui, bw_ratio=bw_ratio,
            filter_type=filter_type
        )

    # 3. Jitter
    if (rj_rms_ui > 0 or dj_amp_ui > 0) and samples_per_ui >= 2:
        output, _ = add_jitter(
            output, samples_per_ui,
            rj_rms_ui=rj_rms_ui, dj_amp_ui=dj_amp_ui,
            dj_freq_ratio=dj_freq_ratio, rng=rng
        )

    # 4. AWGN
    if snr_db is not None or noise_std is not None:
        output, _ = add_awgn(output, snr_db=snr_db, noise_std=noise_std, rng=rng)

    return output


def make_channel_from_il(il_at_nyquist_db, samples_per_ui,
                         num_ui_span=8, model='linear'):
    """Create an oversampled channel FIR from an insertion-loss spec.

    Parameters
    ----------
    il_at_nyquist_db : float
        Insertion loss at baud Nyquist (negative, e.g. -10).
    samples_per_ui : int
        Oversampling factor.
    num_ui_span : int
        Impulse-response length in UI (default 8).
    model : str
        'linear' — IL(f) proportional to f.
        'sqrt'   — IL(f) proportional to sqrt(f)  (skin-effect).

    Returns
    -------
    h : np.ndarray
        Channel FIR at the oversampled rate.
    freq_norm : np.ndarray
        Frequency axis normalised to baud rate  (0 … spui/2).
    H_db : np.ndarray
        Designed magnitude response in dB.
    """
    from scipy.signal import firwin2

    spui = int(samples_per_ui)
    n_taps = num_ui_span * spui + 1          # odd length
    n_freq = 2048
    f01 = np.linspace(0, 1, n_freq)          # 0‥1 = 0‥f_s/2
    f_baud_nyq = 1.0 / spui                  # baud Nyquist in normalised units

    if model == 'sqrt':
        ratio = np.sqrt(np.clip(f01 / f_baud_nyq, 0, None))
    else:                                     # linear
        ratio = f01 / f_baud_nyq

    il_db = float(il_at_nyquist_db) * ratio
    il_db[0] = 0.0
    il_db = np.clip(il_db, -80, 0)
    mag = 10.0 ** (il_db / 20.0)

    h = firwin2(n_taps, f01, mag)

    # Normalise so that DC gain ≈ 1
    dc = np.sum(h)
    if abs(dc) > 1e-12:
        h /= dc

    freq_norm = f01 * (spui / 2.0)           # in multiples of baud rate
    return h, freq_norm, il_db


def default_channel_fir():
    """Return a typical PCB-trace channel impulse response.

    Models a moderate-loss channel with pre/post cursor ISI.

    Returns
    -------
    h : np.ndarray
        Channel FIR taps.
    """
    h = np.array([
        0.02, -0.04, 0.08, 0.85, 0.15, -0.08, 0.04, -0.02, 0.01
    ])
    h = h / np.sum(np.abs(h))  # normalize to unit energy approx
    return h
