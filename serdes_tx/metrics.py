"""TX signal quality metrics for PAM4 SerDes analysis.

Computes SNDR, RLM, eye diagram metrics, and insertion loss.
"""

import numpy as np
from scipy.fft import rfft, rfftfreq


def compute_sndr(signal, reconstructed):
    """Compute signal-to-noise-and-distortion ratio.

    SNDR = 10 * log10( power(signal) / power(error) )

    Parameters
    ----------
    signal : np.ndarray
        Original (or measured) signal.
    reconstructed : np.ndarray
        Reconstructed signal (e.g., from pulse response model).

    Returns
    -------
    sndr_db : float
        SNDR in decibels.
    error : np.ndarray
        Residual error signal.
    """
    signal = np.asarray(signal, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    min_len = min(len(signal), len(reconstructed))
    s = signal[:min_len]
    r = reconstructed[:min_len]

    error = s - r
    sig_power = np.mean(s ** 2)
    err_power = np.mean(error ** 2)

    if err_power < 1e-30:
        sndr_db = np.inf
    else:
        sndr_db = 10.0 * np.log10(sig_power / err_power)

    return float(sndr_db), error


def compute_rlm(signal, symbols=None):
    """Compute ratio level mismatch for PAM4.

    RLM = 3 * min(ES_lower, ES_middle, ES_upper) / V_swing

    Ideal RLM = 1.0. Values < 0.9 indicate significant level mismatch.

    Parameters
    ----------
    signal : np.ndarray
        1 sample/UI PAM4 signal.
    symbols : np.ndarray or None
        Known symbol sequence for level classification. If None,
        levels are estimated from histogram clustering.

    Returns
    -------
    rlm : float
        Ratio level mismatch (1.0 = ideal).
    levels : np.ndarray
        Estimated voltage levels [V0, V1, V2, V3].
    separations : np.ndarray
        Eye separations [ES_lower, ES_middle, ES_upper].
    """
    signal = np.asarray(signal, dtype=np.float64)

    if symbols is not None:
        symbols = np.asarray(symbols, dtype=np.float64)
        min_len = min(len(signal), len(symbols))
        signal = signal[:min_len]
        symbols = symbols[:min_len]

        # Compute mean measured value for each ideal level
        ideal_levels = np.array([-3.0, -1.0, 1.0, 3.0])
        levels = np.zeros(4)
        for i, lev in enumerate(ideal_levels):
            mask = np.abs(symbols - lev) < 0.5
            if np.any(mask):
                levels[i] = np.mean(signal[mask])
            else:
                levels[i] = lev
    else:
        levels = _estimate_levels_kmeans(signal, 4)

    levels = np.sort(levels)
    separations = np.diff(levels)  # [ES_lower, ES_middle, ES_upper]
    v_swing = levels[-1] - levels[0]

    if v_swing < 1e-12:
        return 0.0, levels, separations

    rlm = 3.0 * np.min(separations) / v_swing
    return float(rlm), levels, separations


def _estimate_levels_kmeans(signal, n_levels, max_iter=50):
    """Simple k-means clustering to estimate PAM4 voltage levels."""
    sorted_s = np.sort(signal)
    n = len(sorted_s)

    # Initialize centroids from quantiles
    centroids = np.array([
        np.median(sorted_s[i * n // n_levels:(i + 1) * n // n_levels])
        for i in range(n_levels)
    ])

    for _ in range(max_iter):
        # Assign to nearest centroid
        dists = np.abs(signal[:, None] - centroids[None, :])
        labels = np.argmin(dists, axis=1)

        # Update centroids
        new_centroids = np.array([
            np.mean(signal[labels == i]) if np.any(labels == i) else centroids[i]
            for i in range(n_levels)
        ])

        if np.allclose(new_centroids, centroids, atol=1e-8):
            break
        centroids = new_centroids

    return np.sort(centroids)


def compute_eye_metrics(signal, samples_per_ui, levels=None, phase=None):
    """Compute PAM4 eye diagram metrics.

    Parameters
    ----------
    signal : np.ndarray
        Oversampled waveform (samples_per_ui > 1) or 1-sps signal.
    samples_per_ui : int
        Oversampling factor.
    levels : np.ndarray or None
        Known PAM4 voltage levels [V0, V1, V2, V3]. Estimated if None.

    Returns
    -------
    metrics : dict
        'eye_heights' : np.ndarray, shape (3,)
            Vertical opening of each of the 3 PAM4 eyes.
        'eye_height_worst' : float
            Minimum eye height.
        'eye_widths' : np.ndarray, shape (3,) or None
            Horizontal opening in UI fractions (None if 1 sps).
        'eye_width_worst' : float or None
        'eye_linearity' : float
            Linearity metric: max deviation of level spacing from ideal.
    """
    signal = np.asarray(signal, dtype=np.float64)
    spui = int(samples_per_ui)

    # Get 1-sps samples for level estimation
    n_sym = len(signal) // spui
    if phase is None:
        phase = spui // 2 if spui > 1 else 0
    samples_1sps = signal[phase::spui][:n_sym]

    if levels is None:
        levels = _estimate_levels_kmeans(samples_1sps, 4)
    levels = np.sort(levels)

    # Decision thresholds (midpoints between levels)
    thresholds = 0.5 * (levels[:-1] + levels[1:])  # 3 thresholds

    # Classify each sample to its nearest level
    dists = np.abs(samples_1sps[:, None] - levels[None, :])
    labels = np.argmin(dists, axis=1)

    # --- Eye heights at center of UI ---
    eye_heights = np.zeros(3)
    for eye_idx in range(3):
        # Samples classified as lower level of this eye
        mask_lo = labels == eye_idx
        # Samples classified as upper level of this eye
        mask_hi = labels == eye_idx + 1

        if np.sum(mask_lo) > 1 and np.sum(mask_hi) > 1:
            upper_of_lower = np.percentile(samples_1sps[mask_lo], 99.5)
            lower_of_upper = np.percentile(samples_1sps[mask_hi], 0.5)
            eye_heights[eye_idx] = max(0.0, lower_of_upper - upper_of_lower)
        else:
            eye_heights[eye_idx] = levels[eye_idx + 1] - levels[eye_idx]

    # --- Eye widths (only meaningful for oversampled) ---
    eye_widths = None
    eye_width_worst = None

    if spui >= 4:
        eye_widths = np.zeros(3)
        # Fold signal into UI-length segments
        n_full = (n_sym - 1) * spui
        if n_full > spui:
            folded = signal[:n_full].reshape(-1, spui)

            for eye_idx in range(3):
                lo = levels[eye_idx]
                hi = levels[eye_idx + 1]
                mid = 0.5 * (lo + hi)
                margin = 0.1 * (hi - lo)

                # For each time offset, check if all traces clear the eye
                open_mask = np.ones(spui, dtype=bool)
                for t in range(spui):
                    col = folded[:, t]
                    # A trace crosses this eye if it's between the levels
                    in_eye = (col > lo + margin) & (col < hi - margin)
                    # The eye is open at time t if NO trace violates it
                    # Actually: eye is open if all samples at time t that
                    # belong to this eye region are within bounds.
                    # Simpler: eye is open at t if min gap is positive.
                    near_lo = col[(col >= lo - (hi - lo)) & (col <= mid)]
                    near_hi = col[(col > mid) & (col <= hi + (hi - lo))]
                    if len(near_lo) > 0 and len(near_hi) > 0:
                        gap = np.percentile(near_hi, 1) - np.percentile(near_lo, 99)
                        open_mask[t] = gap > 0
                    else:
                        open_mask[t] = True

                eye_widths[eye_idx] = np.sum(open_mask) / spui

        eye_width_worst = float(np.min(eye_widths)) if eye_widths is not None else None

    # --- Eye linearity ---
    separations = np.diff(levels)
    ideal_sep = (levels[-1] - levels[0]) / 3.0
    if ideal_sep > 1e-12:
        eye_linearity = 1.0 - np.max(np.abs(separations - ideal_sep)) / ideal_sep
    else:
        eye_linearity = 0.0

    return {
        'eye_heights': eye_heights,
        'eye_height_worst': float(np.min(eye_heights)),
        'eye_widths': eye_widths,
        'eye_width_worst': eye_width_worst,
        'eye_linearity': float(eye_linearity),
        'levels': levels,
        'thresholds': thresholds,
    }


def compute_insertion_loss(pulse, samples_per_ui=1, baud_rate_ghz=1.0,
                           n_fft=None):
    """Compute insertion loss from TX pulse response via FFT.

    IL(f) = -20 * log10(|H(f)| / |H(0)|)

    Parameters
    ----------
    pulse : np.ndarray
        Pulse response taps (symbol-rate).
    samples_per_ui : int
        Samples per UI of the pulse response.
    baud_rate_ghz : float
        Baud rate in GHz (for frequency axis scaling).
    n_fft : int or None
        FFT size (None = auto, at least 1024).

    Returns
    -------
    freq_ghz : np.ndarray
        Frequency axis in GHz.
    il_db : np.ndarray
        Insertion loss in dB (positive values = loss).
    H : np.ndarray
        Complex frequency response.
    """
    pulse = np.asarray(pulse, dtype=np.float64)
    fs = baud_rate_ghz * samples_per_ui  # sample rate in GHz

    if n_fft is None:
        n_fft = max(1024, 2 ** int(np.ceil(np.log2(len(pulse) * 8))))

    H = rfft(pulse, n=n_fft)
    freq = rfftfreq(n_fft, d=1.0 / fs)  # in GHz

    H_mag = np.abs(H)
    H_dc = H_mag[0] if H_mag[0] > 1e-30 else 1.0

    # Insertion loss: how much the signal is attenuated relative to DC
    with np.errstate(divide='ignore'):
        il_db = -20.0 * np.log10(np.maximum(H_mag / H_dc, 1e-30))

    return freq, il_db, H
