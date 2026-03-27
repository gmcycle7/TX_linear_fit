"""Clock data recovery and signal alignment for SerDes analysis.

Provides cross-correlation alignment, max-eye CDR, and Gardner CDR
for oversampled PAM4 waveforms.
"""

import numpy as np
from scipy.signal import correlate, correlation_lags


def align_by_correlation(measured, reference):
    """Align measured signal to reference via cross-correlation.

    Parameters
    ----------
    measured : np.ndarray
        Measured waveform (any sample rate).
    reference : np.ndarray
        Reference signal (same sample rate as measured).

    Returns
    -------
    aligned : np.ndarray
        Aligned portion of measured signal (trimmed to reference length).
    lag : int
        Detected lag in samples (positive = measured is delayed).
    corr_peak : float
        Normalized correlation peak value.
    """
    measured = np.asarray(measured, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)

    corr = correlate(measured, reference, mode='full')
    lags = correlation_lags(len(measured), len(reference), mode='full')

    # Normalize
    norm = np.sqrt(np.sum(measured**2) * np.sum(reference**2))
    if norm > 0:
        corr_norm = corr / norm
    else:
        corr_norm = corr

    best_idx = np.argmax(np.abs(corr_norm))
    lag = lags[best_idx]
    corr_peak = corr_norm[best_idx]

    # Apply lag: positive lag means measured starts later
    if lag >= 0:
        aligned = measured[lag:]
    else:
        aligned = measured[:lag]  # trim from end

    min_len = min(len(aligned), len(reference))
    aligned = aligned[:min_len]

    return aligned, int(lag), float(corr_peak)


def find_optimal_phase(signal, samples_per_ui, symbols=None):
    """Find optimal sampling phase by maximizing eye opening.

    Tries all phase offsets in [0, samples_per_ui) and selects the one
    that either maximizes correlation with known symbols or maximizes
    the eye opening metric.

    Parameters
    ----------
    signal : np.ndarray
        Oversampled waveform.
    samples_per_ui : int
        Oversampling factor.
    symbols : np.ndarray or None
        Known PAM4 symbols for correlation-based selection.

    Returns
    -------
    best_phase : int
        Optimal phase offset in [0, samples_per_ui).
    best_metric : float
        Value of the selection metric at optimal phase.
    """
    best_phase = 0
    best_metric = -np.inf

    n_sym = len(signal) // samples_per_ui

    for phase in range(samples_per_ui):
        sampled = signal[phase::samples_per_ui][:n_sym]

        if symbols is not None:
            # Cross-correlation to handle arbitrary symbol delays
            # (TX FIR may introduce multi-symbol delay)
            min_len = min(len(sampled), len(symbols))
            s, r = sampled[:min_len], symbols[:min_len]
            corr = correlate(s, r, mode='full')
            norm = np.sqrt(np.sum(s ** 2) * np.sum(r ** 2))
            if norm > 1e-12:
                metric = float(np.max(np.abs(corr)) / norm)
            else:
                metric = 0.0
        else:
            metric = _eye_opening_metric(sampled)

        if metric > best_metric:
            best_metric = metric
            best_phase = phase

    return best_phase, best_metric


def _eye_opening_metric(samples):
    """Estimate PAM4 eye opening from symbol-rate samples."""
    if len(samples) < 16:
        return 0.0

    sorted_s = np.sort(samples)
    n = len(sorted_s)

    # Estimate 4 levels from quartile medians
    levels = []
    for i in range(4):
        start = i * n // 4
        end = (i + 1) * n // 4
        levels.append(np.median(sorted_s[start:end]))

    # Metric: minimum separation between adjacent levels
    separations = [levels[i + 1] - levels[i] for i in range(3)]
    if min(separations) <= 0:
        return 0.0
    return min(separations)


def gardner_cdr(signal, samples_per_ui, loop_bw=0.005, damping=0.707):
    """Gardner timing error detector with second-order loop.

    Parameters
    ----------
    signal : np.ndarray
        Oversampled waveform (samples_per_ui >= 2).
    samples_per_ui : int
        Oversampling factor.
    loop_bw : float
        Normalized loop bandwidth (0 < bw < 0.1 typical).
    damping : float
        Loop damping factor.

    Returns
    -------
    optimal_phase : int
        Recovered sampling phase in [0, samples_per_ui).
    timing_errors : np.ndarray
        TED output sequence (for diagnostics).
    """
    if samples_per_ui < 2:
        raise ValueError("Gardner CDR requires samples_per_ui >= 2")

    spui = samples_per_ui
    n_sym = len(signal) // spui - 1

    # Second-order loop filter gains
    theta_n = loop_bw * 2 * np.pi
    kp = (2 * damping * theta_n) / spui       # proportional
    ki = (theta_n ** 2) / spui                  # integral

    mu = 0.0          # fractional phase estimate
    integrator = 0.0  # loop integrator state
    timing_errors = np.zeros(n_sym)

    for k in range(n_sym):
        base = k * spui + int(round(mu))
        idx_prev = base
        idx_edge = base + spui // 2
        idx_curr = base + spui

        # Bounds check
        if idx_prev < 0 or idx_curr >= len(signal):
            continue

        y_prev = signal[idx_prev]
        y_edge = signal[idx_edge]
        y_curr = signal[idx_curr]

        # Gardner TED
        ted = (y_curr - y_prev) * y_edge
        timing_errors[k] = ted

        # Second-order loop filter
        integrator += ki * ted
        mu += kp * ted + integrator

        # Wrap mu within one UI
        while mu >= spui / 2:
            mu -= spui
        while mu < -spui / 2:
            mu += spui

    # Convert converged fractional phase to integer phase
    # Use the average of the last 25% of mu trajectory
    final_mu = mu
    optimal_phase = int(round(final_mu)) % spui

    return optimal_phase, timing_errors


def downsample(signal, samples_per_ui, phase=0):
    """Downsample oversampled signal to 1 sample per UI.

    Parameters
    ----------
    signal : np.ndarray
        Oversampled waveform.
    samples_per_ui : int
        Oversampling factor.
    phase : int
        Sampling phase offset in [0, samples_per_ui).

    Returns
    -------
    decimated : np.ndarray
        Symbol-rate signal.
    """
    return np.asarray(signal[phase::samples_per_ui], dtype=np.float64)
