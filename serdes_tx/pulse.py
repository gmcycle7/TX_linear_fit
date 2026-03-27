"""TX pulse response extraction via linear regression.

Estimates the FIR (pulse response) of the TX path by solving:
    y = A @ h
where y is the measured signal, A is a convolution matrix built from
the known data pattern, and h is the unknown pulse response.

The model is the standard causal convolution:
    y[n] = sum_{k=0}^{L-1} h[k] * a[n-k]

where a[n] are known PAM4 symbols and h[k] is the pulse response.
"""

import numpy as np


def build_convolution_matrix(symbols, num_taps):
    """Build the Toeplitz matrix for causal convolution.

    Models: y[n] = sum_{k=0}^{L-1} h[k] * a[n-k]
    Valid for n = L-1, ..., N-1 where L = num_taps, N = len(symbols).

    Parameters
    ----------
    symbols : np.ndarray
        Known PAM4 symbol sequence (1 sample/UI), length N.
    num_taps : int
        Number of FIR taps (L).

    Returns
    -------
    A : np.ndarray, shape (N - L + 1, L)
        Convolution matrix. Row i corresponds to output index n = i + L - 1.
    valid_slice : slice
        Slice into the signal array for the valid output range.
    """
    N = len(symbols)
    L = num_taps
    n_valid = N - L + 1

    if n_valid < L:
        raise ValueError(
            f"Not enough symbols ({N}) for {L} taps. "
            f"Need at least {2 * L - 1} symbols."
        )

    A = np.empty((n_valid, L), dtype=np.float64)
    for k in range(L):
        # A[i, k] = a[n - k] = a[i + L - 1 - k]
        start = L - 1 - k
        A[:, k] = symbols[start: start + n_valid]

    valid_slice = slice(L - 1, L - 1 + n_valid)
    return A, valid_slice


def extract_pulse_response(signal, symbols, num_taps=21, cursor_pos=None,
                           fit_dc_offset=False, regularization=0.0):
    """Extract TX pulse response via least-squares regression.

    Solves y = A @ h (+ optional DC offset) using least squares,
    where A is the convolution matrix built from known symbols.

    Parameters
    ----------
    signal : np.ndarray
        Measured waveform at 1 sample/UI, aligned with symbols.
    symbols : np.ndarray
        Known PAM4 symbol sequence, same length as signal.
    num_taps : int
        Number of FIR taps to extract (default 21).
    cursor_pos : int or None
        Main cursor index for reporting purposes.
        None defaults to the index of the largest extracted tap.
    fit_dc_offset : bool
        If True, also estimate a DC offset term.
    regularization : float
        Tikhonov regularization parameter (0 = none). Helps with
        ill-conditioned matrices from short or correlated data.

    Returns
    -------
    pulse : np.ndarray, shape (num_taps,)
        Estimated pulse response (causal: h[0], h[1], ..., h[L-1]).
    info : dict
        'residual' : np.ndarray - fit residuals
        'dc_offset' : float - estimated DC offset (0 if not fitted)
        'signal_reconstructed' : np.ndarray - A @ pulse
        'condition_number' : float - condition number of A
        'cursor_pos' : int - main cursor position (index of max tap)
        'valid_slice' : slice - which signal samples were used
    """
    signal = np.asarray(signal, dtype=np.float64)
    symbols = np.asarray(symbols, dtype=np.float64)

    min_len = min(len(signal), len(symbols))
    signal = signal[:min_len]
    symbols = symbols[:min_len]

    A, valid_slice = build_convolution_matrix(symbols, num_taps)
    y = signal[valid_slice]

    # Optionally add DC offset column
    if fit_dc_offset:
        ones_col = np.ones((A.shape[0], 1), dtype=np.float64)
        A_aug = np.hstack([A, ones_col])
    else:
        A_aug = A

    # Solve with optional Tikhonov regularization
    if regularization > 0:
        ATA = A_aug.T @ A_aug
        reg_matrix = regularization * np.eye(ATA.shape[0])
        if fit_dc_offset:
            reg_matrix[-1, -1] = 0  # don't regularize DC offset
        x, _, _, _ = np.linalg.lstsq(ATA + reg_matrix, A_aug.T @ y, rcond=None)
    else:
        x, _, _, _ = np.linalg.lstsq(A_aug, y, rcond=None)

    if fit_dc_offset:
        pulse = x[:-1]
        dc_offset = float(x[-1])
    else:
        pulse = x
        dc_offset = 0.0

    # Determine cursor position
    if cursor_pos is None:
        cursor_pos = int(np.argmax(np.abs(pulse)))

    # Compute diagnostics
    y_hat = A @ pulse + dc_offset
    residual = y - y_hat
    cond = np.linalg.cond(A)

    info = {
        'residual': residual,
        'dc_offset': dc_offset,
        'signal_reconstructed': y_hat,
        'condition_number': cond,
        'cursor_pos': cursor_pos,
        'valid_slice': valid_slice,
    }

    return pulse, info


def pulse_to_step_response(pulse):
    """Convert pulse response to step response (cumulative sum).

    Parameters
    ----------
    pulse : np.ndarray
        Pulse response taps.

    Returns
    -------
    step : np.ndarray
        Step response.
    """
    return np.cumsum(pulse)
