"""Visualization functions for PAM4 SerDes TX analysis.

Provides eye diagram, pulse response, histogram, frequency response,
and FIR comparison plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.fft import rfft, rfftfreq


# White-background eye diagram: white -> blue -> red -> yellow
_EYE_COLORS = ['#FFFFFF', '#DDEEFF', '#66AADD', '#2266BB',
               '#CC3333', '#FF8800', '#FFDD00']
_EYE_CMAP = LinearSegmentedColormap.from_list('eye_wb', _EYE_COLORS, N=256)


def plot_eye_diagram(signal, samples_per_ui, num_ui=2, title='PAM4 Eye Diagram',
                     ax=None, density=True, color='#00BFFF', alpha=0.02):
    """Plot PAM4 eye diagram with smooth interpolated traces.

    Parameters
    ----------
    signal : np.ndarray
        Oversampled waveform (samples_per_ui >= 2 recommended).
    samples_per_ui : int
        Oversampling factor.
    num_ui : int
        Number of UI per trace (1 or 2).
    title : str
        Plot title.
    ax : matplotlib Axes or None
        Axes to plot on (creates new figure if None).
    density : bool
        If True, plot as 2D histogram (heatmap). If False, overlaid traces.
    color : str
        Trace color (only used when density=False).
    alpha : float
        Trace transparency (only used when density=False).

    Returns
    -------
    fig : matplotlib Figure
    ax : matplotlib Axes
    """
    from scipy.ndimage import zoom as _ndzoom

    signal = np.asarray(signal, dtype=np.float64)
    spui = int(samples_per_ui)
    seg_len = num_ui * spui

    if seg_len < 2:
        raise ValueError("Need at least 2 samples per segment for eye diagram")

    # ── Phase alignment (max-eye) ──
    from .cdr import _eye_opening_metric
    best_ph, n_sym = 0, len(signal) // spui
    best_m = -np.inf
    for ph in range(spui):
        m = _eye_opening_metric(signal[ph::spui][:n_sym])
        if m > best_m:
            best_m, best_ph = m, ph
    fold_start = (best_ph + spui // 2) % spui
    signal = signal[fold_start:]

    n_segs = len(signal) // seg_len
    if n_segs < 10:
        raise ValueError(f"Not enough data for eye diagram ({n_segs} segments)")

    # Fold signal into segments
    segments = signal[:n_segs * seg_len].reshape(n_segs, seg_len)

    # ── Cubic interpolation to ~128 pts/UI for smooth curves ──
    target_ppui = 128
    interp_f = max(1, target_ppui // spui)
    if interp_f > 1:
        segments = _ndzoom(segments, (1, interp_f), order=3)
    seg_hr = segments.shape[1]

    t_ui = np.linspace(0, num_ui, seg_hr, endpoint=False)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.get_figure()

    if density and n_segs > 100:
        t_rep = np.tile(t_ui, n_segs)
        v_rep = segments.ravel()

        v_min, v_max = np.percentile(v_rep, [0.1, 99.9])
        v_range = v_max - v_min
        v_min -= 0.08 * v_range
        v_max += 0.08 * v_range

        ax.hist2d(t_rep, v_rep,
                  bins=[min(seg_hr, 512), 400],
                  range=[[0, num_ui], [v_min, v_max]],
                  cmap=_EYE_CMAP, cmin=1)
        ax.set_facecolor('white')
    else:
        ax.set_facecolor('#1a1a2e')
        for i in range(min(n_segs, 2000)):
            ax.plot(t_ui, segments[i], color=color, alpha=alpha,
                    linewidth=0.5, rasterized=True)

    ax.set_xlabel('Time (UI)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.set_xlim(0, num_ui)
    fig.tight_layout()

    return fig, ax


def plot_pulse_response(pulse, samples_per_ui=1, cursor_pos=None,
                        baud_rate_ghz=None, title='TX Pulse Response',
                        ax=None):
    """Plot pulse response with main cursor highlighted.

    Parameters
    ----------
    pulse : np.ndarray
        Pulse response taps.
    samples_per_ui : int
        Samples per UI of the pulse (1 = symbol-rate).
    cursor_pos : int or None
        Main cursor index. None = index of max absolute value.
    baud_rate_ghz : float or None
        If provided, x-axis is in nanoseconds instead of taps.
    title : str
    ax : matplotlib Axes or None

    Returns
    -------
    fig, ax
    """
    pulse = np.asarray(pulse, dtype=np.float64)
    n_taps = len(pulse)

    if cursor_pos is None:
        cursor_pos = int(np.argmax(np.abs(pulse)))

    if baud_rate_ghz is not None:
        dt = 1.0 / (baud_rate_ghz * samples_per_ui)  # ns
        t = (np.arange(n_taps) - cursor_pos * samples_per_ui) * dt
        xlabel = 'Time (ns)'
    else:
        t = np.arange(n_taps) - cursor_pos
        xlabel = 'Tap index (relative to cursor)'

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    markerline, stemlines, baseline = ax.stem(t, pulse, basefmt='k-')
    plt.setp(stemlines, linewidth=1.2, color='steelblue')
    plt.setp(markerline, markersize=5, color='steelblue')

    # Highlight main cursor
    ax.plot(t[cursor_pos], pulse[cursor_pos], 'ro', markersize=8,
            label=f'Cursor = {pulse[cursor_pos]:.4f}')

    ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_histogram(signal, symbols=None, n_bins=150,
                   title='PAM4 Level Histogram', ax=None):
    """Plot histogram of PAM4 signal amplitudes.

    Parameters
    ----------
    signal : np.ndarray
        1 sample/UI PAM4 signal.
    symbols : np.ndarray or None
        Known symbols for color-coding by level.
    n_bins : int
        Number of histogram bins.
    title : str
    ax : matplotlib Axes or None

    Returns
    -------
    fig, ax
    """
    signal = np.asarray(signal, dtype=np.float64)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    level_colors = ['#e74c3c', '#e67e22', '#2ecc71', '#3498db']
    level_labels = ['Level -3', 'Level -1', 'Level +1', 'Level +3']
    ideal_levels = [-3.0, -1.0, 1.0, 3.0]

    if symbols is not None:
        symbols = np.asarray(symbols, dtype=np.float64)
        min_len = min(len(signal), len(symbols))
        signal = signal[:min_len]
        symbols = symbols[:min_len]

        for i, lev in enumerate(ideal_levels):
            mask = np.abs(symbols - lev) < 0.5
            if np.any(mask):
                ax.hist(signal[mask], bins=n_bins, alpha=0.7,
                        color=level_colors[i], label=level_labels[i],
                        density=True)
    else:
        ax.hist(signal, bins=n_bins, alpha=0.7, color='steelblue',
                density=True, label='All samples')

    # Mark estimated levels
    from .pam4 import estimate_levels
    try:
        levels = estimate_levels(signal)
        for lev in levels:
            ax.axvline(lev, color='white', linewidth=1, linestyle='--', alpha=0.7)
    except Exception:
        pass

    ax.set_xlabel('Amplitude')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_frequency_response(pulse, samples_per_ui=1, baud_rate_ghz=1.0,
                            n_fft=None, title='TX Frequency Response',
                            ax=None):
    """Plot magnitude and insertion loss of pulse response.

    Parameters
    ----------
    pulse : np.ndarray
        Pulse response taps.
    samples_per_ui : int
        Samples per UI of the pulse.
    baud_rate_ghz : float
        Baud rate in GHz.
    n_fft : int or None
        FFT size.
    title : str
    ax : matplotlib Axes or None

    Returns
    -------
    fig, ax
    """
    pulse = np.asarray(pulse, dtype=np.float64)
    fs = baud_rate_ghz * samples_per_ui

    if n_fft is None:
        n_fft = max(1024, 2 ** int(np.ceil(np.log2(len(pulse) * 8))))

    H = rfft(pulse, n=n_fft)
    freq = rfftfreq(n_fft, d=1.0 / fs)
    H_mag = np.abs(H)
    H_dc = H_mag[0] if H_mag[0] > 1e-30 else 1.0

    mag_db = 20.0 * np.log10(np.maximum(H_mag / H_dc, 1e-30))

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.get_figure()

    ax.plot(freq, mag_db, 'b-', linewidth=1.5)
    ax.axhline(-3, color='red', linewidth=0.8, linestyle='--',
               label='-3 dB', alpha=0.7)
    ax.axvline(baud_rate_ghz / 2, color='orange', linewidth=0.8,
               linestyle=':', label=f'Nyquist ({baud_rate_ghz/2:.1f} GHz)')

    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Magnitude (dB)')
    ax.set_title(title)
    ax.set_xlim(0, fs / 2)
    ax.set_ylim(bottom=max(-40, np.min(mag_db) - 3))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    return fig, ax


def plot_fir_comparison(extracted, ground_truth, cursor_pos=None,
                        title='FIR Comparison: Extracted vs Ground Truth'):
    """Plot extracted FIR against ground truth with error.

    Parameters
    ----------
    extracted : np.ndarray
        Extracted pulse response.
    ground_truth : np.ndarray
        True pulse response.
    cursor_pos : int or None
        Main cursor position.

    Returns
    -------
    fig, axes : tuple of (Figure, array of 2 Axes)
    """
    extracted = np.asarray(extracted, dtype=np.float64)
    ground_truth = np.asarray(ground_truth, dtype=np.float64)

    # Pad shorter to match lengths
    max_len = max(len(extracted), len(ground_truth))
    ext_padded = np.zeros(max_len)
    gt_padded = np.zeros(max_len)
    ext_padded[:len(extracted)] = extracted
    gt_padded[:len(ground_truth)] = ground_truth

    if cursor_pos is None:
        cursor_pos = int(np.argmax(np.abs(gt_padded)))

    taps = np.arange(max_len) - cursor_pos
    error = ext_padded - gt_padded

    fig, axes = plt.subplots(2, 1, figsize=(8, 6), height_ratios=[3, 1],
                             sharex=True)

    # Top: comparison
    axes[0].stem(taps - 0.1, gt_padded, linefmt='b-', markerfmt='bo',
                 basefmt='k-', label='Ground Truth')
    axes[0].stem(taps + 0.1, ext_padded, linefmt='r-', markerfmt='r^',
                 basefmt='k-', label='Extracted')
    axes[0].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[0].set_ylabel('Amplitude')
    axes[0].set_title(title)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bottom: error
    axes[1].stem(taps, error, linefmt='g-', markerfmt='gs', basefmt='k-')
    axes[1].axhline(0, color='gray', linewidth=0.5, linestyle='--')
    axes[1].set_xlabel('Tap index (relative to cursor)')
    axes[1].set_ylabel('Error')
    axes[1].grid(True, alpha=0.3)

    rms_err = np.sqrt(np.mean(error ** 2))
    axes[1].set_title(f'Extraction Error (RMS = {rms_err:.6f})')

    fig.tight_layout()
    return fig, axes


def plot_all(signal_oversampled, signal_1sps, pulse, symbols=None,
             samples_per_ui=8, baud_rate_ghz=1.0, ground_truth_fir=None,
             cursor_pos=None, suptitle='PAM4 TX Analysis'):
    """Generate combined dashboard with all visualization panels.

    Parameters
    ----------
    signal_oversampled : np.ndarray
        Oversampled waveform for eye diagram.
    signal_1sps : np.ndarray
        Symbol-rate signal for histogram and metrics.
    pulse : np.ndarray
        Extracted pulse response.
    symbols : np.ndarray or None
        Known symbol sequence.
    samples_per_ui : int
    baud_rate_ghz : float
    ground_truth_fir : np.ndarray or None
        If provided, adds FIR comparison panel.
    cursor_pos : int or None
    suptitle : str

    Returns
    -------
    fig : matplotlib Figure
    """
    n_rows = 3 if ground_truth_fir is None else 3
    n_cols = 2
    fig = plt.figure(figsize=(14, 12))

    # 1. Eye diagram (top-left)
    ax1 = fig.add_subplot(n_rows, n_cols, 1)
    if samples_per_ui >= 2:
        plot_eye_diagram(signal_oversampled, samples_per_ui, ax=ax1)
    else:
        ax1.text(0.5, 0.5, '(Eye diagram requires\noversampled data)',
                 ha='center', va='center', transform=ax1.transAxes)
        ax1.set_title('PAM4 Eye Diagram')

    # 2. Pulse response (top-right)
    ax2 = fig.add_subplot(n_rows, n_cols, 2)
    plot_pulse_response(pulse, cursor_pos=cursor_pos, ax=ax2)

    # 3. Histogram (mid-left)
    ax3 = fig.add_subplot(n_rows, n_cols, 3)
    plot_histogram(signal_1sps, symbols=symbols, ax=ax3)

    # 4. Frequency response (mid-right)
    ax4 = fig.add_subplot(n_rows, n_cols, 4)
    plot_frequency_response(pulse, baud_rate_ghz=baud_rate_ghz, ax=ax4)

    # 5-6. FIR comparison or waveform
    if ground_truth_fir is not None:
        ax5 = fig.add_subplot(n_rows, n_cols, 5)
        ax6 = fig.add_subplot(n_rows, n_cols, 6)
        _plot_fir_comparison_on_axes(pulse, ground_truth_fir, cursor_pos,
                                     ax5, ax6)
    else:
        # Show a waveform snippet
        ax5 = fig.add_subplot(n_rows, 1, 3)
        n_show = min(100 * samples_per_ui, len(signal_oversampled))
        t = np.arange(n_show) / samples_per_ui
        ax5.plot(t, signal_oversampled[:n_show], 'b-', linewidth=0.8)
        ax5.set_xlabel('Time (UI)')
        ax5.set_ylabel('Amplitude')
        ax5.set_title('Waveform Snippet')
        ax5.grid(True, alpha=0.3)

    fig.suptitle(suptitle, fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig


def _plot_fir_comparison_on_axes(extracted, ground_truth, cursor_pos, ax_top, ax_bot):
    """Helper to plot FIR comparison on pre-existing axes."""
    max_len = max(len(extracted), len(ground_truth))
    ext_padded = np.zeros(max_len)
    gt_padded = np.zeros(max_len)
    ext_padded[:len(extracted)] = extracted
    gt_padded[:len(ground_truth)] = ground_truth

    if cursor_pos is None:
        cursor_pos = int(np.argmax(np.abs(gt_padded)))

    taps = np.arange(max_len) - cursor_pos
    error = ext_padded - gt_padded

    ax_top.stem(taps - 0.1, gt_padded, linefmt='b-', markerfmt='bo',
                basefmt='k-', label='Ground Truth')
    ax_top.stem(taps + 0.1, ext_padded, linefmt='r-', markerfmt='r^',
                basefmt='k-', label='Extracted')
    ax_top.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax_top.set_ylabel('Amplitude')
    ax_top.set_title('FIR Comparison')
    ax_top.legend(fontsize=8)
    ax_top.grid(True, alpha=0.3)

    rms_err = np.sqrt(np.mean(error ** 2))
    ax_bot.stem(taps, error, linefmt='g-', markerfmt='gs', basefmt='k-')
    ax_bot.axhline(0, color='gray', linewidth=0.5, linestyle='--')
    ax_bot.set_xlabel('Tap (relative to cursor)')
    ax_bot.set_ylabel('Error')
    ax_bot.set_title(f'Error (RMS = {rms_err:.6f})')
    ax_bot.grid(True, alpha=0.3)
