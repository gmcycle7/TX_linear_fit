#!/usr/bin/env python3
"""PAM4 SerDes TX Analysis -- Complete Demo Script.

Demonstrates the full analysis pipeline:
  1. Generate PRBS9/PRBSQ9 PAM4 signals
  2. Apply ground-truth TX FIR
  3. Oversample with bandlimited interpolation
  4. Add impairments (AWGN, BW limit, jitter)
  5. CDR and alignment
  6. Extract TX pulse response via linear regression
  7. Compute signal quality metrics
  8. Validate against ground truth
  9. Channel simulation with eye diagram
  10. Generate all visualizations

Usage:
    python demo.py                  # Run full demo
    python demo.py --no-jitter      # Skip jitter
    python demo.py --1sps           # 1 sample/UI mode
    python demo.py --save           # Save plots to files
"""

import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from serdes_tx import (
    generate_pam4_prbs9,
    generate_pam4_prbsq9,
    upsample_pam4,
    GRAY_LEVELS,
    align_by_correlation,
    find_optimal_phase,
    gardner_cdr,
    downsample,
    extract_pulse_response,
    compute_sndr,
    compute_rlm,
    compute_eye_metrics,
    compute_insertion_loss,
    apply_channel,
    default_channel_fir,
    add_awgn,
    apply_isi,
    apply_bandwidth_limit,
    add_jitter,
    plot_eye_diagram,
    plot_pulse_response,
    plot_histogram,
    plot_frequency_response,
    plot_fir_comparison,
    plot_all,
)


# =====================================================================
# Configuration
# =====================================================================

def get_default_config():
    return {
        'pattern': 'prbs9',
        'n_symbols': 511 * 8,      # 4088 PAM4 symbols
        'samples_per_ui': 8,
        'baud_rate_ghz': 28.0,

        'num_taps': 21,

        # Measurement impairments (scope / acquisition)
        'snr_db': 35.0,            # Scope noise floor
        'scope_bw_ratio': None,    # Scope BW / baud (None=wideband; 2.0=realistic)
        'rj_rms_ui': 0.008,        # Random jitter
        'dj_amp_ui': 0.004,        # Deterministic jitter

        # Channel simulation (separate: for eye diagram after extraction)
        'channel_snr_db': 25.0,
        'channel_bw_ratio': 0.5,

        'regularization': 0.0,
        'fit_dc_offset': True,
        'seed': 42,
    }


def make_tx_fir(num_taps=21):
    """Create a realistic TX FIR pulse response (causal, 21 taps)."""
    h = np.zeros(num_taps)
    cursor = num_taps // 2  # tap 10
    h[cursor] = 0.92
    h[cursor - 1] = -0.04   # pre-cursor 1
    h[cursor - 2] = 0.01    # pre-cursor 2
    h[cursor + 1] = -0.12   # post-cursor 1
    h[cursor + 2] = 0.07    # post-cursor 2
    h[cursor + 3] = -0.04   # post-cursor 3
    h[cursor + 4] = 0.02    # post-cursor 4
    h[cursor + 5] = -0.01   # post-cursor 5
    return h


# =====================================================================
# Main Pipeline
# =====================================================================

def run_demo(cfg=None, save_plots=False, show_plots=True):
    if cfg is None:
        cfg = get_default_config()

    rng = np.random.default_rng(cfg['seed'])
    spui = cfg['samples_per_ui']
    num_taps = cfg['num_taps']

    print("=" * 65)
    print("  PAM4 SerDes TX Analysis")
    print("=" * 65)

    # -- 1. Generate PAM4 signal ----------------------------------------
    print("\n[1] Generating PAM4 signal...")
    if cfg['pattern'] == 'prbsq9':
        symbols, indices = generate_pam4_prbsq9(length=cfg['n_symbols'])
    else:
        symbols, indices = generate_pam4_prbs9(length=cfg['n_symbols'])

    counts = {v: int(np.sum(symbols == v)) for v in [-3, -1, 1, 3]}
    print(f"    {cfg['pattern'].upper()}, {cfg['n_symbols']} symbols")
    print(f"    Level counts: {counts}")

    # -- 2. Apply TX FIR -----------------------------------------------
    print("\n[2] Applying TX FIR (ground truth)...")
    h_true = make_tx_fir(num_taps)
    cursor_pos = num_taps // 2
    print(f"    {num_taps} taps, main cursor at [{cursor_pos}] = {h_true[cursor_pos]:.4f}")
    print(f"    Pre-cursor [{cursor_pos-1}] = {h_true[cursor_pos-1]:.4f}")
    print(f"    Post-cursor [{cursor_pos+1}] = {h_true[cursor_pos+1]:.4f}")

    # Ideal TX output at symbol rate
    tx_1sps = np.convolve(symbols, h_true, mode='full')[:len(symbols)]

    # -- 3. Oversample and add measurement impairments ------------------
    if spui > 1:
        print(f"\n[3] Oversampling ({spui}x) and adding impairments...")

        # Bandlimited interpolation (no ZOH ISI artifacts)
        tx_os = upsample_pam4(tx_1sps, spui, method='interp')

        # Scope bandwidth limitation (optional)
        if cfg.get('scope_bw_ratio') and spui >= 2:
            tx_os = apply_bandwidth_limit(tx_os, spui,
                                          bw_ratio=cfg['scope_bw_ratio'])
            print(f"    Scope BW: {cfg['scope_bw_ratio']:.1f}x baud rate")

        # Jitter
        if cfg.get('rj_rms_ui', 0) > 0 or cfg.get('dj_amp_ui', 0) > 0:
            tx_os, _ = add_jitter(
                tx_os, spui,
                rj_rms_ui=cfg.get('rj_rms_ui', 0),
                dj_amp_ui=cfg.get('dj_amp_ui', 0),
                rng=rng,
            )
            print(f"    Jitter: RJ={cfg.get('rj_rms_ui',0)*100:.1f}% RMS, "
                  f"DJ={cfg.get('dj_amp_ui',0)*100:.1f}% pk")

        # Scope noise
        if cfg['snr_db'] is not None:
            tx_os, _ = add_awgn(tx_os, snr_db=cfg['snr_db'], rng=rng)
            print(f"    AWGN: SNR = {cfg['snr_db']:.1f} dB")

        print(f"    Signal: {len(tx_os)} samples")

        # -- 4. CDR and alignment ------------------------------------
        print("\n[4] CDR and alignment...")

        # Max-eye CDR (uses cross-correlation, handles TX FIR delay)
        phase, cdr_metric = find_optimal_phase(tx_os, spui, symbols)
        print(f"    Optimal phase: {phase}/{spui} (metric = {cdr_metric:.4f})")

        # Gardner CDR for comparison
        try:
            g_phase, _ = gardner_cdr(tx_os, spui)
            print(f"    Gardner CDR:   {g_phase}/{spui}")
        except Exception as e:
            print(f"    Gardner CDR:   skipped ({e})")

        # Downsample
        rx_1sps = downsample(tx_os, spui, phase)
    else:
        print("\n[3] 1 sample/UI mode (no CDR needed)...")
        tx_os = tx_1sps.copy()
        if cfg['snr_db'] is not None:
            rx_1sps, _ = add_awgn(tx_1sps, snr_db=cfg['snr_db'], rng=rng)
            print(f"    AWGN: SNR = {cfg['snr_db']:.1f} dB")
        else:
            rx_1sps = tx_1sps.copy()
        phase = 0

    # Align with known pattern via cross-correlation
    rx_aligned, lag, corr_peak = align_by_correlation(rx_1sps, tx_1sps)
    print(f"    Alignment: lag = {lag}, correlation = {corr_peak:.4f}")

    symbols_aligned = symbols[lag:] if lag >= 0 else symbols
    min_len = min(len(rx_aligned), len(symbols_aligned))
    rx_aligned = rx_aligned[:min_len]
    symbols_aligned = symbols_aligned[:min_len]
    print(f"    Usable symbols: {min_len}")

    # -- 5. Pulse response extraction -----------------------------------
    print("\n[5] Extracting TX pulse response...")

    pulse_est, info = extract_pulse_response(
        rx_aligned, symbols_aligned,
        num_taps=num_taps,
        fit_dc_offset=cfg.get('fit_dc_offset', False),
        regularization=cfg.get('regularization', 0.0),
    )
    cursor_est = info['cursor_pos']

    print(f"    Condition number: {info['condition_number']:.1f}")
    print(f"    DC offset: {info['dc_offset']:.6f}")
    print(f"    Main cursor [{cursor_est}]: extracted={pulse_est[cursor_est]:.6f}, "
          f"true={h_true[cursor_est]:.6f}")

    # -- 6. Metrics -----------------------------------------------------
    print("\n[6] Signal quality metrics...")

    # SNDR
    sndr, error = compute_sndr(
        rx_aligned[info['valid_slice']],
        info['signal_reconstructed'],
    )
    print(f"    SNDR = {sndr:.2f} dB")

    # RLM -- shift symbols by FIR delay so rx[n] pairs with symbols[n-cursor]
    cp = cursor_est
    rx_for_rlm = rx_aligned[cp:]
    syms_for_rlm = symbols_aligned[:len(rx_for_rlm)]
    rlm, levels, separations = compute_rlm(rx_for_rlm, syms_for_rlm)
    print(f"    RLM  = {rlm:.4f} (ideal = 1.0)")
    print(f"    Measured levels: [{', '.join(f'{v:.4f}' for v in levels)}]")

    # Insertion loss
    freq_il, il_db, H_il = compute_insertion_loss(
        pulse_est, baud_rate_ghz=cfg['baud_rate_ghz'],
    )
    nyquist = cfg['baud_rate_ghz'] / 2
    idx_nyq = np.argmin(np.abs(freq_il - nyquist))
    print(f"    IL at Nyquist ({nyquist:.1f} GHz): {il_db[idx_nyq]:.2f} dB")

    # Eye metrics (from oversampled signal used for extraction)
    if spui >= 4:
        # Recompute oversampled aligned signal for eye metrics
        os_start = max(0, lag * spui + phase) if lag >= 0 else phase
        rx_os_for_eye = tx_os[os_start:]
        eye = compute_eye_metrics(rx_os_for_eye, spui, phase=phase)
        print(f"    Eye height (worst): {eye['eye_height_worst']:.4f}")
        if eye['eye_widths'] is not None:
            print(f"    Eye width  (worst): {eye['eye_width_worst']:.3f} UI")
        print(f"    Eye linearity: {eye['eye_linearity']:.4f}")
    else:
        rx_os_for_eye = rx_aligned
        eye = None

    # -- 7. Validation against ground truth -----------------------------
    print("\n[7] FIR extraction validation...")

    fir_error = pulse_est - h_true
    rms_error = np.sqrt(np.mean(fir_error ** 2))
    max_error = np.max(np.abs(fir_error))
    cursor_error = abs(pulse_est[cursor_pos] - h_true[cursor_pos])

    print(f"    RMS error:    {rms_error:.6f}")
    print(f"    Max error:    {max_error:.6f}")
    print(f"    Cursor error: {cursor_error:.6f}")

    print(f"\n    {'Tap':>4s} {'True':>10s} {'Extracted':>10s} {'Error':>10s}")
    print(f"    {'---':>4s} {'----------':>10s} {'----------':>10s} {'----------':>10s}")
    for i in range(num_taps):
        if abs(h_true[i]) > 1e-6 or abs(pulse_est[i]) > 1e-4:
            marker = '  <-- cursor' if i == cursor_pos else ''
            print(f"    {i - cursor_pos:>+4d} {h_true[i]:>10.6f} "
                  f"{pulse_est[i]:>10.6f} {fir_error[i]:>10.6f}{marker}")

    # -- 8. Channel simulation for eye diagram --------------------------
    print("\n[8] Channel simulation (for eye diagram)...")

    # Build channel-impaired signal using ZOH for realistic eye shape
    tx_os_zoh = upsample_pam4(tx_1sps, spui, method='zoh')
    ch_signal = apply_channel(
        tx_os_zoh, samples_per_ui=spui,
        channel_fir=default_channel_fir(),
        bw_ratio=cfg.get('channel_bw_ratio', 0.5),
        snr_db=cfg.get('channel_snr_db', 25.0),
        rng=np.random.default_rng(cfg['seed'] + 1),
    )
    print(f"    Channel FIR: {len(default_channel_fir())} taps")
    print(f"    Channel BW: {cfg.get('channel_bw_ratio', 0.5):.0%} of baud rate")
    print(f"    Channel SNR: {cfg.get('channel_snr_db', 25.0):.1f} dB")

    # -- 9. Visualization -----------------------------------------------
    print("\n[9] Generating plots...")

    # Combined dashboard (use delay-shifted symbols for histogram)
    fig_all = plot_all(
        ch_signal, rx_for_rlm, pulse_est,
        symbols=syms_for_rlm,
        samples_per_ui=spui,
        baud_rate_ghz=cfg['baud_rate_ghz'],
        ground_truth_fir=h_true,
        cursor_pos=cursor_pos,
        suptitle=f"PAM4 TX Analysis -- {cfg['pattern'].upper()}, "
                 f"{cfg['baud_rate_ghz']:.0f} Gbaud, {spui}x OS",
    )
    if save_plots:
        fig_all.savefig('pam4_dashboard.png', dpi=150, bbox_inches='tight')
        print("    Saved: pam4_dashboard.png")

    # FIR comparison
    fig_fir, _ = plot_fir_comparison(pulse_est, h_true, cursor_pos=cursor_pos)
    if save_plots:
        fig_fir.savefig('fir_comparison.png', dpi=150, bbox_inches='tight')
        print("    Saved: fir_comparison.png")

    # Eye diagram (channel-impaired)
    if spui >= 2:
        fig_eye, _ = plot_eye_diagram(
            ch_signal, spui, num_ui=2,
            title=f"PAM4 Eye (after channel) -- {cfg['baud_rate_ghz']:.0f} Gbaud",
        )
        if save_plots:
            fig_eye.savefig('eye_diagram.png', dpi=150, bbox_inches='tight')
            print("    Saved: eye_diagram.png")

    if show_plots:
        plt.show()

    # -- Results --------------------------------------------------------
    results = {
        'pulse_extracted': pulse_est,
        'pulse_true': h_true,
        'sndr_db': sndr,
        'rlm': rlm,
        'levels': levels,
        'eye_metrics': eye,
        'insertion_loss_freq': freq_il,
        'insertion_loss_db': il_db,
        'fir_rms_error': rms_error,
        'fir_max_error': max_error,
        'alignment_lag': lag,
        'cdr_phase': phase,
        'extraction_info': info,
    }

    print("\n" + "=" * 65)
    print("  Done.")
    print("=" * 65)
    return results


# =====================================================================
# 1-sps Demo
# =====================================================================

def run_1sps_demo(save_plots=False, show_plots=True):
    cfg = get_default_config()
    cfg['samples_per_ui'] = 1
    cfg['scope_bw_ratio'] = None
    cfg['rj_rms_ui'] = 0.0
    cfg['dj_amp_ui'] = 0.0
    return run_demo(cfg, save_plots=save_plots, show_plots=show_plots)


# =====================================================================
# CLI
# =====================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PAM4 SerDes TX Analysis Demo')
    parser.add_argument('--1sps', dest='one_sps', action='store_true',
                        help='Run in 1 sample/UI mode')
    parser.add_argument('--no-jitter', dest='no_jitter', action='store_true',
                        help='Disable jitter')
    parser.add_argument('--snr', type=float, default=None, help='Override SNR (dB)')
    parser.add_argument('--spui', type=int, default=None, help='Samples per UI')
    parser.add_argument('--pattern', choices=['prbs9', 'prbsq9'], default=None)
    parser.add_argument('--save', action='store_true', help='Save plots to PNG')
    parser.add_argument('--no-show', action='store_true', help='Skip interactive display')

    args = parser.parse_args()

    if args.one_sps:
        run_1sps_demo(save_plots=args.save, show_plots=not args.no_show)
    else:
        cfg = get_default_config()
        if args.no_jitter:
            cfg['rj_rms_ui'] = 0.0
            cfg['dj_amp_ui'] = 0.0
        if args.snr is not None:
            cfg['snr_db'] = args.snr
        if args.spui is not None:
            cfg['samples_per_ui'] = args.spui
        if args.pattern is not None:
            cfg['pattern'] = args.pattern
        run_demo(cfg, save_plots=args.save, show_plots=not args.no_show)
