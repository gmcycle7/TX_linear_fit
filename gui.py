#!/usr/bin/env python3
"""PAM4 SerDes TX Analyzer — GUI

Workflow
--------
1.  Generate **ideal PRBS** (ZOH, near-zero rise time, no ISI).
2.  Pass it through a **loss channel** (user-defined IL at Nyquist)
    to produce the **measured waveform** (has ISI).
3.  **Linear-fit** the measured waveform against the known ideal
    pattern to extract the system **pulse response**.
4.  FFT the pulse response → **AC response**; should match the
    channel that was applied.
5.  Convolve the ideal pattern with the extracted pulse → **fitted
    waveform**; should closely match the measured waveform.

Run:  python gui.py
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle as _MplRect

from serdes_tx import (
    generate_pam4_prbs9, generate_pam4_prbsq9,
    upsample_pam4, GRAY_LEVELS,
    align_by_correlation, find_optimal_phase, downsample,
    extract_pulse_response,
    compute_sndr, compute_rlm, compute_eye_metrics, compute_insertion_loss,
    add_awgn, add_jitter,
    make_channel_from_il,
)
from serdes_tx.pulse import build_convolution_matrix
from serdes_tx.pam4 import estimate_levels

EYE_CMAP = LinearSegmentedColormap.from_list(
    "eye_wb", ["#FFFFFF", "#DDEEFF", "#66AADD", "#2266BB",
               "#CC3333", "#FF8800", "#FFDD00"], N=256)

P  = dict(padx=4, pady=2)
EW = 8


# =====================================================================
class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("PAM4 SerDes TX Analyzer")
        self.geometry("1500x960")
        self.protocol("WM_DELETE_WINDOW", self.destroy)

        # ---- data slots ----
        self.symbols       = None   # PAM4 pattern (1 sps)
        self.ideal_os      = None   # oversampled ideal PRBS (ZOH)
        self.ch_fir        = None   # channel FIR (oversampled)
        self.ch_freq       = None   # channel freq axis (norm)
        self.ch_H_db       = None   # channel |H| in dB
        self.meas_os       = None   # oversampled measured waveform
        self.meas_1sps     = None   # 1-sps measured (CDR + aligned)
        self.sym_aligned   = None   # symbols aligned to measured
        self.cdr_phase     = 0
        self.align_lag     = 0
        self.pulse_est     = None   # extracted pulse response
        self.true_pulse    = None   # ground-truth pulse (impulse test)
        self.fitted_1sps   = None   # conv(symbols, pulse)
        self.ext_info      = None

        self._build()

    # =================================================================
    #  BUILD UI
    # =================================================================
    def _build(self):
        pw = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        pw.pack(fill=tk.BOTH, expand=True)
        left = self._mk_left(pw);  pw.add(left, weight=0)
        right = ttk.Frame(pw);     pw.add(right, weight=1)
        rpw = ttk.PanedWindow(right, orient=tk.VERTICAL)
        rpw.pack(fill=tk.BOTH, expand=True)
        self.nb = ttk.Notebook(rpw); rpw.add(self.nb, weight=5)
        bot = ttk.Frame(rpw);        rpw.add(bot, weight=1)
        self._mk_tabs()
        self._mk_bot(bot)

    # ---- helpers for rows ---
    def _row(self, lf, r, label, var, **kw):
        ttk.Label(lf, text=label).grid(row=r, column=0, sticky=tk.W, **P)
        w = kw.pop("widget", "entry")
        if w == "combo":
            wid = ttk.Combobox(lf, textvariable=var, width=EW,
                               state="readonly", **kw)
        elif w == "check":
            wid = ttk.Checkbutton(lf, variable=var, **kw)
        else:
            wid = ttk.Entry(lf, textvariable=var, width=EW)
        wid.grid(row=r, column=1, sticky=tk.W, **P)

    # ---- left panel ---
    def _mk_left(self, parent):
        outer = ttk.Frame(parent, width=290)
        cv = tk.Canvas(outer, width=280)
        sb = ttk.Scrollbar(outer, orient=tk.VERTICAL, command=cv.yview)
        inner = ttk.Frame(cv)
        inner.bind("<Configure>",
                    lambda e: cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0, 0), window=inner, anchor="nw")
        cv.configure(yscrollcommand=sb.set)
        cv.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

        # -- Signal --
        lf = ttk.LabelFrame(inner, text="Signal Generation")
        lf.pack(fill=tk.X, **P)
        self.v_pat = tk.StringVar(value="PRBSQ9")
        self._row(lf, 0, "Pattern:", self.v_pat,
                  widget="combo", values=["PRBS9","PRBSQ9"])
        self.v_nsym = tk.IntVar(value=4088)
        self._row(lf, 1, "Symbols:", self.v_nsym)
        self.v_spui = tk.IntVar(value=64)
        self._row(lf, 2, "Samp/UI:", self.v_spui,
                  widget="combo", values=["8","16","32","64","128"])
        self.v_baud = tk.DoubleVar(value=28.0)
        self._row(lf, 3, "Baud (GHz):", self.v_baud)

        # -- Channel --
        lf2 = ttk.LabelFrame(inner, text="Channel (System Under Test)")
        lf2.pack(fill=tk.X, **P)
        self.v_il = tk.DoubleVar(value=-10.0)
        self._row(lf2, 0, "IL @ Nyq (dB):", self.v_il)
        self.v_loss_model = tk.StringVar(value="linear")
        self._row(lf2, 1, "Loss model:", self.v_loss_model,
                  widget="combo", values=["linear","sqrt"])
        self.v_ch_snr = tk.DoubleVar(value=40.0)
        self._row(lf2, 2, "Noise SNR (dB):", self.v_ch_snr)
        self.v_ch_rj = tk.DoubleVar(value=0.5)
        self._row(lf2, 3, "RJ (% UI):", self.v_ch_rj)

        # -- Extraction --
        lf3 = ttk.LabelFrame(inner, text="Pulse Extraction")
        lf3.pack(fill=tk.X, **P)
        self.v_ntaps = tk.IntVar(value=31)
        self._row(lf3, 0, "Num taps:", self.v_ntaps)
        self.v_fitdc = tk.BooleanVar(value=True)
        ttk.Checkbutton(lf3, text="Fit DC offset",
                        variable=self.v_fitdc).grid(
            row=1, column=0, columnspan=2, sticky=tk.W, **P)

        # -- Buttons --
        lf4 = ttk.LabelFrame(inner, text="Actions")
        lf4.pack(fill=tk.X, **P)
        kw = dict(width=26)
        ttk.Button(lf4, text="\u25b6  Run Full Analysis",
                   command=self._run_all, **kw).pack(**P)
        ttk.Separator(lf4).pack(fill=tk.X, pady=4)
        ttk.Button(lf4, text="Load Measured Waveform\u2026",
                   command=self._load, **kw).pack(**P)
        ttk.Button(lf4, text="Export Plots\u2026",
                   command=self._export, **kw).pack(**P)

        return outer

    # ---- tabs ---
    def _mk_tabs(self):
        self.figs, self.cvs = {}, {}
        for name in ["Waveform", "Eye Diagram", "Pulse Response",
                     "AC Response", "Linear Fit", "Histogram"]:
            f = ttk.Frame(self.nb); self.nb.add(f, text=name)
            fig = Figure(figsize=(9, 5), dpi=100)
            c = FigureCanvasTkAgg(fig, master=f)
            NavigationToolbar2Tk(c, f)
            c.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.figs[name] = fig; self.cvs[name] = c
        self._attach_zoom()

    # ---- interactive zoom ---
    def _attach_zoom(self):
        for _, canvas in self.cvs.items():
            st = {"p": None, "r": None}
            def _scr(ev, _s=st):
                ax = ev.inaxes
                if ax is None or ev.xdata is None: return
                f = 0.8 if ev.button == "up" else 1.25
                xl, yl = ax.get_xlim(), ax.get_ylim()
                w2 = (xl[1]-xl[0])*f; h2 = (yl[1]-yl[0])*f
                rx = (ev.xdata-xl[0])/(xl[1]-xl[0])
                ry = (ev.ydata-yl[0])/(yl[1]-yl[0])
                ax.set_xlim(ev.xdata-w2*rx, ev.xdata+w2*(1-rx))
                ax.set_ylim(ev.ydata-h2*ry, ev.ydata+h2*(1-ry))
                ev.canvas.draw_idle()
            def _pr(ev, _s=st):
                if ev.button != 3 or ev.inaxes is None: return
                if ev.dblclick:
                    for a in ev.canvas.figure.get_axes(): a.autoscale()
                    ev.canvas.draw_idle(); return
                _s["p"] = (ev.xdata, ev.ydata, ev.inaxes)
                r = _MplRect((ev.xdata,ev.ydata),0,0,
                             fill=False,ec="red",lw=1.5,ls="--")
                ev.inaxes.add_patch(r); _s["r"] = r
                ev.canvas.draw_idle()
            def _mo(ev, _s=st):
                if _s["p"] is None or _s["r"] is None: return
                x0,y0,ax = _s["p"]
                if ev.inaxes != ax or ev.xdata is None: return
                _s["r"].set_xy((min(x0,ev.xdata),min(y0,ev.ydata)))
                _s["r"].set_width(abs(ev.xdata-x0))
                _s["r"].set_height(abs(ev.ydata-y0))
                ev.canvas.draw_idle()
            def _re(ev, _s=st):
                if ev.button != 3 or _s["p"] is None: return
                x0,y0,ax = _s["p"]
                try:
                    if _s["r"]: _s["r"].remove()
                except Exception: pass
                _s["r"]=None; _s["p"]=None
                if ev.inaxes!=ax or ev.xdata is None:
                    ev.canvas.draw_idle(); return
                x1,y1 = ev.xdata,ev.ydata
                xr,yr = ax.get_xlim(),ax.get_ylim()
                if abs(x1-x0)>0.01*(xr[1]-xr[0]) and \
                   abs(y1-y0)>0.01*(yr[1]-yr[0]):
                    ax.set_xlim(min(x0,x1),max(x0,x1))
                    ax.set_ylim(min(y0,y1),max(y0,y1))
                ev.canvas.draw_idle()
            canvas.mpl_connect("scroll_event",_scr)
            canvas.mpl_connect("button_press_event",_pr)
            canvas.mpl_connect("motion_notify_event",_mo)
            canvas.mpl_connect("button_release_event",_re)

    # ---- bottom panel ---
    def _mk_bot(self, parent):
        mf = ttk.LabelFrame(parent, text="Metrics")
        mf.pack(fill=tk.X, **P)
        self.mlbl = {}
        for i,(txt,key) in enumerate([
                ("SNDR(fit)","sf"),("SNDR(meas vs fitted)","sm"),
                ("RLM","rlm"),("FIR err RMS","fe")]):
            ttk.Label(mf,text=txt+":").grid(row=0,column=2*i,sticky=tk.E,padx=2)
            l = ttk.Label(mf,text="--",width=14,anchor=tk.W,
                          font=("Consolas",10,"bold"))
            l.grid(row=0,column=2*i+1,padx=(0,12))
            self.mlbl[key] = l
        lf = ttk.LabelFrame(parent, text="Log")
        lf.pack(fill=tk.BOTH, expand=True, **P)
        self.log = tk.Text(lf, height=8, wrap=tk.WORD, font=("Consolas",9))
        sb = ttk.Scrollbar(lf, command=self.log.yview)
        self.log.configure(yscrollcommand=sb.set)
        self.log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)

    # =================================================================
    #  HELPERS
    # =================================================================
    def _pr(self, m):
        self.log.insert(tk.END, m+"\n"); self.log.see(tk.END)
        self.update_idletasks()
    def _met(self, k, v): self.mlbl[k].config(text=v)

    # =================================================================
    #  CORE PIPELINE
    # =================================================================
    def _run_all(self):
        self._pr("="*58)
        self._pr("  FULL ANALYSIS")
        self._pr("="*58)
        try:
            self._step1_gen_ideal()
            self._step2_channel()
            self._step3_cdr_align()
            self._step4_linear_fit()
            self._compute_true_pulse()
            self._step5_fitted()
            self._step6_metrics()
            self._draw_all()
            self._pr("="*58)
            self._pr("  DONE")
            self._pr("="*58)
        except Exception as e:
            self._pr(f"ERROR: {e}")
            import traceback; self._pr(traceback.format_exc())

    # ---- Step 1: Ideal PRBS ------------------------------------------
    def _step1_gen_ideal(self):
        self._pr("--- Step 1: Generate Ideal PRBS (no ISI) ---")
        n = self.v_nsym.get()
        if self.v_pat.get() == "PRBSQ9":
            self.symbols, _ = generate_pam4_prbsq9(length=n)
        else:
            self.symbols, _ = generate_pam4_prbs9(length=n)
        spui = self.v_spui.get()
        self.ideal_os = upsample_pam4(self.symbols, spui, method="zoh")
        self._pr(f"  Pattern: {self.v_pat.get()},  {n} symbols")
        self._pr(f"  Oversampled: {len(self.ideal_os)} samples  "
                 f"({spui} samp/UI)")
        self._pr(f"  Rise/fall: ~1/{spui} UI  (ZOH)")

    # ---- Step 2: Apply channel loss -----------------------------------
    def _step2_channel(self):
        self._pr("--- Step 2: Apply loss channel (freq-domain) ---")
        spui = self.v_spui.get()
        il   = float(self.v_il.get())         # e.g. -10
        model = self.v_loss_model.get()
        baud = self.v_baud.get()

        N = len(self.ideal_os)
        X = np.fft.rfft(self.ideal_os)
        f01 = np.fft.rfftfreq(N)              # 0 .. 0.5  (norm to f_s)
        f_baud_nyq = 0.5 / spui              # baud Nyquist normalised

        # Magnitude
        if model == "sqrt":
            ratio = np.sqrt(np.clip(f01 / f_baud_nyq, 0, None))
        else:
            ratio = f01 / f_baud_nyq
        il_db = il * ratio
        il_db[0] = 0.0
        il_db = np.clip(il_db, -80, 0)
        H_mag = 10.0 ** (il_db / 20.0)

        # Linear-phase delay (4 UI) for causality
        delay = 4 * spui
        phase = -2 * np.pi * np.arange(len(f01)) * delay / N
        H = H_mag * np.exp(1j * phase)

        self.meas_os = np.fft.irfft(X * H, N)

        # Store exact channel response for later comparison
        self.ch_freq_ghz = f01 * spui * baud   # freq in GHz
        self.ch_il_db    = il_db                # IL in dB (negative)

        self._pr(f"  IL @ Nyquist ({baud/2:.1f} GHz): {il:.1f} dB")
        self._pr(f"  Loss model: {model}")
        self._pr(f"  Applied in frequency domain (exact)")

        # Optional noise / jitter
        rng = np.random.default_rng(42)
        snr = self.v_ch_snr.get()
        if snr > 0 and snr < 100:
            self.meas_os, _ = add_awgn(self.meas_os, snr_db=snr, rng=rng)
            self._pr(f"  Noise: SNR = {snr:.0f} dB")
        rj = self.v_ch_rj.get() / 100.0
        if rj > 0 and spui >= 2:
            self.meas_os, _ = add_jitter(self.meas_os, spui,
                                          rj_rms_ui=rj, rng=rng)
            self._pr(f"  Jitter: RJ = {rj*100:.1f}% UI")
        self._pr(f"  Measured waveform: {len(self.meas_os)} samples")

    # ---- Step 3: CDR + align ------------------------------------------
    def _step3_cdr_align(self):
        self._pr("--- Step 3: CDR + Alignment ---")
        spui = self.v_spui.get()
        if spui >= 2:
            ph, met = find_optimal_phase(self.meas_os, spui, self.symbols)
            self.cdr_phase = ph
            self._pr(f"  CDR: phase = {ph}/{spui}  metric = {met:.4f}")
            meas_1 = downsample(self.meas_os, spui, ph)
        else:
            self.cdr_phase = 0
            meas_1 = self.meas_os.copy()

        # Also downsample ideal (same phase) for reference
        ideal_1 = downsample(self.ideal_os, spui, self.cdr_phase) \
                  if spui >= 2 else self.symbols.copy()

        # Align measured to ideal
        al, lag, cp = align_by_correlation(meas_1, ideal_1)
        self.align_lag = lag
        self._pr(f"  Align: lag = {lag},  corr = {cp:.4f}")

        sa = self.symbols[lag:] if lag >= 0 else self.symbols
        ml = min(len(al), len(sa))
        self.meas_1sps  = al[:ml]
        self.sym_aligned = sa[:ml]
        self._pr(f"  Usable symbols: {ml}")

    # ---- Step 4: Linear fit -------------------------------------------
    def _step4_linear_fit(self):
        nt = self.v_ntaps.get()
        A, vs = build_convolution_matrix(self.sym_aligned, nt)
        y = self.meas_1sps[vs]
        N, L = A.shape

        # ============================================================
        #  STEP 4-A: What is the model?
        # ============================================================
        self._pr("=" * 62)
        self._pr("  Step 4: LINEAR FIT  --  detailed walkthrough")
        self._pr("=" * 62)
        self._pr("")
        self._pr("  GOAL: find h[0..L-1] such that")
        self._pr("")
        self._pr("    y[n]  =  h[0]*a[n] + h[1]*a[n-1] + ... "
                 f"+ h[{L-1}]*a[n-{L-1}]")
        self._pr("")
        self._pr("    where  a[n] = known ideal PRBS symbols")
        self._pr("           y[n] = measured waveform (1 sample/UI)")
        self._pr("           h[k] = unknown channel pulse response")
        self._pr(f"           L    = {L} taps")
        self._pr("")
        self._pr("  This is a LINEAR system  y = A * h")
        self._pr(f"  with A = {N} x {L}  (one row per output sample)")
        self._pr(f"  valid range: n = {vs.start} .. {vs.stop-1}  ({N} eqs)")

        # ============================================================
        #  STEP 4-B: Show the convolution matrix A
        # ============================================================
        self._pr("")
        self._pr("-" * 62)
        self._pr("  4-B: Build convolution matrix A")
        self._pr("-" * 62)
        self._pr("")
        self._pr("  Each ROW of A is the symbol sequence reversed around n:")
        self._pr(f"    Row for y[n]:  [ a[n], a[n-1], a[n-2], ... a[n-{L-1}] ]")
        self._pr("")

        SH = 6
        cols_show = min(8, L)
        self._pr(f"  First {SH} rows (showing {cols_show} of {L} cols):")
        hdr = "     n  |" + "".join(f" a[n-{k}] " for k in range(cols_show))
        self._pr(f"  {hdr}  ...")
        self._pr(f"  {'--------+' + '---------' * cols_show}")
        for i in range(min(SH, N)):
            n_idx = i + vs.start
            rs = " ".join(f"{A[i,k]:>8.1f}" for k in range(cols_show))
            self._pr(f"  {n_idx:>6d}  | {rs}  ...")
        self._pr(f"  {'   ...  ':>6s}  |")

        # ============================================================
        #  STEP 4-C: Show y vector (the target)
        # ============================================================
        self._pr("")
        self._pr("-" * 62)
        self._pr("  4-C: Target vector y  (measured waveform at 1 sps)")
        self._pr("-" * 62)
        self._pr("")
        for i in range(min(SH, N)):
            self._pr(f"  y[{i+vs.start:>4d}] = {y[i]:>10.6f}")
        self._pr(f"  ...  ({N} values total)")

        # ============================================================
        #  STEP 4-D: Solve   min || y - A*h ||^2
        # ============================================================
        self._pr("")
        self._pr("-" * 62)
        self._pr("  4-D: Solve least-squares   min || y - A*h ||^2")
        self._pr("-" * 62)
        self._pr("")
        self._pr("  Method: numpy.linalg.lstsq  (SVD-based)")

        fit_dc = self.v_fitdc.get()
        if fit_dc:
            A_aug = np.hstack([A, np.ones((N, 1))])
            self._pr(f"  (A augmented with 1-column for DC offset"
                     f" -> {A_aug.shape[0]}x{A_aug.shape[1]})")
        else:
            A_aug = A

        x, _, rank, sv = np.linalg.lstsq(A_aug, y, rcond=None)
        cond = sv[0] / sv[-1] if sv[-1] > 0 else np.inf
        self._pr(f"  Rank = {rank},  Cond# = {cond:.2f}")
        self._pr(f"  Singular values: max={sv[0]:.2f}  min={sv[-1]:.2f}")

        if fit_dc:
            h_est = x[:-1]; dc = float(x[-1])
            self._pr(f"  DC offset = {dc:.6f}")
        else:
            h_est = x; dc = 0.0

        # ============================================================
        #  STEP 4-E: Show the solution  h[k]
        # ============================================================
        self._pr("")
        self._pr("-" * 62)
        self._pr("  4-E: Solution  h[k]  =  extracted pulse response")
        self._pr("-" * 62)
        self._pr("")
        cp = int(np.argmax(np.abs(h_est)))
        for k in range(L):
            bar = "#" * int(abs(h_est[k]) / max(abs(h_est)) * 30)
            tag = "  <-- CURSOR (main tap)" if k == cp else ""
            self._pr(f"  h[{k:>2d}] = {h_est[k]:>12.8f}  |{bar}{tag}")

        # ============================================================
        #  STEP 4-F: Verify with a concrete numerical example
        # ============================================================
        self._pr("")
        self._pr("-" * 62)
        self._pr("  4-F: VERIFICATION  --  pick one row, compute by hand")
        self._pr("-" * 62)
        self._pr("")
        ex_row = N // 2  # pick a row in the middle
        n_ex = ex_row + vs.start
        self._pr(f"  Example: n = {n_ex}")
        self._pr(f"  y[{n_ex}] should equal  "
                 f"h[0]*a[{n_ex}] + h[1]*a[{n_ex-1}] + ... "
                 f"+ h[{L-1}]*a[{n_ex-L+1}]")
        self._pr("")

        # Show the computation term by term
        terms = []
        total = dc
        self._pr(f"  {'k':>4s}   {'h[k]':>12s}   {'a[n-k]':>8s}   "
                 f"{'h[k]*a[n-k]':>12s}   {'running sum':>12s}")
        self._pr(f"  {'----':>4s}   {'------------':>12s}   {'--------':>8s}   "
                 f"{'------------':>12s}   {'------------':>12s}")
        for k in range(L):
            hk = h_est[k]
            ak = A[ex_row, k]
            prod = hk * ak
            total += prod
            # Only print taps with significant contribution
            if abs(prod) > 0.001 or k == cp:
                tag = " <--" if k == cp else ""
                self._pr(f"  {k:>4d}   {hk:>12.6f}   {ak:>8.1f}   "
                         f"{prod:>12.6f}   {total:>12.6f}{tag}")
        if dc != 0:
            self._pr(f"  {'dc':>4s}   {dc:>12.6f}   {'':>8s}   "
                     f"{dc:>12.6f}")

        y_actual = y[ex_row]
        y_computed = float(A[ex_row] @ h_est + dc)
        err = y_actual - y_computed
        self._pr("")
        self._pr(f"  Computed:  Sum h[k]*a[n-k] + dc = {y_computed:>12.6f}")
        self._pr(f"  Actual y[{n_ex}]:                  = {y_actual:>12.6f}")
        self._pr(f"  Error:                           = {err:>12.6f}")
        self._pr(f"  --> This is ONE of {N} such equations. "
                 f"The least-squares")
        self._pr(f"      solver finds h that minimises the TOTAL squared error.")

        # ============================================================
        #  STEP 4-G: Overall fit quality
        # ============================================================
        y_hat = A @ h_est + dc
        resid = y - y_hat
        rms_r = np.sqrt(np.mean(resid ** 2))
        ss_res = np.sum(resid ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        p_s = np.mean(y ** 2); p_e = np.mean(resid ** 2)
        sndr = 10 * np.log10(p_s / p_e) if p_e > 1e-30 else np.inf

        self._pr("")
        self._pr("-" * 62)
        self._pr("  4-G: Overall fit quality")
        self._pr("-" * 62)
        self._pr(f"  RMS residual = {rms_r:.6f}")
        self._pr(f"  R^2          = {r2:.8f}   "
                 f"(1.0 = perfect fit)")
        self._pr(f"  SNDR (fit)   = {sndr:.2f} dB")
        self._met("sf", f"{sndr:.2f} dB")

        self.pulse_est = h_est
        self.ext_info = dict(
            cursor_pos=cp, dc_offset=dc,
            signal_reconstructed=y_hat, residual=resid,
            condition_number=cond, valid_slice=vs,
            singular_values=sv, r_squared=r2,
            A_matrix=A, y_signal=y)

    # ---- Compute ground-truth pulse via impulse test --------------------
    def _compute_true_pulse(self):
        """Impulse -> ZOH -> channel -> downsample = true 1-sps pulse."""
        spui = self.v_spui.get()
        nt = self.v_ntaps.get()
        nsym = nt * 2 + 40
        imp = np.zeros(nsym); imp[nsym // 2] = 1.0
        imp_os = np.repeat(imp, spui)  # ZOH

        # Same channel as _step2
        N = len(imp_os)
        X = np.fft.rfft(imp_os)
        f01 = np.fft.rfftfreq(N)
        f_bnyq = 0.5 / spui
        il = float(self.v_il.get())
        model = self.v_loss_model.get()
        if model == "sqrt":
            ratio = np.sqrt(np.clip(f01 / f_bnyq, 0, None))
        else:
            ratio = f01 / f_bnyq
        il_db = il * ratio; il_db[0] = 0; il_db = np.clip(il_db, -80, 0)
        H_mag = 10.0 ** (il_db / 20.0)
        delay = 4 * spui
        phase_lin = -2 * np.pi * np.arange(len(f01)) * delay / N
        H = H_mag * np.exp(1j * phase_lin)
        out = np.fft.irfft(X * H, N)

        # Downsample at same CDR phase
        out_1sps = out[self.cdr_phase::spui]

        # Align with extracted pulse via cross-correlation
        from scipy.signal import correlate, correlation_lags
        corr = correlate(out_1sps, self.pulse_est, mode="full")
        lags = correlation_lags(len(out_1sps), len(self.pulse_est), mode="full")
        best_lag = lags[np.argmax(np.abs(corr))]
        start = max(0, best_lag)
        self.true_pulse = out_1sps[start:start + nt]

        if len(self.true_pulse) < nt:
            self.true_pulse = np.pad(self.true_pulse, (0, nt - len(self.true_pulse)))

        cp_t = int(np.argmax(np.abs(self.true_pulse)))
        cp_e = self.ext_info["cursor_pos"]
        self._pr(f"  True pulse (impulse test): cursor = "
                 f"{self.true_pulse[cp_t]:.6f} @ [{cp_t}]  "
                 f"(extracted cursor @ [{cp_e}])")

    # ---- Step 5: Fitted waveform --------------------------------------
    def _step5_fitted(self):
        self._pr("--- Step 5: Fitted waveform = conv(symbols, h) ---")
        self.fitted_1sps = np.convolve(
            self.sym_aligned, self.pulse_est, "full"
        )[:len(self.sym_aligned)] + self.ext_info["dc_offset"]
        vs = self.ext_info["valid_slice"]
        sndr_m, _ = compute_sndr(self.meas_1sps[vs],
                                  self.fitted_1sps[vs])
        self._met("sm", f"{sndr_m:.2f} dB")
        self._pr(f"  SNDR (meas vs fitted) = {sndr_m:.2f} dB")

    # ---- Step 6: Metrics ----------------------------------------------
    def _step6_metrics(self):
        self._pr("--- Step 6: Metrics ---")
        cp = self.ext_info["cursor_pos"]
        rx = self.meas_1sps[cp:]; sy = self.sym_aligned[:len(rx)]
        rlm, levels, _ = compute_rlm(rx, sy)
        self._met("rlm", f"{rlm:.4f}")
        self._pr(f"  RLM = {rlm:.4f}")
        self._pr(f"  Levels = {np.round(levels, 4)}")

        # Verify AC response at Nyquist
        baud = self.v_baud.get()
        n_fft = max(1024, 2**int(np.ceil(np.log2(len(self.pulse_est)*16))))
        H_ext = np.fft.rfft(self.pulse_est, n=n_fft)
        f_n = np.fft.rfftfreq(n_fft)     # normalised to f_baud
        H_mag = np.abs(H_ext)
        H_dc = H_mag[0] if H_mag[0] > 1e-30 else 1.0
        ext_db = 20*np.log10(np.maximum(H_mag/H_dc, 1e-30))
        # De-embed ZOH sinc to recover pure channel
        sinc_f = np.sinc(f_n)
        sinc_f[sinc_f < 0.05] = 0.05
        sinc_db = 20*np.log10(sinc_f)
        ch_db = ext_db - sinc_db
        idx_nyq = np.argmin(np.abs(f_n - 0.5))
        self._pr(f"  Extracted IL @ Nyquist = {-ch_db[idx_nyq]:.2f} dB  "
                 f"(channel input: {self.v_il.get():.1f} dB)")

    # =================================================================
    #  DRAWING
    # =================================================================
    def _draw_all(self):
        self._draw_waveform()
        self._draw_eye()
        self._draw_pulse()
        self._draw_ac()
        self._draw_linfit()
        self._draw_hist()

    # ---- Waveform: ideal / measured / fitted --------------------------
    def _draw_waveform(self):
        fig = self.figs["Waveform"]; fig.clear()
        if self.meas_1sps is None:
            self._placeholder(fig, "Waveform"); return

        vs = self.ext_info["valid_slice"]
        N = min(80, vs.stop - vs.start)
        t = np.arange(N)
        cp = self.ext_info["cursor_pos"]

        # All three at 1 sps in the valid range
        ideal_seg  = self.sym_aligned[vs.start:vs.start+N]
        meas_seg   = self.meas_1sps[vs.start:vs.start+N]
        fitted_seg = self.fitted_1sps[vs.start:vs.start+N]

        ax1 = fig.add_subplot(2,1,1)
        ax1.plot(t, ideal_seg,  "b-",  lw=0.9, label="Ideal PRBS (symbols)")
        ax1.plot(t, meas_seg,   "r-",  lw=0.9, alpha=0.8,
                 label="Measured (after channel)")
        ax1.plot(t, fitted_seg, "g--", lw=1.0,
                 label="Fitted (conv(symbols, h))")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Waveform Comparison  (1 sample / UI)")
        ax1.legend(fontsize=8, loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Error
        err = meas_seg - fitted_seg
        ax2 = fig.add_subplot(2,1,2, sharex=ax1)
        ax2.fill_between(t, err, color="green", alpha=0.3)
        ax2.plot(t, err, "g-", lw=0.7)
        rms = np.sqrt(np.mean(err**2))
        ax2.set_xlabel("Symbol index (in valid range)")
        ax2.set_ylabel("Error")
        ax2.set_title(f"Measured - Fitted   (RMS = {rms:.6f})")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout(); self.cvs["Waveform"].draw()

    # ---- Eye diagram (white background) --------------------------------
    def _draw_eye(self):
        fig = self.figs["Eye Diagram"]; fig.clear()
        spui = self.v_spui.get()
        sig = self.meas_os
        if sig is None or spui < 2:
            self._placeholder(fig, "Eye Diagram"); return

        from scipy.ndimage import zoom as ndzoom
        from serdes_tx.cdr import _eye_opening_metric

        best_ph, ns = 0, len(sig)//spui
        best_m = -np.inf
        for ph in range(spui):
            m = _eye_opening_metric(sig[ph::spui][:ns])
            if m > best_m: best_m, best_ph = m, ph
        fold0 = (best_ph + spui//2) % spui
        sig = sig[fold0:]

        seg_raw = 2*spui; nseg = len(sig)//seg_raw
        if nseg < 20: self._placeholder(fig, "Eye Diagram"); return
        data = sig[:nseg*seg_raw].reshape(nseg, seg_raw)

        ifact = max(1, 128//spui)
        if ifact > 1: data = ndzoom(data, (1, ifact), order=3)
        seg = data.shape[1]
        t_ui = np.linspace(0, 2, seg, endpoint=False)
        ta = np.tile(t_ui, nseg); va = data.ravel()
        vlo,vhi = np.percentile(va,[0.1,99.9]); vm=0.08*(vhi-vlo)

        ax = fig.add_subplot(111)
        ax.hist2d(ta, va, bins=[min(seg,512),400],
                  range=[[0,2],[vlo-vm,vhi+vm]], cmap=EYE_CMAP, cmin=1)
        ax.set_facecolor("white")
        ax.set_xlabel("Time (UI)"); ax.set_ylabel("Amplitude")
        ax.set_title("PAM4 Eye Diagram — Measured Waveform")
        fig.tight_layout(); self.cvs["Eye Diagram"].draw()

    # ---- Pulse response -----------------------------------------------
    def _draw_pulse(self):
        fig = self.figs["Pulse Response"]; fig.clear()
        if self.pulse_est is None:
            self._placeholder(fig, "Pulse Response"); return
        h = self.pulse_est; cp = self.ext_info["cursor_pos"]
        taps = np.arange(len(h)) - cp

        has_true = (self.true_pulse is not None
                    and len(self.true_pulse) == len(h))

        if has_true:
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
            ax1.stem(taps - 0.15, self.true_pulse, linefmt="g-",
                     markerfmt="gs", basefmt="k-", label="True (impulse)")
            ax1.stem(taps + 0.15, h, linefmt="b-",
                     markerfmt="bo", basefmt="k-", label="Extracted")
            ax1.axhline(0, color="gray", lw=0.5, ls="--")
            ax1.set_ylabel("Amplitude")
            ax1.set_title("Pulse Response: Extracted vs True")
            ax1.legend(fontsize=8); ax1.grid(True, alpha=0.3)

            err = h - self.true_pulse
            rms = np.sqrt(np.mean(err**2))
            self._met("fe", f"{rms:.6f}")
            ax2.stem(taps, err, linefmt="r-", markerfmt="r^", basefmt="k-")
            ax2.axhline(0, color="gray", lw=0.5, ls="--")
            ax2.set_xlabel("Tap (relative to cursor)")
            ax2.set_ylabel("Error")
            ax2.set_title(f"Extraction Error (RMS = {rms:.6f})")
            ax2.grid(True, alpha=0.3)
        else:
            ax = fig.add_subplot(111)
            ml, sl, bl = ax.stem(taps, h, basefmt="k-")
            import matplotlib.pyplot as plt
            plt.setp(sl, lw=1.2, color="steelblue")
            plt.setp(ml, ms=5, color="steelblue")
            ax.plot(0, h[cp], "ro", ms=8,
                    label=f"Cursor = {h[cp]:.6f}")
            ax.axhline(0, color="gray", lw=0.5, ls="--")
            ax.set_xlabel("Tap (relative to cursor)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Extracted Channel Pulse Response")
            ax.legend(); ax.grid(True, alpha=0.3)
        fig.tight_layout(); self.cvs["Pulse Response"].draw()

    # ---- AC Response: extracted vs true pulse vs known channel ----------
    def _draw_ac(self):
        fig = self.figs["AC Response"]; fig.clear()
        if self.pulse_est is None:
            self._placeholder(fig, "AC Response"); return

        baud = self.v_baud.get()
        n_fft = max(2048, 2**int(np.ceil(np.log2(len(self.pulse_est)*16))))

        def _spectrum_db(h):
            H = np.fft.rfft(h, n=n_fft)
            m = np.abs(H); dc = m[0] if m[0] > 1e-30 else 1.0
            return 20*np.log10(np.maximum(m / dc, 1e-30))

        f_norm = np.fft.rfftfreq(n_fft)   # 0..0.5 (in f_baud)
        f_ghz  = f_norm * baud
        mask   = f_ghz <= baud / 2 * 1.05  # up to ~Nyquist

        ax = fig.add_subplot(111)

        # 1) Extracted pulse spectrum
        ext_db = _spectrum_db(self.pulse_est)
        ax.plot(f_ghz[mask], -ext_db[mask], "b-", lw=1.8,
                label="Extracted pulse |G(f)|")

        # 2) True pulse spectrum (impulse test ground truth)
        if self.true_pulse is not None and len(self.true_pulse) > 0:
            true_db = _spectrum_db(self.true_pulse)
            ax.plot(f_ghz[mask], -true_db[mask], "g--", lw=1.5,
                    label="True pulse (impulse test)")

        # 3) Known channel (for reference — does NOT include sinc+alias)
        if self.ch_freq_ghz is not None:
            ch_mask = self.ch_freq_ghz <= baud / 2 * 1.05
            ax.plot(self.ch_freq_ghz[ch_mask], -self.ch_il_db[ch_mask],
                    "r:", lw=1.2, alpha=0.7,
                    label="Pure channel H(f) (ref)")

        ax.axvline(baud/2, color="gray", lw=0.8, ls=":")
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel("Insertion Loss (dB)")
        ax.set_title("AC Response — blue/green should overlap\n"
                     "(red = pure channel before ZOH+alias)")
        ax.set_xlim(0, baud * 0.55)
        ym = max(20, -self.v_il.get() * 1.8)
        ax.set_ylim(-1, ym); ax.invert_yaxis()
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        fig.tight_layout(); self.cvs["AC Response"].draw()

    # ---- Linear-fit visualisation -------------------------------------
    def _draw_linfit(self):
        fig = self.figs["Linear Fit"]; fig.clear()
        info = self.ext_info
        if info is None:
            self._placeholder(fig, "Linear Fit"); return

        A = info["A_matrix"]; y = info["y_signal"]
        y_hat = info["signal_reconstructed"]; resid = info["residual"]
        sv = info["singular_values"]; r2 = info["r_squared"]
        h = self.pulse_est; cp = info["cursor_pos"]; dc = info["dc_offset"]
        N, L = A.shape; show = min(200, N)

        # ---- (1) Matrix A heatmap ----
        ax1 = fig.add_subplot(2, 3, 1)
        rows = min(50, N)
        im = ax1.imshow(A[:rows], aspect="auto", cmap="RdBu_r",
                        interpolation="nearest", vmin=-3.5, vmax=3.5)
        ax1.set_xlabel("Column k  (= tap index)")
        ax1.set_ylabel("Row  (= output sample)")
        ax1.set_title(f"(1) Matrix A  [{N}x{L}]\n"
                      f"A[i,k] = a[n-k]  (shifted symbols)")
        fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

        # ---- (2) One-row example ----
        ax2 = fig.add_subplot(2, 3, 2)
        ex = N // 2
        n_ex = ex + info["valid_slice"].start
        ax2.bar(range(L), A[ex], color="steelblue", alpha=0.7)
        ax2.set_xlabel("k"); ax2.set_ylabel("a[n-k]")
        ax2.set_title(f"(2) Example: row n={n_ex}\n"
                      f"= [a[{n_ex}], a[{n_ex-1}], ..., a[{n_ex-L+1}]]")
        ax2.grid(True, alpha=0.3)

        # ---- (3) y measured vs y_hat = A*h ----
        t = np.arange(show)
        ax3 = fig.add_subplot(2, 3, 3)
        ax3.plot(t, y[:show], "b-", lw=0.9, label="y (measured)")
        ax3.plot(t, y_hat[:show], "r--", lw=0.9, label="A*h + dc (fitted)")
        ax3.set_xlabel("Sample n")
        ax3.set_title("(3) y  vs  A*h\n(should overlap)")
        ax3.legend(fontsize=7); ax3.grid(True, alpha=0.3)

        # ---- (4) Residual ----
        ax4 = fig.add_subplot(2, 3, 4)
        ax4.plot(t, resid[:show], "g-", lw=0.6)
        rms = np.sqrt(np.mean(resid ** 2))
        ax4.axhline(rms, color="orange", ls="--", lw=0.8)
        ax4.axhline(-rms, color="orange", ls="--", lw=0.8)
        ax4.set_xlabel("Sample n"); ax4.set_ylabel("Error")
        ax4.set_title(f"(4) Residual = y - A*h\n(RMS = {rms:.6f})")
        ax4.grid(True, alpha=0.3)

        # ---- (5) Singular values ----
        ax5 = fig.add_subplot(2, 3, 5)
        ax5.bar(range(len(sv)), sv, color="teal", alpha=0.7)
        ax5.set_xlabel("Index"); ax5.set_ylabel("Singular value")
        ax5.set_title(f"(5) SVD of A\n"
                      f"cond# = {sv[0]/sv[-1]:.1f}  "
                      f"(flat = well-conditioned)")
        ax5.grid(True, alpha=0.3)

        # ---- (6) Summary text ----
        ax6 = fig.add_subplot(2, 3, 6); ax6.axis("off")
        p_s = np.mean(y**2); p_e = np.mean(resid**2)
        sndr = 10*np.log10(p_s/p_e) if p_e > 1e-30 else np.inf

        # One-row verification
        y_ex = float(A[ex] @ h + dc)
        y_act = float(y[ex])

        txt = "\n".join([
            "HOW IT WORKS", "=" * 36,
            "",
            "We KNOW: a[n] (ideal PRBS symbols)",
            "         y[n] (measured waveform)",
            "",
            "We WANT: h[k] (pulse response)",
            "",
            "Each equation:",
            f"  y[n] = h[0]*a[n] + h[1]*a[n-1]",
            f"       + ... + h[{L-1}]*a[n-{L-1}]",
            "",
            f"Stack {N} equations -> y = A*h",
            f"Solve by least-squares (SVD)",
            "",
            f"Verify row n={n_ex}:",
            f"  Sum h[k]*a[n-k] = {y_ex:.6f}",
            f"  Actual y[{n_ex}]    = {y_act:.6f}",
            f"  Error            = {abs(y_act-y_ex):.6f}",
            "",
            f"Overall: R^2={r2:.6f}  SNDR={sndr:.1f}dB",
        ])
        ax6.text(0.02, 0.98, txt, transform=ax6.transAxes, fontsize=8,
                 fontfamily="monospace", va="top",
                 bbox=dict(boxstyle="round", fc="#f0f8ff"))

        fig.tight_layout(); self.cvs["Linear Fit"].draw()

    # ---- Histogram ----------------------------------------------------
    def _draw_hist(self):
        fig = self.figs["Histogram"]; fig.clear()
        if self.meas_1sps is None:
            self._placeholder(fig, "Histogram"); return
        cp = self.ext_info["cursor_pos"] if self.ext_info else 0
        sig = self.meas_1sps[cp:]; sy = self.sym_aligned[:len(sig)]
        ax = fig.add_subplot(111)
        cols = ["#e74c3c","#e67e22","#2ecc71","#3498db"]
        labs = ["Level -3","Level -1","Level +1","Level +3"]
        nb = min(200, max(50, len(sig)//20))
        for i,lev in enumerate([-3,-1,1,3]):
            m = np.abs(sy - lev) < 0.5
            if np.any(m):
                ax.hist(sig[m],bins=nb,alpha=0.6,color=cols[i],
                        label=labs[i],density=True)
        try:
            for lv in estimate_levels(sig):
                ax.axvline(lv,color="gray",lw=1,ls="--",alpha=0.6)
        except Exception: pass
        ax.set_xlabel("Amplitude"); ax.set_ylabel("Density")
        ax.set_title("Level Histogram (Measured, Aligned)")
        ax.legend(fontsize=8); ax.grid(True,alpha=0.3)
        fig.tight_layout(); self.cvs["Histogram"].draw()

    def _placeholder(self, fig, name):
        fig.text(0.5,0.5,"Run analysis first",
                 ha="center",va="center",fontsize=14,color="gray")
        self.cvs[name].draw()

    # =================================================================
    #  FILE I/O
    # =================================================================
    def _load(self):
        path = filedialog.askopenfilename(
            filetypes=[("CSV","*.csv"),("NumPy","*.npy"),("All","*.*")])
        if not path: return
        try:
            if path.endswith(".npy"):
                d = np.load(path).astype(np.float64)
            else:
                d = np.loadtxt(path, delimiter=",", comments="#")
                if d.ndim > 1: d = d[:,0]
                d = d.astype(np.float64)
            self._pr(f"Loaded: {path}  ({len(d)} samples)")
            self.meas_os = d
            # Need ideal PRBS pattern for fitting
            if self.symbols is None:
                self._step1_gen_ideal()
            self._step3_cdr_align()
            self._step4_linear_fit()
            self._step5_fitted()
            self._step6_metrics()
            self._draw_all()
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _export(self):
        d = filedialog.askdirectory()
        if not d: return
        for n,fig in self.figs.items():
            fn = n.lower().replace(" ","_").replace("/","_")
            fig.savefig(f"{d}/{fn}.png",dpi=150,bbox_inches="tight")
        self._pr("Plots exported.")


# =====================================================================
if __name__ == "__main__":
    App().mainloop()
