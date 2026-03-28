# PAM4 SerDes TX Linear Fit

A Python framework for extracting channel pulse response from PAM4 SerDes waveforms using linear fitting (least-squares regression).

**[Live Tutorial & Documentation](https://nxgs.github.io/TX_linear_fit/)**

## What it does

Given a **known PRBS pattern** and a **measured waveform** (after passing through a lossy channel), this tool:

1. Builds a convolution matrix from the known symbols
2. Solves `y = A * h` via least-squares (SVD) to extract the channel **pulse response**
3. Computes the **fitted waveform** = conv(symbols, h) for verification
4. Derives the **AC frequency response** via FFT of the pulse response
5. Reports **SNDR, RLM, eye metrics**, and **confidence intervals** per tap

## Quick Start

```bash
# Install dependencies
pip install numpy scipy matplotlib

# Launch GUI
python gui.py
```

Click **Run Full Analysis** to see all results across 8 tabs.

## GUI Tabs

| Tab | Shows |
|-----|-------|
| Waveform | Ideal / measured / fitted signals overlaid (1 sps) |
| Waveform (OS) | Oversampled waveforms (64 sps) showing transition shapes |
| Eye Diagram | PAM4 eye with CDR alignment, interpolated, white background |
| Pulse Response | Extracted vs ground-truth pulse with 95% confidence interval |
| Step Response | Cumulative sum of pulse (rise time, settling, DC gain) |
| AC Response | Frequency response: extracted vs true vs pure channel |
| Linear Fit | Matrix heatmap, SVD, residual, one-row verification |
| Histogram | PAM4 level distributions, RLM |

## Mouse Interaction (all plots)

| Action | Effect |
|--------|--------|
| Scroll wheel | Zoom in/out centered on cursor |
| Right-click drag | Rectangle select to zoom |
| Double right-click | Reset to original scale |

## Project Structure

```
serdes_tx/          Core library
  prbs.py           PRBS9/PRBSQ9 generation
  pam4.py           PAM4 Gray coding, upsampling
  cdr.py            CDR + alignment
  pulse.py          Pulse extraction (convolution matrix + lstsq)
  metrics.py        SNDR, RLM, eye metrics, insertion loss
  channel.py        Freq-domain channel, AWGN, BW limit, jitter
  visualization.py  Eye diagram, plots
gui.py              Tkinter GUI (8 tabs)
demo.py             CLI demo script
docs/               GitHub Pages tutorial site
  index.html        English tutorial
  tutorial_zh.html  Chinese tutorial
  img/              Screenshots
```

## Documentation

- **English**: [docs/index.html](docs/index.html)
- **Chinese**: [docs/tutorial_zh.html](docs/tutorial_zh.html)

The tutorial covers the complete mathematical derivation, explains every GUI panel, and includes FAQ, channel model comparison, and PRBS pattern guide.
