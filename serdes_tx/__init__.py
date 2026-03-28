"""PAM4 SerDes TX Analysis and Simulation Framework.

A modular Python framework for PAM4 SerDes transmitter analysis,
including PRBS generation, pulse response extraction, signal quality
metrics, channel simulation, and visualization.

Modules
-------
prbs : PRBS9 / PRBSQ9 pattern generation
pam4 : PAM4 encoding/decoding with Gray coding
cdr  : Clock data recovery and alignment
pulse : Pulse response extraction via linear regression
metrics : SNDR, RLM, eye metrics, insertion loss
channel : AWGN, ISI, bandwidth limitation, jitter
visualization : Eye diagram, histogram, frequency response plots
"""

from .prbs import (
    generate_prbs, generate_prbs9, generate_prbsq9, verify_prbs9,
    SUPPORTED_ORDERS, PRBS_POLYS, prbs_period, prbs_info,
)
from .pam4 import (
    generate_pam_prbs, pam_levels, SUPPORTED_PAM,
    bits_to_pam4, pam4_to_bits,
    generate_pam4_prbs9, generate_pam4_prbsq9,
    upsample_pam4, estimate_levels,
    GRAY_LEVELS,
)
from .cdr import (
    align_by_correlation,
    find_optimal_phase,
    gardner_cdr,
    downsample,
)
from .pulse import (
    extract_pulse_response,
    build_convolution_matrix,
    pulse_to_step_response,
)
from .metrics import (
    compute_sndr,
    compute_rlm,
    compute_eye_metrics,
    compute_insertion_loss,
)
from .channel import (
    add_awgn,
    apply_isi,
    apply_bandwidth_limit,
    add_jitter,
    apply_channel,
    default_channel_fir,
    make_channel_from_il,
)
from .visualization import (
    plot_eye_diagram,
    plot_pulse_response,
    plot_histogram,
    plot_frequency_response,
    plot_fir_comparison,
    plot_all,
)

__version__ = '1.0.0'
