"""bellgame: the CHSH game over simulated quantum networks.

Use it as ``import bellgame as bg``. Everything is available as ``bg.<name>``.
"""

from .defaults import DEFAULTS, default_values, describe
from .fock import fiber_transmissivity
from .game import (
    CLASSICAL_S,
    CLASSICAL_WIN_RATE,
    TSIRELSON_S,
    TSIRELSON_WIN_RATE,
    chsh_value,
    correlations,
    play_chsh,
    play_classical,
    referee_wins,
    sample_counts,
    summarize,
    table_from_classical,
    table_from_counts,
    table_from_state,
    table_from_strategy,
    win_rate,
)
from .link import (
    arriving_state,
    check_link,
    dark_count_prob,
    describe_link,
    link,
    link_outcomes,
    link_state,
    link_table,
    photon_numbers,
)
from .qubits import (
    apply_local,
    bell_pair,
    dephase,
    fidelity,
    misalign,
    polarization_rotation,
    rx,
    ry,
    rz,
    swap,
    werner,
)
from .strategies import (
    CLASSICAL_PLAYERS,
    always_one,
    always_zero,
    check_strategy,
    copy_bit,
    flip_bit,
    optimal_strategy,
)

__version__ = "0.1.0"
