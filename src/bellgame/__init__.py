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
from .network import (
    add_node,
    connect,
    empty_network,
    from_edges,
    from_matrix,
    from_networkx,
    link_params,
    links,
    memory_params,
    node_names,
    parameters,
    set_all_links,
    set_detectors,
    set_link,
    set_memory,
    set_source,
    star,
    to_matrix,
    to_networkx,
    two_player_network,
    two_star,
)
from .paths import build, compose_path, end_to_end
from .plots import (
    plot_angles,
    plot_convergence,
    plot_correlations,
    plot_network,
    plot_photon_numbers,
    plot_results,
    plot_sweep,
    reference_lines,
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
