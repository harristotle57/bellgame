"""bellgame: the CHSH game over simulated quantum networks.

SeQUeNCe simulates the network (``bg.run_network``); bellgame plays the game on
the pairs it delivers.

Use it as ``import bellgame as bg``. Everything is available as ``bg.<name>``.
"""

from .defaults import DEFAULTS, default_values, describe
from .e91 import E91_OLD_SETTINGS, E91_SETTINGS, run_e91
from .fock import fiber_transmissivity
from .game import (
    CLASSICAL_S,
    CLASSICAL_WIN_RATE,
    TSIRELSON_S,
    TSIRELSON_WIN_RATE,
    answer,
    chsh_value,
    correlations,
    play_chsh,
    play_classical,
    play_rounds,
    referee_wins,
    sample_counts,
    summarize,
    summarize_counts,
    table_from_classical,
    table_from_counts,
    table_from_state,
    table_from_strategy,
    uses_history,
    win_rate,
)
from .link import (
    arriving_state,
    check_link,
    dark_count_prob,
    describe_link,
    fock_pair_weights,
    link,
    link_outcomes,
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
    node_names,
    node_params,
    parameters,
    set_all_links,
    set_link,
    set_node,
    star,
    to_matrix,
    to_networkx,
    two_player_network,
    two_star,
)
from .optimize import (
    chsh_objective,
    compare_optimizers,
    find_alignment,
    misaligned_source,
    random_misalignment,
    with_correction,
    wrap_angles,
)
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
    bell_diagonal,
    bell_pair,
    fidelity,
    misalign,
    polarization_rotation,
    rx,
    ry,
    rz,
    werner,
)
from .simulate import analytic_pair_weights, bell_diagonal_mode, pair_weights, run_network, to_sequence
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
