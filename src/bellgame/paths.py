"""End-to-end entanglement across a network path.

Two halves work together (the "hybrid" design):

* **SeQUeNCe** runs the real network: routing, Barrett-Kok entanglement
  generation on each link, swapping at the middle nodes. From it we get the
  path, how many end-to-end pairs arrive per second, and how long every qubit
  waited in a memory.
* **bellgame's Fock model** gives each link's two-qubit state (``bg.link_state``).
  We let each stored qubit dephase for as long as it waited, then join the
  links with exact entanglement swaps (``bg.swap``).

Assumption: each link's photon pair is stored in quantum memories when it is
heralded, so a link's state is the Fock model's one-photon-at-each-end state.

Example:
    >>> import bellgame as bg
    >>> net = bg.two_player_network(distance_km=5)
    >>> path = bg.end_to_end(net, "Alice", "Bob", seed=1)
    >>> path["path"]
    ['Alice', 'Bob']
"""

import numpy as np

from ._sequence_adapter import build_topology, run_request
from .link import link_state
from .network import link_params, memory_params
from .qubits import dephase, fidelity, swap


def build(net, seed=None):
    """Turn the network dict into SeQUeNCe's ``RouterNetTopo`` so you can explore it.

    Example:
        >>> import bellgame as bg
        >>> topo = bg.build(bg.two_player_network())
        >>> sorted(r.name for r in topo.get_nodes_by_type("QuantumRouter"))
        ['Alice', 'Bob']
    """
    return build_topology(net, seed=seed)


def _dephase_one(rho, qubit, wait_ms, coherence_time_ms):
    if coherence_time_ms == float("inf") or wait_ms <= 0:
        return rho
    return dephase(rho, wait_ms, coherence_time_ms, qubits=(qubit,))


def compose_path(link_states, nodes, waits_ms, coherence_ms):
    """Join link states along a path, after letting every stored qubit dephase.

    ``link_states[i]`` is the 4x4 state between ``nodes[i]`` and ``nodes[i + 1]``.
    ``waits_ms[node]`` is how long that node's qubits waited; ``coherence_ms[node]``
    is that node's memory coherence time.

    Example:
        >>> import bellgame as bg
        >>> rho = compose_path([bg.bell_pair(), bg.bell_pair()], ["A", "H", "B"],
        ...                    {"A": 0, "H": 0, "B": 0}, {"A": 1, "H": 1, "B": 1})
        >>> round(bg.fidelity(rho), 3)
        1.0
    """
    states = []
    for i, rho in enumerate(link_states):
        left, right = nodes[i], nodes[i + 1]
        rho = _dephase_one(rho, 0, waits_ms[left], coherence_ms[left])
        rho = _dephase_one(rho, 1, waits_ms[right], coherence_ms[right])
        states.append(rho)
    out = states[0]
    for rho in states[1:]:
        out = swap(out, rho)
    return out


def end_to_end(net, src, dst, seed=None, sim_time_s=0.2):
    """Entanglement between ``src`` and ``dst``: path, rate, waits, and the delivered state.

    Returns a dict with:

    * ``path``: the nodes SeQUeNCe routed through
    * ``state``: the average delivered two-qubit state (use it with ``bg.play_chsh``)
    * ``fidelity``: how close ``state`` is to a perfect Bell pair
    * ``rate_hz``: end-to-end pairs delivered per second
    * ``wait_ms``: average time a qubit waited in memory, per node
    * ``links``: each link's own fidelity and chance per pulse of a one-photon-each event
    """
    run = run_request(net, src, dst, sim_time_s, seed=seed)
    path = run["path"]
    if not run["reserved"] or len(path) < 2:
        raise ValueError(f"SeQUeNCe could not reserve a path from '{src}' to '{dst}'. Are they "
                         "connected, and does every node have enough memories (memory_size)?")

    link_info = []
    states = []
    for u, v in zip(path, path[1:]):
        state, prob = link_state(link_params(net, u, v))
        states.append(state)
        link_info.append({"nodes": [u, v], "fidelity": fidelity(state), "pair_prob": prob})

    coherence = {n: memory_params(net, n)["coherence_time_ms"] for n in path}
    middle = {n: float(np.mean(run["swap_waits_ms"].get(n, [0.0]))) for n in path[1:-1]}
    n_pairs = min(len(run["src_waits_ms"]), len(run["dst_waits_ms"]))
    if n_pairs == 0:
        pairs = [(0.0, 0.0)]
    else:
        pairs = list(zip(run["src_waits_ms"][:n_pairs], run["dst_waits_ms"][:n_pairs]))

    total = np.zeros((4, 4), dtype=complex)
    for w_src, w_dst in pairs:
        waits = dict(middle)
        waits[path[0]] = w_src
        waits[path[-1]] = w_dst
        total += compose_path(states, path, waits, coherence)
    state = total / len(pairs)

    wait_ms = {path[0]: float(np.mean([p[0] for p in pairs]))}
    wait_ms.update(middle)
    wait_ms[path[-1]] = float(np.mean([p[1] for p in pairs]))
    return {
        "path": path,
        "state": state,
        "fidelity": fidelity(state),
        "rate_hz": n_pairs / sim_time_s,
        "pairs": n_pairs,
        "wait_ms": wait_ms,
        "links": link_info,
        "sim_time_s": sim_time_s,
    }
