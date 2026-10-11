"""Play the CHSH game across a network simulated by SeQUeNCe.

``bg.run_network(net, "A1", "B2")`` does what a real experiment would:

1. SeQUeNCe finds a path and starts making pairs on every link along it
   (single-heralded generation), joins them by entanglement swapping at the
   middle nodes, and keeps the halves in quantum memories that slowly decay.
   It tracks every pair's state the whole time, in its Bell-diagonal form.
2. A referee asks Alice and Bob a question ``questions_hz`` times per second.
   When a question arrives, the players use a pair they share right now, in
   whatever state SeQUeNCe says it has decayed to. If they share none, the
   round has no pair.
3. The result holds the **average state of the pairs the players used**.
   Hand it to ``bg.play``, ``bg.run_e91`` or ``bg.find_alignment`` like
   any other source. (The players' measurements don't change what the network
   does, and outcome probabilities are linear in the state, so this average
   gives exactly the statistics of playing every round.)

How good a link's fresh pair is comes from its ``link_model``:

* ``"fixed"``: ``raw_fidelity`` and ``raw_errors``, the same at any distance
* ``"analytic"``: the same, mixed with noise from dark counts, which matter more
  the fewer photons survive the fiber (a closed formula, see ``analytic_pair_weights``)
* ``"fock"``: the full photon-number simulation of an SPDC source (``bg.fock_pair_weights``)

Example:
    >>> import bellgame as bg
    >>> run = bg.run_network(bg.two_player_network(5), "Alice", "Bob", sim_time_s=0.2, seed=1)
    >>> run["path"], run["rounds"] > 20
    (['Alice', 'Bob'], True)
"""

import collections
import contextlib
import itertools

import networkx as nx
import numpy as np
from sequence.app.request_app import RequestApp
from sequence.constants import BELL_DIAGONAL_STATE_FORMALISM, SINGLE_HERALDED
from sequence.entanglement_management.generation import EntanglementGenerationA, EntanglementGenerationB
from sequence.entanglement_management.generation.single_heralded import SingleHeraldedA
from sequence.entanglement_management.purification import PurificationProtocol
from sequence.entanglement_management.swapping import EntanglementSwappingA, EntanglementSwappingB
from sequence.kernel.event import Event
from sequence.kernel.process import Process
from sequence.kernel.quantum_manager import QuantumManager
from sequence.topology.router_net_topo import RouterNetTopo
from sequence.utils import metrics
from sequence.utils.metrics import EventTypes

from .defaults import _as_number, _in_range
from .link import fock_pair_weights
from .network import link_params, links, node_names, node_params, to_networkx
from .qubits import bell_diagonal, fidelity

PS_PER_S = 1e12
PS_PER_MS = 1e9
FIBER_KM_PER_S = 2e5            # light in fiber travels about 200,000 km/s
START_PS = int(1e9)             # the game starts 1 ms in, plus time to reserve the path
BSM_COUNT_RATE_HZ = 5e7
BSM_TIME_RESOLUTION_PS = 100
GENERATION = "bellgame_heralded"
NO_PAIR_POLICIES = ("discard", "random")
PICK_POLICIES = ("newest", "oldest")


# ---------------------------------------------------------------- link physics


def analytic_pair_weights(link):
    """Bell weights ``[Phi+, Phi-, Psi+, Psi-]`` of a link's fresh pair, with dark counts.

    Each end detects its photon with probability eta (fiber loss over half the
    link, times the detector efficiency). With probability d, at least one of its
    two detectors fires by itself (a dark count) during the coincidence window.
    An end clicks with probability ``eta + (1 - eta) * d``, and a pair is heralded
    when both ends click. Only a fraction ``w = eta^2 / (eta + (1 - eta) * d)^2``
    of heralds come from the real photons; the rest give random bits (white noise).

    Example:
        >>> import bellgame as bg
        >>> link = bg.link_params(bg.two_player_network(), "Alice", "Bob")  # raw_fidelity 0.98
        >>> link["dark_count_rate_hz"] = 1e5  # noisy detectors
        >>> [round(analytic_pair_weights({**link, "distance_km": km})[0], 3) for km in (1, 150, 300)]
        [0.98, 0.969, 0.717]
    """
    eta = 10 ** (-link["loss_db_per_km"] * link["distance_km"] / 2 / 10) * link["detector_efficiency"]
    d = 1 - (1 - link["dark_count_rate_hz"] * link["coincidence_window_ns"] * 1e-9) ** 2
    click = eta + (1 - eta) * d
    w = eta**2 / click**2 if click > 0 else 0.0
    return [w * x + (1 - w) / 4 for x in fixed_pair_weights(link)]


def fixed_pair_weights(link):
    """Bell weights of a link's fresh pair from ``raw_fidelity`` and ``raw_errors`` alone."""
    f = link["raw_fidelity"]
    ex, ey, ez = link["raw_errors"]
    return [f, (1 - f) * ez, (1 - f) * ex, (1 - f) * ey]


def pair_weights(link):
    """Bell weights ``[Phi+, Phi-, Psi+, Psi-]`` of a link's fresh pair, by its ``link_model``."""
    if link["link_model"] == "fixed":
        return fixed_pair_weights(link)
    if link["link_model"] == "analytic":
        return analytic_pair_weights(link)
    return fock_pair_weights(link)[0]


@EntanglementGenerationA.register(GENERATION)
class _LinkGenerationA(SingleHeraldedA):
    """SeQUeNCe's single-heralded generation, with the fresh pair's state taken from bellgame's link model.

    SeQUeNCe would use one fidelity per memory; we look up the state for this
    particular link (``node.bellgame_pairs[remote node]``) just before the pair is made.
    """

    def update_memory(self):
        weights = self.owner.bellgame_pairs[self.remote_node_name]
        self.raw_fidelity = weights[0]
        errors = weights[2], weights[3], weights[1]  # X, Y, Z
        total = sum(errors)
        self.raw_epr_errors = [e / total for e in errors] if total > 0 else [1 / 3] * 3
        return super().update_memory()


# ---------------------------------------------------------------- SeQUeNCe topology


@contextlib.contextmanager
def bell_diagonal_mode():
    """Inside this ``with`` block, SeQUeNCe uses the protocols bellgame needs.

    SeQUeNCe picks its protocols (how pairs are made, swapped and purified) from
    settings shared by the whole program. This sets them to bellgame's
    Bell-diagonal ones, and puts back whatever was there before when the block
    ends. ``bg.run_network`` does this for you; you only need it to run a
    ``bg.to_sequence`` topology yourself.

    Example:
        >>> import bellgame as bg
        >>> with bg.bell_diagonal_mode():
        ...     topo = bg.to_sequence(bg.two_player_network())
        ...     topo.get_timeline().init()
    """
    settings = [  # (read it, set it, bellgame's value)
        (QuantumManager.get_active_formalism, QuantumManager.set_global_manager_formalism,
         BELL_DIAGONAL_STATE_FORMALISM),
        (EntanglementGenerationA.get_global_type, EntanglementGenerationA.set_global_type, GENERATION),
        (EntanglementGenerationB.get_global_type, EntanglementGenerationB.set_global_type, SINGLE_HERALDED),
        (EntanglementSwappingA.get_formalism, EntanglementSwappingA.set_formalism, BELL_DIAGONAL_STATE_FORMALISM),
        (EntanglementSwappingB.get_formalism, EntanglementSwappingB.set_formalism, BELL_DIAGONAL_STATE_FORMALISM),
        (PurificationProtocol.get_formalism, PurificationProtocol.set_formalism, BELL_DIAGONAL_STATE_FORMALISM),
    ]
    before = [get() for get, _, _ in settings]
    try:
        for _, put, value in settings:
            put(value)
        yield
    finally:
        for (_, put, _), value in zip(settings, before):
            put(value)


def sequence_config(net, seed):
    """The SeQUeNCe ``RouterNetTopo`` configuration for a bellgame network."""
    seed = int(seed)
    templates, nodes, qconnections = {}, [], []
    for i, name in enumerate(node_names(net)):
        p = node_params(net, name)
        coherence_s = -1 if p["coherence_time_ms"] == float("inf") else p["coherence_time_ms"] / 1000
        templates[f"node_{name}"] = {"MemoryArray": {
            "frequency": float(p["memory_frequency_hz"]),
            "coherence_time": coherence_s,
            "efficiency": float(p["memory_efficiency"]),
            "fidelity": 1.0,  # replaced per link by _LinkGenerationA
            "decoherence_errors": [float(e) for e in p["memory_errors"]],
        }}
        nodes.append({"name": name, "type": "QuantumRouter", "seed": seed * 1000 + i,
                      "memo_size": int(p["memory_size"]), "template": f"node_{name}"})
    for i, (a, b) in enumerate(links(net)):
        p = link_params(net, a, b)
        detector = {"efficiency": float(p["detector_efficiency"]), "dark_count": float(p["dark_count_rate_hz"]),
                    "count_rate": BSM_COUNT_RATE_HZ, "time_resolution": BSM_TIME_RESOLUTION_PS}
        templates[f"bsm_{a}_{b}"] = {"encoding_type": SINGLE_HERALDED,
                                    "SingleHeraldedBSM": {"detectors": [dict(detector), dict(detector)]}}
        qconnections.append({"node1": a, "node2": b, "type": "meet_in_the_middle",
                             "attenuation": p["loss_db_per_km"] / 1000, "distance": p["distance_km"] * 1000,
                             "template": f"bsm_{a}_{b}", "seed": seed * 1000 + 500 + i})
    return {"nodes": nodes, "qconnections": qconnections, "cconnections": _classical_channels(net),
            "templates": templates, "formalism": BELL_DIAGONAL_STATE_FORMALISM}


def new_seed():
    """A fresh random seed, for when you don't pick one."""
    return int(np.random.default_rng().integers(2**31))


def to_sequence(net, seed=None, stop_time_s=None):
    """Translate a bellgame network into SeQUeNCe's own objects (a ``RouterNetTopo``).

    The dict says *what* the network is; SeQUeNCe needs Python objects that
    *behave* like it: a timeline (the simulation clock), a router per node with
    its memories, and the quantum and classical channels between them. This
    makes those objects, without running anything, so you can look inside.

    Each router also gets ``router.bellgame_pairs``: the Bell weights of the
    fresh pairs on each of its links. To run the topology yourself, make it and
    run it inside ``with bg.bell_diagonal_mode():``.

    Example:
        >>> import bellgame as bg
        >>> topo = bg.to_sequence(bg.two_player_network())
        >>> sorted(r.name for r in topo.get_nodes_by_type("QuantumRouter"))
        ['Alice', 'Bob']
    """
    config = sequence_config(net, new_seed() if seed is None else seed)
    if stop_time_s is not None:
        config["stop_time"] = int(stop_time_s * PS_PER_S)
    with bell_diagonal_mode():
        topo = RouterNetTopo(config)
    routers = {r.name: r for r in topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)}
    for name, router in routers.items():
        router.bellgame_pairs = {}
        # RouterNetTopo ignores gate/measurement fidelity in the config, so set them on the router itself
        p = node_params(net, name)
        router.gate_fid = float(p["gate_fidelity"])
        router.meas_fid = float(p["measurement_fidelity"])
    for a, b in links(net):
        weights = pair_weights(link_params(net, a, b))
        routers[a].bellgame_pairs[b] = weights
        routers[b].bellgame_pairs[a] = weights
    return topo


def _classical_channels(net):
    """Classical channels between every pair of nodes; delay = shortest fiber path / speed of light."""
    lengths = dict(nx.all_pairs_dijkstra_path_length(to_networkx(net), weight="length"))
    out = []
    for a, b in itertools.combinations(node_names(net), 2):
        km = lengths.get(a, {}).get(b, 0.0)
        out.append({"node1": a, "node2": b, "delay": max(1, int(round(km / FIBER_KM_PER_S * PS_PER_S)))})
    return out


# ---------------------------------------------------------------- the game


class _Player(RequestApp):
    """A player's app: keeps every pair shared with the other player until a question uses it."""

    def __init__(self, node):
        super().__init__(node)
        self.held = {}  # my memory name -> memory
        self.delivered = 0  # pairs that arrived, used or not
        self.counted = {}  # my memory name -> generation time of the last pair counted in it

    def partner(self, reservation):
        return reservation.responder if self.node.name == reservation.initiator else reservation.initiator

    def get_memory(self, info):
        # a pair arrives as "ENTANGLED", or as "PURIFIED" once purification has lifted it to the target
        if info.state not in ("ENTANGLED", "PURIFIED") or info.index not in self.memo_to_reservation:
            return
        reservation = self.memo_to_reservation[info.index]
        if info.remote_node == self.partner(reservation) and info.fidelity >= reservation.fidelity:
            # a memory can lose its pair and get a new one between two questions, while still in
            # self.held; a new generation time tells the new pair apart from the same pair reported twice
            memory = info.memory
            if self.counted.get(memory.name) != memory.generation_time:
                self.counted[memory.name] = memory.generation_time
                self.delivered += 1
            self.held[memory.name] = memory

    def forget_lost(self, partner):
        """Drop memories no longer holding a pair with ``partner`` (expired, or reused by SeQUeNCe)."""
        for name, memory in list(self.held.items()):
            if memory.entangled_memory["node_id"] != partner:
                del self.held[name]

    def release(self, memory):
        self.held.pop(memory.name, None)
        self.node.resource_manager.update(None, memory, "RAW")


class _Referee:
    """Asks a question every ``period_ps``; the players answer on a shared pair if they have one."""

    def __init__(self, timeline, alice, bob, period_ps, pick):
        self.timeline, self.alice, self.bob = timeline, alice, bob
        self.period_ps, self.pick = period_ps, pick
        self.questions = 0
        self.discarded = 0  # pairs both players held but let go unused (pick="newest")
        self.history = {"t_ps": [], "had_pair": [], "age_ps": [], "states": []}

    def shared_pairs(self):
        """Pairs whose two halves each name the other: like a real experiment, players never
        combine halves of two different pairs, even if one side's records are briefly out of date."""
        self.alice.forget_lost(self.bob.node.name)
        self.bob.forget_lost(self.alice.node.name)
        pairs = []
        for memory in self.alice.held.values():
            remote = self.bob.held.get(memory.entangled_memory["memo_id"])
            if remote is not None and remote.entangled_memory["memo_id"] == memory.name:
                pairs.append((memory, remote))
        return sorted(pairs, key=lambda pair: pair[0].generation_time)

    def ask(self):
        self.questions += 1
        pairs = self.shared_pairs()
        self.history["t_ps"].append(self.timeline.now())
        self.history["had_pair"].append(bool(pairs))
        if pairs:
            if self.pick == "newest":
                used, rest = pairs[-1], pairs[:-1]
            else:
                used, rest = pairs[0], []
            ma, mb = used
            ma.bds_decohere()
            mb.bds_decohere()
            # SeQUeNCe keeps the pair as its four Bell weights; record the full 4x4 state
            weights = np.clip(np.real(self.timeline.quantum_manager.get(ma.qstate_key).state), 0, None)
            self.history["states"].append(bell_diagonal(weights / np.sum(weights)))
            self.history["age_ps"].append(self.timeline.now() - ma.generation_time)
            self.discarded += len(rest)
            for a_mem, b_mem in [used] + rest:  # used pair is measured; older ones are let go
                self.alice.release(a_mem)
                self.bob.release(b_mem)
        else:
            self.history["states"].append(np.full((4, 4), np.nan))
            self.history["age_ps"].append(np.nan)
        self.timeline.schedule(Event(self.timeline.now() + self.period_ps, Process(self, "ask", [])))


@contextlib.contextmanager
def _recording_metrics():
    """Record SeQUeNCe's pair generation, swap and memory expiry events, and stop recording afterwards.

    Uses only SeQUeNCe's public metrics switches: ``configure`` starts a fresh store,
    ``enable`` picks the built-in metrics whose events we need, and ``enable([])``
    at the end records nothing from then on.
    """
    metrics.configure()
    # ES_METRIC: swap successes and failures; MEMORY_DECOHERENCE_RATE_METRIC: generations and expiries
    metrics.enable([metrics.ES_METRIC, metrics.MEMORY_DECOHERENCE_RATE_METRIC])
    try:
        yield metrics.storage
    finally:
        metrics.enable([])


def expected_path(net, alice, bob):
    """The path SeQUeNCe will route ``alice``'s request to ``bob`` through.

    SeQUeNCe, not bellgame, decides the route: this builds the SeQUeNCe topology
    and follows its routers' forwarding tables, hop by hop, as the request will.

    Example:
        >>> import bellgame as bg
        >>> bg.expected_path(bg.two_star("HubA", ["A1"], "HubB", ["B1"]), "A1", "B1")
        ['A1', 'HubA', 'HubB', 'B1']
    """
    return _route(to_sequence(net, seed=0), alice, bob)


def _route(topo, alice, bob):
    """Follow the forwarding tables of ``topo``'s routers from ``alice`` to ``bob``."""
    routers = {r.name: r for r in topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)}
    alice, bob = str(alice), str(bob)
    for name in (alice, bob):
        if name not in routers:
            raise ValueError(f"no node called '{name}' in the network")
    path = [alice]
    while path[-1] != bob:
        hop = routers[path[-1]].network_manager.get_forwarding_table().get(bob)
        if hop is None:
            raise ValueError(f"'{alice}' and '{bob}' are not connected")
        if hop in path:
            raise ValueError(f"SeQUeNCe's forwarding tables loop on the way from '{alice}' to '{bob}': {path + [hop]}")
        path.append(hop)
    return path


def path_memories(net, path):
    """How many memories SeQUeNCe can reserve on each link of ``path``.

    Every link on the path gets the same number. The two players use one memory
    per pair; a node in the middle needs one for each side, so twice as many.

    Example:
        >>> import bellgame as bg
        >>> path_memories(bg.two_player_network(), ["Alice", "Bob"])  # 10 memories each
        10
        >>> net = bg.two_star("HubA", ["A1"], "HubB", ["B1"])
        >>> path_memories(net, ["A1", "HubA", "HubB", "B1"])
        5
    """
    sizes = [int(node_params(net, n)["memory_size"]) for n in path]
    return min([sizes[0], sizes[-1]] + [m // 2 for m in sizes[1:-1]])


def run_network(net, alice, bob, questions_hz=1000.0, sim_time_s=1.0, seed=None, no_pair="discard",
                pick="newest"):
    """Simulate ``sim_time_s`` seconds of CHSH rounds between ``alice`` and ``bob`` over the network.

    * ``questions_hz``: how often the referee asks a question (more than 0, and at most
      one question per picosecond, SeQUeNCe's clock tick)
    * ``sim_time_s``: how long the game runs, in seconds (more than 0)
    * ``seed``: the same seed gives the same run; None picks a new random one
      (it is returned as ``run["seed"]``, so you can repeat a run you liked)
    * ``no_pair``: a round with no shared pair is ``"discard"``-ed, or the players
      answer at ``"random"`` (which counts against them)
    * ``pick``: which stored pair a question uses: the ``"newest"`` (older ones
      are thrown away) or the ``"oldest"`` (first in, first out)

    Returns a dict with:

    * ``state``: the average state the players measured (use it with ``bg.play``).
      All NaN if no round was played: with ``no_pair="discard"``, that is when no
      question found a pair, so there is nothing to average
    * ``fidelity``: how close that is to a perfect Bell pair
    * ``path``: the nodes SeQUeNCe routed through, and ``memories``: how many
      memories it reserved on each link of it
    * three rates: ``questions_hz`` (the referee's demand), ``pairs_hz`` (pairs the
      network delivered to the players: the supply), and ``rounds_hz``
      (questions answered with a pair). ``rounds_hz`` can't beat either of the
      other two. If it is close to ``questions_hz``, the network keeps up; if it is
      close to ``pairs_hz``, the players are waiting for pairs.
    * ``questions`` and ``rounds``: the same, as counts
    * ``pair_age_ms``: how long, on average, a used pair had been waiting
    * ``discarded_hz``: pairs the players held but threw away unused (``pick="newest"``)
    * ``links``: for each link on the path, its fresh-pair ``fidelity`` and
      ``pairs_hz``, how fast it makes pairs. The slowest link limits everything after it.
    * ``nodes``: for each node on the path, ``swaps_hz`` (successful swaps; zero at
      the players) and ``expired_hz`` (pairs lost because a memory ran out of time)
    * ``history``: one entry per question, as numpy arrays: ``t_ms`` (when it was
      asked), ``had_pair``, ``age_ms`` and ``states`` (the 4x4 state of the pair it
      used, all NaN when there was no pair; ``state`` above is the average of these).
      ``bg.play(strategy, run, rounds="run")`` plays these rounds one by one, and
      ``bg.fidelity`` takes the whole stack at once.
    * ``seed``: the seed this run used

    Example:
        >>> import bellgame as bg
        >>> run = bg.run_network(bg.two_player_network(50), "Alice", "Bob", sim_time_s=0.2, seed=1)
        >>> run["rounds_hz"] <= min(run["questions_hz"], run["pairs_hz"])
        True
        >>> len(run["history"]["t_ms"]) == run["questions"]
        True
    """
    questions_hz = _run_number("questions_hz", questions_hz, f"(0, {PS_PER_S:g}]")  # at most one question per ps
    sim_time_s = _run_number("sim_time_s", sim_time_s, "(0, inf)")
    if no_pair not in NO_PAIR_POLICIES:
        raise ValueError(f"no_pair must be one of {NO_PAIR_POLICIES}, got {no_pair!r}")
    if pick not in PICK_POLICIES:
        raise ValueError(f"pick must be one of {PICK_POLICIES}, got {pick!r}")
    seed = new_seed() if seed is None else int(seed)

    with bell_diagonal_mode(), _recording_metrics() as records:
        topo = to_sequence(net, seed=seed)
        # SeQUeNCe decides the route; we read it now because the reservation needs its size and length
        path = _route(topo, alice, bob)
        memories = path_memories(net, path)
        if memories < 1:
            raise ValueError(f"a node in the middle of the path {path} needs at least 2 memories (memory_size)")
        km = nx.path_weight(to_networkx(net), path, weight="length")

        # From here to the results, all times are in picoseconds, SeQUeNCe's unit.
        # Reserving the path needs a round trip between the players (2 one-way trips); allow twice that.
        start_ps = START_PS + 4 * int(km / FIBER_KM_PER_S * PS_PER_S)
        end_ps = start_ps + int(sim_time_s * PS_PER_S)
        period_ps = int(PS_PER_S / questions_hz)

        routers = {r.name: r for r in topo.get_nodes_by_type(RouterNetTopo.QUANTUM_ROUTER)}
        timeline = topo.get_timeline()
        timeline.stop_time = end_ps
        pa, pb = _Player(routers[str(alice)]), _Player(routers[str(bob)])
        referee = _Referee(timeline, pa, pb, period_ps, pick)
        timeline.init()
        pa.start(str(bob), start_ps, end_ps, memories, 1e-6)  # any fidelity: no purification
        timeline.schedule(Event(start_ps + period_ps, Process(referee, "ask", [])))
        timeline.run()

    if not pa.reservation_result:
        raise ValueError(f"SeQUeNCe could not reserve a path from '{alice}' to '{bob}'. Are they connected, "
                         "and does every node on the way have enough memories (memory_size)?")
    states = np.array(referee.history["states"], dtype=complex).reshape(-1, 4, 4)
    had_pair = np.array(referee.history["had_pair"], dtype=bool)
    used = states[had_pair]
    rounds = len(used)
    missed = referee.questions - rounds
    if no_pair == "random" and referee.questions:  # a guessed answer is as good as a fully mixed state
        state = (used.sum(axis=0) + missed * np.eye(4) / 4) / referee.questions
    elif rounds:
        state = used.mean(axis=0)
    else:
        state = np.full((4, 4), np.nan)  # no round was played: there is no data, so no state either

    # Back to the units people read: milliseconds for times, per second for rates.
    def per_s(count):
        return count / sim_time_s

    age_ms = np.array(referee.history["age_ps"], dtype=float) / PS_PER_MS
    made, swaps, expired = _event_counts(records)
    return {
        "path": list(pa.path),
        "memories": memories,
        "state": state,
        "fidelity": fidelity(state),
        "questions_hz": per_s(referee.questions),
        "pairs_hz": per_s(pa.delivered),
        "rounds_hz": per_s(rounds),
        "questions": referee.questions,
        "rounds": rounds,
        "pair_age_ms": float(np.nanmean(age_ms)) if rounds else float("nan"),
        "discarded_hz": per_s(referee.discarded),
        "links": [{"nodes": [u, v], "fidelity": routers[u].bellgame_pairs[v][0], "pairs_hz": per_s(made[u, v])}
                  for u, v in zip(pa.path, pa.path[1:])],
        "nodes": [{"node": n, "swaps_hz": per_s(swaps[n]), "expired_hz": per_s(expired[n])} for n in pa.path],
        "history": {
            "t_ms": np.array(referee.history["t_ps"], dtype=float) / PS_PER_MS,
            "had_pair": had_pair,
            "age_ms": age_ms,
            "states": states,
        },
        "no_pair": no_pair,
        "sim_time_s": sim_time_s,
        "seed": seed,
    }


def _run_number(name, value, interval):
    """``value`` as a float, or a ValueError if it is not a number inside ``interval``."""
    x = _as_number(name, value, float)
    if not _in_range(x, interval):
        raise ValueError(f"{name} must be in {interval}, got {value}")
    return x


def _event_counts(records):
    """Counts from SeQUeNCe's records: pairs made per (node, neighbour), and swaps and expiries per node."""
    # both ends of a link record each pair they make; keying by (owner, remote) keeps one per direction
    made = collections.Counter((r.owner_name, r.data.remote_node) for r in records.get_by_event(EventTypes.EG_SUCCESS))
    swaps = collections.Counter(r.owner_name for r in records.get_by_event(EventTypes.ES_SUCCESS))
    expired = collections.Counter(r.owner_name for r in records.get_by_event(EventTypes.MEMORY_EXPIRED))
    return made, swaps, expired
