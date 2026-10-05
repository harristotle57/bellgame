"""Describe a quantum network as one plain dict.

The dict is a valid SeQUeNCe ``RouterNetTopo`` configuration (so
``bg.build(net)`` can hand it straight to SeQUeNCe), stored in SeQUeNCe's own
units: meters, dB per meter, picoseconds. Each quantum connection also carries
a ``"photonics"`` sub-dict with the rest of the photonic link parameters
(source, detectors, misalignment, ...), which bellgame strips off before
SeQUeNCe sees the dict.

You can build the same network many ways:

* ``bg.from_matrix(names, dist_km)``: a numpy distance matrix (0 = no link)
* ``bg.from_edges([("A", "B", 10), ...])``: a list of (node, node, km)
* ``bg.from_networkx(G)``: a networkx graph with a ``length`` (km) on each edge
* ``bg.two_player_network``, ``bg.star``, ``bg.two_star``, ``bg.add_node``, ``bg.connect``

Example:
    >>> import bellgame as bg
    >>> net = bg.two_star("HubA", ["A1", "A2"], "HubB", ["B1", "B2"], leaf_km=5, hub_km=20)
    >>> names, dist = bg.to_matrix(net)
    >>> names
    ['HubA', 'A1', 'A2', 'HubB', 'B1', 'B2']
    >>> float(dist[0, 3])
    20.0
"""

import itertools

import networkx as nx
import numpy as np

from .defaults import DEFAULTS, default_values

KM = 1000.0  # meters
FIBER_KM_PER_S = 2e5  # light in fiber travels about 200,000 km/s
PS_PER_S = 1e12
LINK_ONLY_KEYS = ("distance_km", "loss_db_per_km")  # stored in SeQUeNCe's own fields
BSM_COUNT_RATE_HZ = 5e7
BSM_TIME_RESOLUTION_PS = 100


# ---------------------------------------------------------------- building blocks

def empty_network():
    """A network with no nodes yet.

    Example:
        >>> empty_network()["nodes"]
        []
    """
    return {
        "nodes": [],
        "qconnections": [],
        "cconnections": [],
        "templates": {},
        "formalism": "ket_vector",
    }


def add_node(net, name):
    """Add a quantum router (a node with quantum memories) called ``name``.

    Example:
        >>> net = add_node(empty_network(), "Alice")
        >>> node_names(net)
        ['Alice']
    """
    name = str(name)
    if name in node_names(net):
        raise ValueError(f"there is already a node called '{name}'")
    memory = default_values("node")
    net["nodes"].append({
        "name": name,
        "type": "QuantumRouter",
        "seed": len(net["nodes"]),
        "memo_size": memory["memory_size"],
        "template": _node_template_name(name),
    })
    net["templates"][_node_template_name(name)] = _memory_template(memory)
    _refresh_classical(net)
    return net


def connect(net, a, b, distance_km):
    """Add a fiber link between nodes ``a`` and ``b`` (``distance_km`` long).

    The photon source sits in the middle. Every other link parameter starts at
    its default from ``bg.DEFAULTS``; change it with ``bg.set_link``.

    Example:
        >>> net = add_node(add_node(empty_network(), "A"), "B")
        >>> net = connect(net, "A", "B", 12)
        >>> link_params(net, "A", "B")["distance_km"]
        12.0
    """
    a, b = str(a), str(b)
    names = node_names(net)
    for n in (a, b):
        if n not in names:
            raise ValueError(f"no node called '{n}'; add it first with bg.add_node")
    if a == b:
        raise ValueError("a link needs two different nodes")
    if _find_qconnection(net, a, b) is not None:
        raise ValueError(f"'{a}' and '{b}' are already connected; use bg.set_link to change it")
    if distance_km <= 0:
        raise ValueError(f"distance_km must be positive, got {distance_km}")
    # keep node1 as the node that was added first, so every way of building gives the same dict
    if names.index(a) > names.index(b):
        a, b = b, a
    params = default_values("link")
    photonics = {k: v for k, v in params.items() if k not in LINK_ONLY_KEYS}
    net["qconnections"].append({
        "node1": a,
        "node2": b,
        "attenuation": params["loss_db_per_km"] / KM,
        "distance": float(distance_km) * KM,
        "type": "meet_in_the_middle",
        "template": _bsm_template_name(a, b),
        "photonics": photonics,
    })
    net["qconnections"].sort(key=lambda q: (names.index(q["node1"]), names.index(q["node2"])))
    net["templates"][_bsm_template_name(a, b)] = _bsm_template(photonics)
    _refresh_classical(net)
    return net


def two_player_network(distance_km=10.0, alice="Alice", bob="Bob"):
    """The simplest network: Alice and Bob joined by one link.

    Example:
        >>> node_names(two_player_network())
        ['Alice', 'Bob']
    """
    net = add_node(add_node(empty_network(), alice), bob)
    return connect(net, alice, bob, distance_km)


def star(hub, leaves, leaf_km=5.0, net=None):
    """A star: every leaf node has one link to the hub. Pass ``net`` to add to an existing network.

    Example:
        >>> len(links(star("Hub", ["A", "B", "C"])))
        3
    """
    if net is None:
        net = empty_network()
    add_node(net, hub)
    for leaf in leaves:
        add_node(net, leaf)
        connect(net, hub, leaf, leaf_km)
    return net


def two_star(hub_a, leaves_a, hub_b, leaves_b, leaf_km=5.0, hub_km=20.0):
    """Two stars whose hubs are joined by one link. Leaves only connect to their own hub.

    Example:
        >>> net = two_star("HubA", ["A1"], "HubB", ["B1"])
        >>> links(net)
        [('HubA', 'A1'), ('HubA', 'HubB'), ('HubB', 'B1')]
    """
    net = star(hub_a, leaves_a, leaf_km)
    star(hub_b, leaves_b, leaf_km, net=net)
    return connect(net, hub_a, hub_b, hub_km)


# ---------------------------------------------------------------- other ways in

def from_edges(edges, names=None):
    """Build a network from a list of ``(node, node, distance_km)``.

    Nodes are added in the order they first appear, or in the order of ``names``.

    Example:
        >>> net = from_edges([("A", "B", 3), ("B", "C", 4)])
        >>> links(net)
        [('A', 'B'), ('B', 'C')]
    """
    if names is None:
        names = []
        for a, b, _ in edges:
            for n in (str(a), str(b)):
                if n not in names:
                    names.append(n)
    net = empty_network()
    for n in names:
        add_node(net, n)
    for a, b, km in edges:
        connect(net, a, b, km)
    return net


def from_matrix(names, dist_km):
    """Build a network from a symmetric distance matrix in km (0 means no link).

    Example:
        >>> d = np.array([[0, 5], [5, 0]])
        >>> links(from_matrix(["A", "B"], d))
        [('A', 'B')]
    """
    dist_km = np.asarray(dist_km, dtype=float)
    n = len(names)
    if dist_km.shape != (n, n):
        raise ValueError(f"distance matrix must be {n} x {n} for {n} names, got {dist_km.shape}")
    if not np.allclose(dist_km, dist_km.T):
        i, j = np.argwhere(~np.isclose(dist_km, dist_km.T))[0]
        raise ValueError(f"distance matrix must be symmetric: dist[{i}, {j}] = {dist_km[i, j]} "
                         f"but dist[{j}, {i}] = {dist_km[j, i]}")
    if np.any(np.diag(dist_km) != 0):
        raise ValueError("the diagonal of the distance matrix must be 0 (a node has no link to itself)")
    if np.any(dist_km < 0):
        raise ValueError("distances can't be negative")
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if dist_km[i, j] > 0:
                edges.append((names[i], names[j], float(dist_km[i, j])))
    return from_edges(edges, names=[str(x) for x in names])


def from_networkx(graph):
    """Build a network from a networkx graph; each edge's ``length`` attribute is its km.

    This lets you use SeQUeNCe's graph builders in ``sequence.utils.graphs``.

    Example:
        >>> G = nx.Graph()
        >>> G.add_edge("A", "B", length=7)
        >>> link_params(from_networkx(G), "A", "B")["distance_km"]
        7.0
    """
    edges = []
    for a, b, data in graph.edges(data=True):
        if "length" not in data:
            raise ValueError(f"edge ({a}, {b}) has no 'length' (km) attribute")
        edges.append((str(a), str(b), float(data["length"])))
    return from_edges(edges, names=[str(n) for n in graph.nodes])


def to_matrix(net):
    """(names, dist_km): the network as a node list and a distance matrix in km.

    Example:
        >>> names, d = to_matrix(two_player_network(10))
        >>> d.tolist()
        [[0.0, 10.0], [10.0, 0.0]]
    """
    names = node_names(net)
    dist = np.zeros((len(names), len(names)))
    for q in net["qconnections"]:
        i, j = names.index(q["node1"]), names.index(q["node2"])
        dist[i, j] = dist[j, i] = q["distance"] / KM
    return names, dist


def to_networkx(net):
    """The network as a networkx Graph with ``length`` (km) on each edge."""
    g = nx.Graph()
    g.add_nodes_from(node_names(net))
    for q in net["qconnections"]:
        g.add_edge(q["node1"], q["node2"], length=q["distance"] / KM)
    return g


# ---------------------------------------------------------------- changing knobs

def set_link(net, a, b, **changes):
    """Change parameters of the link between ``a`` and ``b`` (any link parameter in bg.DEFAULTS).

    Example:
        >>> net = set_link(two_player_network(), "Alice", "Bob", mean_photon_number=0.05)
        >>> link_params(net, "Alice", "Bob")["mean_photon_number"]
        0.05
    """
    q = _require_qconnection(net, a, b)
    allowed = default_values("link")
    for name, value in changes.items():
        if name not in allowed:
            raise ValueError(f"'{name}' is not a link parameter. Link parameters are: "
                             f"{', '.join(sorted(allowed))}")
        if name == "distance_km":
            if value <= 0:
                raise ValueError(f"distance_km must be positive, got {value}")
            q["distance"] = float(value) * KM
        elif name == "loss_db_per_km":
            q["attenuation"] = float(value) / KM
        else:
            q["photonics"][name] = value
    net["templates"][q["template"]] = _bsm_template(q["photonics"])
    _refresh_classical(net)
    return net


def set_all_links(net, **changes):
    """Change the same parameters on every link.

    Example:
        >>> net = set_all_links(star("H", ["A", "B"]), detector_efficiency=0.9)
    """
    for a, b in links(net):
        set_link(net, a, b, **changes)
    return net


def set_source(net, a, b, mean_photon_number):
    """Change the photon source brightness on one link."""
    return set_link(net, a, b, mean_photon_number=mean_photon_number)


def set_detectors(net, a, b, detector_efficiency=None, dark_count_rate_hz=None):
    """Change the detectors on one link (leave a value as None to keep it)."""
    changes = {}
    if detector_efficiency is not None:
        changes["detector_efficiency"] = detector_efficiency
    if dark_count_rate_hz is not None:
        changes["dark_count_rate_hz"] = dark_count_rate_hz
    return set_link(net, a, b, **changes)


def set_memory(net, node=None, **changes):
    """Change quantum-memory parameters of one node, or of every node when ``node`` is None.

    Example:
        >>> net = set_memory(two_player_network(), "Alice", coherence_time_ms=5)
        >>> memory_params(net, "Alice")["coherence_time_ms"]
        5
    """
    allowed = default_values("node")
    targets = node_names(net) if node is None else [str(node)]
    for name in targets:
        n = _require_node(net, name)
        memory = memory_params(net, name)
        for key, value in changes.items():
            if key not in allowed:
                raise ValueError(f"'{key}' is not a memory parameter. Memory parameters are: "
                                 f"{', '.join(sorted(allowed))}")
            memory[key] = value
        n["memo_size"] = int(memory["memory_size"])
        net["templates"][n["template"]] = _memory_template(memory)
    return net


# ---------------------------------------------------------------- reading

def node_names(net):
    """List of node names, in the order they were added."""
    return [n["name"] for n in net["nodes"]]


def links(net):
    """List of (node1, node2) pairs that have a fiber link."""
    return [(q["node1"], q["node2"]) for q in net["qconnections"]]


def link_params(net, a, b):
    """The full photonic link dict (like ``bg.link(...)``) for the link between ``a`` and ``b``.

    You can pass it to ``bg.play_chsh`` to play over that single link.
    """
    q = _require_qconnection(net, a, b)
    params = dict(q["photonics"])
    params["distance_km"] = q["distance"] / KM
    params["loss_db_per_km"] = q["attenuation"] * KM
    return params


def memory_params(net, node):
    """Memory parameters of one node, in bellgame units."""
    n = _require_node(net, node)
    mem = net["templates"][n["template"]]["MemoryArray"]
    coherence_s = mem["coherence_time"]
    return {
        "memory_efficiency": mem["efficiency"],
        "memory_frequency_hz": mem["frequency"],
        "coherence_time_ms": float("inf") if coherence_s < 0 else _clean(coherence_s * 1000),
        "memory_size": n["memo_size"],
    }


def parameters(net):
    """Print every knob in the network: each node's memory and each link's photonics."""
    print("Nodes (quantum memories):")
    for name in node_names(net):
        print(f"  {name}")
        for key, value in memory_params(net, name).items():
            entry = DEFAULTS[key]
            print(f"      {key:24s} = {value!s:12s} {entry['unit']:8s} {entry['meaning']}")
    print("Links (photon source, fiber, detectors):")
    for a, b in links(net):
        print(f"  {a} -- {b}")
        for key, value in link_params(net, a, b).items():
            entry = DEFAULTS[key]
            print(f"      {key:24s} = {value!s:18s} {entry['unit']:8s} {entry['meaning']}")


# ---------------------------------------------------------------- internals

def _clean(x):
    """Round away floating-point fuzz (5.000000000000001 -> 5)."""
    y = round(x, 9)
    return int(y) if y == int(y) else y


def _node_template_name(name):
    return f"memory_{name}"


def _bsm_template_name(a, b):
    return f"bsm_{a}_{b}"


def _memory_template(memory):
    coherence_ms = memory["coherence_time_ms"]
    coherence_s = -1 if coherence_ms == float("inf") else coherence_ms / 1000
    return {"MemoryArray": {
        "frequency": float(memory["memory_frequency_hz"]),
        "coherence_time": coherence_s,
        "efficiency": float(memory["memory_efficiency"]),
        "fidelity": 1.0,
    }}


def _bsm_template(photonics):
    detector = {
        "efficiency": float(photonics["detector_efficiency"]),
        "dark_count": float(photonics["dark_count_rate_hz"]),
        "count_rate": BSM_COUNT_RATE_HZ,
        "time_resolution": BSM_TIME_RESOLUTION_PS,
    }
    return {"SingleAtomBSM": {"detectors": [dict(detector), dict(detector)]}}


def _find_qconnection(net, a, b):
    for q in net["qconnections"]:
        if {q["node1"], q["node2"]} == {str(a), str(b)}:
            return q
    return None


def _require_qconnection(net, a, b):
    q = _find_qconnection(net, a, b)
    if q is None:
        raise ValueError(f"there is no link between '{a}' and '{b}'")
    return q


def _require_node(net, name):
    for n in net["nodes"]:
        if n["name"] == str(name):
            return n
    raise ValueError(f"no node called '{name}'")


def _refresh_classical(net):
    """Classical channels between every pair of nodes, delay = shortest fiber path / speed of light."""
    graph = to_networkx(net)
    lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="length"))
    cconnections = []
    for a, b in itertools.combinations(node_names(net), 2):
        km = lengths.get(a, {}).get(b, 0.0)
        delay_ps = int(round(km / FIBER_KM_PER_S * PS_PER_S)) if km > 0 else 1
        cconnections.append({"node1": a, "node2": b, "delay": max(delay_ps, 1)})
    net["cconnections"] = cconnections
