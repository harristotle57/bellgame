"""Describe a quantum network as one plain dict.

A network is::

    {"nodes": {"Alice": {node parameters}, ...},
     "links": [{"nodes": ["Alice", "Bob"], link parameters}, ...]}

Every parameter starts at its default from ``bg.DEFAULTS`` (node parameters
for nodes; fiber, detector, pair and source parameters for links). Change them
with ``bg.set_node`` and ``bg.set_link``. ``bg.run_network`` hands the network
to SeQUeNCe; ``bg.to_sequence`` shows you the SeQUeNCe topology it becomes.

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

import networkx as nx
import numpy as np

from .defaults import DEFAULTS, LINK_GROUPS, default_values

# ---------------------------------------------------------------- building blocks


def empty_network():
    """A network with no nodes yet.

    Example:
        >>> empty_network()
        {'nodes': {}, 'links': []}
    """
    return {"nodes": {}, "links": []}


def add_node(net, name):
    """Add a quantum router (a node with quantum memories) called ``name``.

    Example:
        >>> node_names(add_node(empty_network(), "Alice"))
        ['Alice']
    """
    name = str(name)
    if name in net["nodes"]:
        raise ValueError(f"there is already a node called '{name}'")
    net["nodes"][name] = default_values("node")
    return net


def connect(net, a, b, distance_km):
    """Add a fiber link between nodes ``a`` and ``b`` (``distance_km`` long).

    Every other link parameter starts at its default; change it with ``bg.set_link``.

    Example:
        >>> net = add_node(add_node(empty_network(), "A"), "B")
        >>> link_params(connect(net, "A", "B", 12), "A", "B")["distance_km"]
        12.0
    """
    a, b = str(a), str(b)
    for n in (a, b):
        _require_node(net, n)
    if a == b:
        raise ValueError("a link needs two different nodes")
    if _find_link(net, a, b) is not None:
        raise ValueError(f"'{a}' and '{b}' are already connected; use bg.set_link to change it")
    names = node_names(net)
    if names.index(a) > names.index(b):  # same dict whichever way round you connect
        a, b = b, a
    link = {"nodes": [a, b]}
    link.update(default_values(*LINK_GROUPS))
    net["links"].append(link)
    net["links"].sort(key=lambda q: (names.index(q["nodes"][0]), names.index(q["nodes"][1])))
    return set_link(net, a, b, distance_km=distance_km)


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
        >>> links(star("Hub", ["A", "B", "C"]))
        [('Hub', 'A'), ('Hub', 'B'), ('Hub', 'C')]
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
        >>> links(two_star("HubA", ["A1"], "HubB", ["B1"]))
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
        >>> links(from_edges([("A", "B", 3), ("B", "C", 4)]))
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
    """Build a network from a symmetric distance matrix in km (0 or ``np.inf`` means no link).

    Example:
        >>> links(from_matrix(["A", "B", "C"], np.array([[0, 5, 0], [5, 0, np.inf], [0, np.inf, 0]])))
        [('A', 'B')]
    """
    dist_km = np.asarray(dist_km, dtype=float)
    n = len(names)
    if dist_km.shape != (n, n):
        raise ValueError(f"distance matrix must be {n} x {n} for {n} names, got {dist_km.shape}")
    if np.any(np.isnan(dist_km)):
        i, j = np.argwhere(np.isnan(dist_km))[0]
        raise ValueError(f"dist[{i}, {j}] is NaN; use 0 or np.inf for no link")
    dist_km = np.where(np.isinf(dist_km) & (dist_km > 0), 0.0, dist_km)
    if not np.allclose(dist_km, dist_km.T):
        i, j = np.argwhere(~np.isclose(dist_km, dist_km.T))[0]
        raise ValueError(f"distance matrix must be symmetric: dist[{i}, {j}] = {dist_km[i, j]} "
                         f"but dist[{j}, {i}] = {dist_km[j, i]}")
    if np.any(np.diag(dist_km) != 0):
        raise ValueError("the diagonal of the distance matrix must be 0 (a node has no link to itself)")
    if np.any(dist_km < 0):
        raise ValueError("distances can't be negative")
    edges = [(names[i], names[j], float(dist_km[i, j]))
             for i in range(n) for j in range(i + 1, n) if dist_km[i, j] > 0]
    return from_edges(edges, names=[str(x) for x in names])


def from_networkx(graph):
    """Build a network from a networkx graph; each edge's ``length`` attribute is its km.

    This lets you use SeQUeNCe's graph builders in ``sequence.utils.graphs``. An
    edge's ``attenuation`` (dB/m, as those builders set it) becomes ``loss_db_per_km``.

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
    net = from_edges(edges, names=[str(n) for n in graph.nodes])
    for a, b, data in graph.edges(data=True):
        if "attenuation" in data:
            set_link(net, a, b, loss_db_per_km=float(data["attenuation"]) * 1000)
    return net


def to_matrix(net):
    """(names, dist_km): the network as a node list and a distance matrix in km.

    Example:
        >>> names, d = to_matrix(two_player_network(10))
        >>> d.tolist()
        [[0.0, 10.0], [10.0, 0.0]]
    """
    names = node_names(net)
    dist = np.zeros((len(names), len(names)))
    for q in net["links"]:
        i, j = (names.index(n) for n in q["nodes"])
        dist[i, j] = dist[j, i] = q["distance_km"]
    return names, dist


def to_networkx(net):
    """The network as a networkx Graph with ``length`` (km) on each edge."""
    g = nx.Graph()
    g.add_nodes_from(node_names(net))
    for q in net["links"]:
        g.add_edge(*q["nodes"], length=q["distance_km"])
    return g


# ---------------------------------------------------------------- changing knobs


def set_link(net, a, b, **changes):
    """Change parameters of the link between ``a`` and ``b``.

    Example:
        >>> net = set_link(two_player_network(), "Alice", "Bob", raw_fidelity=0.9)
        >>> link_params(net, "Alice", "Bob")["raw_fidelity"]
        0.9
    """
    q = _require_link(net, a, b)
    allowed = default_values(*LINK_GROUPS)
    for name, value in changes.items():
        if name not in allowed:
            raise ValueError(f"'{name}' is not a link parameter. Link parameters are: "
                             f"{', '.join(sorted(allowed))}")
        q[name] = list(value) if isinstance(value, (list, tuple)) else value
    q["distance_km"] = float(q["distance_km"])
    _check_link(q)
    return net


def set_all_links(net, **changes):
    """Change the same parameters on every link.

    Example:
        >>> net = set_all_links(star("H", ["A", "B"]), detector_efficiency=0.9)
    """
    for a, b in links(net):
        set_link(net, a, b, **changes)
    return net


def set_node(net, node=None, **changes):
    """Change parameters of one node, or of every node when ``node`` is None.

    Example:
        >>> net = set_node(two_player_network(), "Alice", coherence_time_ms=5)
        >>> node_params(net, "Alice")["coherence_time_ms"]
        5
    """
    allowed = default_values("node")
    targets = node_names(net) if node is None else [str(node)]
    for name in targets:
        params = _require_node(net, name)
        for key, value in changes.items():
            if key not in allowed:
                raise ValueError(f"'{key}' is not a node parameter. Node parameters are: "
                                 f"{', '.join(sorted(allowed))}")
            params[key] = list(value) if isinstance(value, (list, tuple)) else value
        _check_node(name, params)
    return net


# ---------------------------------------------------------------- reading


def node_names(net):
    """List of node names, in the order they were added."""
    return list(net["nodes"])


def links(net):
    """List of (node1, node2) pairs that have a fiber link."""
    return [tuple(q["nodes"]) for q in net["links"]]


def link_params(net, a, b):
    """A copy of the parameters of the link between ``a`` and ``b``."""
    q = _require_link(net, a, b)
    return {k: (list(v) if isinstance(v, list) else v) for k, v in q.items() if k != "nodes"}


def node_params(net, node):
    """A copy of the parameters of one node."""
    return {k: (list(v) if isinstance(v, list) else v) for k, v in _require_node(net, node).items()}


def parameters(net):
    """Print every knob in the network, node by node and link by link."""
    print("Nodes (quantum memories and swapping):")
    for name in node_names(net):
        print(f"  {name}")
        _print_params(node_params(net, name))
    print("Links (fiber, detectors, pair quality):")
    for a, b in links(net):
        print(f"  {a} -- {b}")
        _print_params(link_params(net, a, b))


# ---------------------------------------------------------------- internals

LINK_MODELS = ("fixed", "analytic", "fock")


def _print_params(params):
    for key, value in params.items():
        entry = DEFAULTS[key]
        print(f"      {key:22s} = {value!s:16s} {entry['unit']:8s} {entry['meaning']}")


def _check_link(q):
    name = f"link {q['nodes'][0]} -- {q['nodes'][1]}"
    if q["distance_km"] <= 0:
        raise ValueError(f"{name}: distance_km must be positive, got {q['distance_km']}")
    if q["link_model"] not in LINK_MODELS:
        raise ValueError(f"{name}: link_model must be one of {LINK_MODELS}, got {q['link_model']!r}")
    if not 0.25 <= q["raw_fidelity"] <= 1:
        raise ValueError(f"{name}: raw_fidelity must be between 0.25 and 1, got {q['raw_fidelity']}")
    _check_shares(name, "raw_errors", q["raw_errors"])


def _check_node(name, params):
    if int(params["memory_size"]) < 1:
        raise ValueError(f"node {name}: memory_size must be at least 1")
    if params["coherence_time_ms"] <= 0:
        raise ValueError(f"node {name}: coherence_time_ms must be positive (use float('inf') for a perfect memory)")
    _check_shares(f"node {name}", "memory_errors", params["memory_errors"])


def _check_shares(where, key, shares):
    if len(shares) != 3 or min(shares) < 0 or not np.isclose(sum(shares), 1):
        raise ValueError(f"{where}: {key} must be three non-negative numbers (X, Y, Z) adding up to 1, got {shares}")


def _find_link(net, a, b):
    for q in net["links"]:
        if set(q["nodes"]) == {str(a), str(b)}:
            return q
    return None


def _require_link(net, a, b):
    q = _find_link(net, a, b)
    if q is None:
        raise ValueError(f"there is no link between '{a}' and '{b}'")
    return q


def _require_node(net, name):
    if str(name) not in net["nodes"]:
        raise ValueError(f"no node called '{name}'")
    return net["nodes"][str(name)]
