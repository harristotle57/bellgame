import networkx as nx
import numpy as np
import pytest

import bellgame as bg

NAMES = ["HubA", "A1", "A2", "A3", "HubB", "B1", "B2", "B3"]


def two_star_matrix(leaf_km=5, hub_km=20):
    d = np.zeros((8, 8))
    d[0, 1:4] = d[1:4, 0] = leaf_km
    d[4, 5:8] = d[5:8, 4] = leaf_km
    d[0, 4] = d[4, 0] = hub_km
    return d


def builder_net():
    return bg.two_star("HubA", ["A1", "A2", "A3"], "HubB", ["B1", "B2", "B3"], leaf_km=5, hub_km=20)


def test_every_way_in_builds_the_same_dict():
    ref = builder_net()
    from_matrix = bg.from_matrix(NAMES, two_star_matrix())
    edges = [("HubA", a, 5) for a in ["A1", "A2", "A3"]] + [("HubB", b, 5) for b in ["B1", "B2", "B3"]]
    edges.append(("HubA", "HubB", 20))
    from_edges = bg.from_edges(edges, names=NAMES)
    g = nx.Graph()
    g.add_nodes_from(NAMES)
    g.add_weighted_edges_from(edges, weight="length")
    from_nx = bg.from_networkx(g)
    assert from_matrix == ref
    assert from_edges == ref
    assert from_nx == ref


def test_to_matrix_round_trip():
    names, d = bg.to_matrix(builder_net())
    assert names == NAMES
    assert np.allclose(d, two_star_matrix())
    assert bg.from_matrix(names, d) == builder_net()


def test_matrix_must_be_symmetric():
    d = two_star_matrix()
    d[0, 1] = 6
    with pytest.raises(ValueError, match="symmetric"):
        bg.from_matrix(NAMES, d)


def test_set_link_and_memory():
    net = builder_net()
    bg.set_link(net, "A1", "HubA", distance_km=7, mean_photon_number=0.05)
    p = bg.link_params(net, "HubA", "A1")
    assert p["distance_km"] == 7 and p["mean_photon_number"] == 0.05
    assert net["templates"]["bsm_HubA_A1"]["SingleAtomBSM"]["detectors"][0]["efficiency"] == 0.8
    bg.set_memory(net, coherence_time_ms=50, memory_size=6)
    assert bg.memory_params(net, "B3")["coherence_time_ms"] == 50
    assert net["nodes"][-1]["memo_size"] == 6
    with pytest.raises(ValueError, match="not a link parameter"):
        bg.set_link(net, "A1", "HubA", colour="red")


def test_build_gives_sequence_topology_without_photonics():
    net = builder_net()
    topo = bg.build(net)
    assert sorted(r.name for r in topo.get_nodes_by_type("QuantumRouter")) == sorted(NAMES)
    assert "photonics" in net["qconnections"][0]  # original dict untouched


@pytest.fixture(scope="module")
def two_star_path():
    return bg.end_to_end(builder_net(), "A1", "B2", seed=1)


def test_two_star_routes_through_both_hubs(two_star_path):
    assert two_star_path["path"] == ["A1", "HubA", "HubB", "B2"]
    assert two_star_path["pairs"] > 10
    assert set(two_star_path["wait_ms"]) == {"A1", "HubA", "HubB", "B2"}


def test_two_star_s_below_single_link(two_star_path):
    net = builder_net()
    s_path = bg.play_chsh(bg.optimal_strategy(), two_star_path)["S"]
    s_link = bg.play_chsh(bg.optimal_strategy(), bg.link_params(net, "A1", "HubA"))["S"]
    assert 2 < s_path < s_link


def test_rate_falls_with_hub_distance():
    rates = []
    for hub_km in [10, 60]:
        net = bg.two_star("HubA", ["A1"], "HubB", ["B1"], leaf_km=5, hub_km=hub_km)
        rates.append(bg.end_to_end(net, "A1", "B1", seed=2)["rate_hz"])
    assert rates[0] > rates[1]


def test_compose_path_matches_qubit_swap():
    a, b = bg.bell_pair(0.9), bg.bell_pair(0.95)
    zero = {"A": 0, "H": 0, "B": 0}
    out = bg.compose_path([a, b], ["A", "H", "B"], zero, {"A": 1, "H": 1, "B": 1})
    assert np.allclose(out, bg.swap(a, b))
