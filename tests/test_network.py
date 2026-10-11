import copy

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


# ---------------------------------------------------------------- describing networks

def test_every_way_in_builds_the_same_dict():
    ref = builder_net()
    from_matrix = bg.from_matrix(NAMES, two_star_matrix())
    edges = [("HubA", a, 5) for a in ["A1", "A2", "A3"]] + [("HubB", b, 5) for b in ["B1", "B2", "B3"]]
    edges.append(("HubA", "HubB", 20))
    from_edges = bg.from_edges(edges, names=NAMES)
    g = nx.Graph()
    g.add_nodes_from(NAMES)
    g.add_weighted_edges_from(edges, weight="length")
    assert from_matrix == ref
    assert from_edges == ref
    assert bg.from_networkx(g) == ref


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


def test_matrix_inf_means_no_link():
    d = two_star_matrix()
    d[d == 0] = np.inf
    np.fill_diagonal(d, 0)
    assert bg.from_matrix(NAMES, d) == builder_net()
    d[0, 1] = d[1, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        bg.from_matrix(NAMES, d)


def test_from_networkx_keeps_sequence_attenuation():
    from sequence.utils.graphs import build_star
    net = bg.from_networkx(build_star(2, length=7, attenuation=0.0003))
    assert bg.links(net) == [("0", "1"), ("0", "2")]
    p = bg.link_params(net, "0", "1")
    assert p["distance_km"] == 7 and np.isclose(p["loss_db_per_km"], 0.3)


def test_set_link_and_node():
    net = builder_net()
    bg.set_link(net, "A1", "HubA", distance_km=7, link_model="fock", mean_photon_number=0.05)
    p = bg.link_params(net, "HubA", "A1")
    assert p["distance_km"] == 7 and p["link_model"] == "fock" and p["mean_photon_number"] == 0.05
    bg.set_node(net, coherence_time_ms=50, memory_size=6)
    assert bg.node_params(net, "B3")["coherence_time_ms"] == 50
    assert bg.node_params(net, "HubA")["memory_size"] == 6
    with pytest.raises(ValueError, match="not a link parameter"):
        bg.set_link(net, "A1", "HubA", colour="red")
    with pytest.raises(ValueError, match="link_model"):
        bg.set_link(net, "A1", "HubA", link_model="magic")
    with pytest.raises(ValueError, match="memory_errors"):
        bg.set_node(net, "A1", memory_errors=[1, 1, 1])


def test_failed_edits_leave_the_network_unchanged():
    net = builder_net()
    before = copy.deepcopy(net)
    with pytest.raises(ValueError, match="raw_fidelity"):
        bg.set_link(net, "HubA", "A1", detector_efficiency=0.5, raw_fidelity=2)
    with pytest.raises(ValueError, match="not a node parameter"):
        bg.set_node(net, coherence_time_ms=5, colour="red")
    with pytest.raises(ValueError, match="distance_km"):
        bg.connect(net, "A1", "B1", -1)
    with pytest.raises(ValueError, match="already a node"):
        bg.star("New", ["A1"], net=net)
    assert net == before
    bg.set_link(net, "HubA", "A1", detector_efficiency=0.5)  # the failed changes left nothing behind


def test_copy_leaves_the_original_alone():
    net = builder_net()
    before = copy.deepcopy(net)
    changed = bg.set_node(net, memory_errors=[1, 0, 0], copy=True)
    changed = bg.remove_node(bg.disconnect(changed, "HubA", "HubB", copy=True), "B1", copy=True)
    assert net == before
    assert "B1" not in bg.node_names(changed) and ("HubA", "HubB") not in bg.links(changed)
    assert bg.node_params(changed, "A1")["memory_errors"] == [1, 0, 0]
    nodes = changed["nodes"]
    assert nodes["A1"]["memory_errors"] is not nodes["A2"]["memory_errors"]  # each node gets its own list


def test_remove_node_drops_its_links():
    net = bg.remove_node(builder_net(), "HubB")
    assert bg.links(net) == [("HubA", "A1"), ("HubA", "A2"), ("HubA", "A3")]
    with pytest.raises(ValueError, match="no link"):
        bg.disconnect(net, "A1", "A2")


def test_values_are_stored_as_floats_and_ints():
    net = bg.set_node(builder_net(), "A1", memory_size=np.float64(4.0), coherence_time_ms=np.int64(5),
                      memory_errors=np.array([0, 0, 1]))
    p = bg.node_params(net, "A1")
    assert type(p["memory_size"]) is int and type(p["coherence_time_ms"]) is float
    assert p["memory_errors"] == [0.0, 0.0, 1.0] and all(type(x) is float for x in p["memory_errors"])
    with pytest.raises(ValueError, match="whole number"):
        bg.set_node(net, memory_size=1.5)
    with pytest.raises(ValueError, match="True or False"):
        bg.set_link(net, "HubA", "A1", heralded="no")


@pytest.mark.parametrize("change", [
    lambda net: bg.connect(net, "A1", "B1", np.nan),
    lambda net: bg.set_link(net, "HubA", "A1", detector_efficiency=np.nan),
    lambda net: bg.set_node(net, coherence_time_ms=float("nan")),
    lambda net: bg.set_node(net, "A1", memory_errors=[np.nan, 0, 1]),
])
def test_nan_is_always_refused(change):
    with pytest.raises(ValueError, match="NaN"):
        change(builder_net())


def test_ranges_are_checked():
    net = builder_net()
    with pytest.raises(ValueError, match=r"detector_efficiency must be in \[0, 1\]"):
        bg.set_all_links(net, detector_efficiency=7)
    with pytest.raises(ValueError, match="gate_fidelity"):
        bg.set_node(net, gate_fidelity=3)
    with pytest.raises(ValueError, match="distance_km"):
        bg.from_edges([("A", "B", np.inf)])
    bg.set_node(net, coherence_time_ms=np.inf)  # a perfect memory is allowed
