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


def test_to_sequence_gives_topology_with_link_states():
    topo = bg.to_sequence(builder_net())
    routers = {r.name: r for r in topo.get_nodes_by_type("QuantumRouter")}
    assert sorted(routers) == sorted(NAMES)
    assert set(routers["HubA"].bellgame_pairs) == {"A1", "A2", "A3", "HubB"}


# ---------------------------------------------------------------- link models

def test_fixed_pair_weights_follow_raw_errors():
    link = bg.link_params(bg.two_player_network(), "Alice", "Bob")
    link.update(link_model="fixed", raw_fidelity=0.9, raw_errors=[0, 0, 1])
    assert np.allclose(bg.pair_weights(link), [0.9, 0.1, 0, 0])


def test_analytic_dark_counts_hurt_long_links():
    net = bg.two_player_network(1)
    bg.set_link(net, "Alice", "Bob", dark_count_rate_hz=1e5)
    near = bg.pair_weights(bg.link_params(net, "Alice", "Bob"))
    bg.set_link(net, "Alice", "Bob", distance_km=300)
    far = bg.pair_weights(bg.link_params(net, "Alice", "Bob"))
    assert near[0] > 0.94 and far[0] < 0.8
    assert np.isclose(sum(far), 1)


def test_fock_weights_reproduce_link_correlations():
    params = bg.link(distance_km=20, mean_photon_number=0.05)
    weights, _ = bg.fock_pair_weights(params)
    s_direct = bg.play_chsh(bg.optimal_strategy(), params)["S"]
    s_weights = bg.play_chsh(bg.optimal_strategy(), bg.bell_diagonal(weights))["S"]
    # SeQUeNCe stores only the Bell-diagonal part; the tiny rest (from double clicks) is dropped
    assert np.isclose(s_direct, s_weights, atol=1e-3)


def test_analytic_matches_fock_dark_count_limit():
    # With a faint source the Fock model's only noise is dark counts, like the analytic model.
    for km in (50, 200):
        net = bg.two_player_network(km)
        bg.set_link(net, "Alice", "Bob", dark_count_rate_hz=1e5, raw_fidelity=1.0, mean_photon_number=1e-4,
                    heralded=True)
        link = bg.link_params(net, "Alice", "Bob")
        analytic = bg.analytic_pair_weights(link)[0]
        fock = bg.fock_pair_weights(link)[0][0]
        assert abs(analytic - fock) < 0.01, (km, analytic, fock)


# ---------------------------------------------------------------- playing over SeQUeNCe

@pytest.fixture(scope="module")
def two_star_run():
    return bg.run_network(builder_net(), "A1", "B2", sim_time_s=0.3, seed=1)


def test_two_star_routes_through_both_hubs(two_star_run):
    assert two_star_run["path"] == ["A1", "HubA", "HubB", "B2"]
    assert two_star_run["rounds"] > 20
    assert two_star_run["questions"] >= two_star_run["rounds"]


def test_two_star_matches_swap_theory_without_decay():
    # Three links of fidelity F joined by two (Werner-twirled) swaps: visibility multiplies.
    net = builder_net()
    bg.set_all_links(net, link_model="fixed", raw_fidelity=0.95)
    bg.set_node(net, coherence_time_ms=float("inf"))
    run = bg.run_network(net, "A1", "B2", sim_time_s=0.3, seed=1)
    v = (4 * 0.95 - 1) / 3
    assert np.isclose(run["fidelity"], (3 * v**3 + 1) / 4, atol=1e-6)


def test_gate_fidelity_reaches_the_swaps():
    # Werner-twirled swap with gate fidelity g: visibility picks up a factor g per swap.
    net = builder_net()
    bg.set_all_links(net, link_model="fixed", raw_fidelity=0.95)
    bg.set_node(net, coherence_time_ms=float("inf"), gate_fidelity=0.9)
    topo = bg.to_sequence(net, seed=1)
    assert all(r.gate_fid == 0.9 for r in topo.get_nodes_by_type("QuantumRouter"))
    run = bg.run_network(net, "A1", "B2", sim_time_s=0.3, seed=1)
    v = (4 * 0.95 - 1) / 3
    assert np.isclose(run["fidelity"], (3 * 0.9**2 * v**3 + 1) / 4, atol=1e-6)


def test_memory_decay_lowers_s():
    s = []
    for coherence_ms in (float("inf"), 5):
        net = bg.two_player_network(20)
        bg.set_node(net, coherence_time_ms=coherence_ms)
        run = bg.run_network(net, "Alice", "Bob", questions_hz=200, sim_time_s=0.5, seed=1, pick="oldest")
        s.append(bg.play_chsh(bg.optimal_strategy(), run)["S"])
    assert s[0] > s[1]


def test_missing_pairs_count_against_random_policy():
    net = bg.two_player_network(60)
    kept = bg.run_network(net, "Alice", "Bob", sim_time_s=0.2, seed=3)
    guessed = bg.run_network(net, "Alice", "Bob", sim_time_s=0.2, seed=3, no_pair="random")
    assert kept["rounds"] < kept["questions"]
    strategy = bg.optimal_strategy()
    assert bg.play_chsh(strategy, guessed)["S"] < bg.play_chsh(strategy, kept)["S"]


def test_unconnected_players_are_reported():
    net = bg.add_node(bg.two_player_network(), "Carol")
    with pytest.raises(ValueError, match="not connected"):
        bg.run_network(net, "Alice", "Carol", sim_time_s=0.01)


def test_rates_are_supply_demand_and_rounds():
    run = bg.run_network(bg.two_player_network(50), "Alice", "Bob", sim_time_s=0.2, seed=1)
    assert run["rounds_hz"] <= min(run["questions_hz"], run["pairs_hz"])
    assert np.isclose(run["rounds_hz"], run["rounds"] / run["sim_time_s"])


def test_seed_none_is_random_and_returned_seed_repeats_a_run():
    net = bg.two_player_network(5)
    a = bg.run_network(net, "Alice", "Bob", sim_time_s=0.05)
    b = bg.run_network(net, "Alice", "Bob", sim_time_s=0.05)
    again = bg.run_network(net, "Alice", "Bob", sim_time_s=0.05, seed=a["seed"])
    assert a["seed"] != b["seed"]
    assert np.allclose(a["state"], again["state"]) and a["pair_age_ms"] == again["pair_age_ms"]


def test_sequence_settings_are_put_back():
    from sequence.entanglement_management.generation import EntanglementGenerationA
    from sequence.entanglement_management.purification import PurificationProtocol
    from sequence.kernel.quantum_manager import QuantumManager

    def settings():
        return (QuantumManager.get_active_formalism(), EntanglementGenerationA.get_global_type(),
                PurificationProtocol.get_formalism())

    before = settings()
    bg.to_sequence(bg.two_player_network())
    bg.run_network(bg.two_player_network(5), "Alice", "Bob", sim_time_s=0.01, seed=1)
    assert settings() == before
