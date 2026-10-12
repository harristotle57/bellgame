from types import SimpleNamespace

import numpy as np
import pytest

import bellgame as bg
from bellgame.simulate import _Referee

NAMES = ["HubA", "A1", "A2", "A3", "HubB", "B1", "B2", "B3"]


def builder_net():
    return bg.two_star("HubA", ["A1", "A2", "A3"], "HubB", ["B1", "B2", "B3"], leaf_km=5, hub_km=20)


# ---------------------------------------------------------------- SeQUeNCe topology


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
    s_direct = bg.play(bg.optimal_strategy(), params)["S"]
    s_weights = bg.play(bg.optimal_strategy(), bg.bell_diagonal(weights))["S"]
    # SeQUeNCe stores only the Bell-diagonal part; the tiny rest (from double clicks) is dropped
    assert np.isclose(s_direct, s_weights, atol=1e-3)


def test_analytic_matches_fock_dark_count_limit():
    # With a faint source the Fock model's only noise is dark counts, like the analytic model.
    for km in (50, 200):
        net = bg.two_player_network(km)
        bg.set_link(
            net, "Alice", "Bob", dark_count_rate_hz=1e5, raw_fidelity=1.0, mean_photon_number=1e-4, heralded=True
        )
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
        s.append(bg.play(bg.optimal_strategy(), run)["S"])
    assert s[0] > s[1]


def test_missing_pairs_count_against_random_policy():
    net = bg.two_player_network(60)
    kept = bg.run_network(net, "Alice", "Bob", sim_time_s=0.2, seed=3)
    guessed = bg.run_network(net, "Alice", "Bob", sim_time_s=0.2, seed=3, no_pair="random")
    assert kept["rounds"] < kept["questions"]
    strategy = bg.optimal_strategy()
    assert bg.play(strategy, guessed)["S"] < bg.play(strategy, kept)["S"]


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
        return (
            QuantumManager.get_active_formalism(),
            EntanglementGenerationA.get_global_type(),
            PurificationProtocol.get_formalism(),
        )

    before = settings()
    bg.to_sequence(bg.two_player_network())
    bg.run_network(bg.two_player_network(5), "Alice", "Bob", sim_time_s=0.01, seed=1)
    assert settings() == before


# ---------------------------------------------------------------- memories, rates and history


def test_players_get_all_their_memories_on_a_direct_link():
    run = bg.run_network(bg.two_player_network(50), "Alice", "Bob", sim_time_s=0.05, seed=1)
    assert run["memories"] == 10  # the default memory_size; no node in the middle to share them


def test_middle_nodes_split_their_memories(two_star_run):
    assert two_star_run["memories"] == 5
    net = bg.set_node(builder_net(), "HubA", memory_size=1)
    with pytest.raises(ValueError, match="at least 2 memories"):
        bg.run_network(net, "A1", "B2", sim_time_s=0.01)


def test_rates_show_each_step_of_the_path(two_star_run):
    links, nodes = two_star_run["links"], two_star_run["nodes"]
    assert [q["nodes"] for q in links] == [["A1", "HubA"], ["HubA", "HubB"], ["HubB", "B2"]]
    assert all(q["pairs_hz"] >= two_star_run["pairs_hz"] > 0 for q in links)  # every end-to-end pair needs one per link
    swaps = {n["node"]: n["swaps_hz"] for n in nodes}
    assert swaps["A1"] == swaps["B2"] == 0
    assert swaps["HubA"] >= two_star_run["pairs_hz"] and swaps["HubB"] >= two_star_run["pairs_hz"]


def test_metrics_stop_recording_after_a_run():
    from sequence.utils import metrics

    bg.run_network(bg.two_player_network(5), "Alice", "Bob", sim_time_s=0.01, seed=1)
    recorded = len(metrics.storage.get_all())
    with bg.bell_diagonal_mode():  # a plain SeQUeNCe run afterwards adds nothing to the store
        topo = bg.to_sequence(bg.two_player_network(5), seed=1, stop_time_s=0.01)
        topo.get_timeline().init()
        topo.get_timeline().run()
    assert len(metrics.storage.get_all()) == recorded


def test_history_has_one_entry_per_question(two_star_run):
    h = two_star_run["history"]
    assert all(len(v) == two_star_run["questions"] for v in h.values())
    assert h["had_pair"].sum() == two_star_run["rounds"]
    assert np.all(np.isnan(h["age_ms"][~h["had_pair"]])) and not np.any(np.isnan(h["age_ms"][h["had_pair"]]))
    assert np.allclose(two_star_run["state"], h["states"][h["had_pair"]].mean(axis=0))


def test_replay_matches_the_average_state(two_star_run):
    strategy = bg.optimal_strategy()
    exact = bg.play(strategy, two_star_run)["win_rate"]
    result = bg.play(strategy, two_star_run, rounds="run", seed=1)
    assert result["rounds"] == two_star_run["rounds"]
    h = result["history"]
    assert np.all(h["a"][~h["played"]] == -1)
    assert result["wins"] == h["win"].sum()
    assert abs(result["win_rate"] - exact) < 4 * np.sqrt(exact * (1 - exact) / result["rounds"])


def test_replay_with_player_functions(two_star_run):
    # the same rounds, played by functions instead of a strategy dict, each seeing its history
    seen = []
    alice, bob = bg.quantum_players(bg.optimal_strategy())

    def counting_alice(x, history, qubit):
        seen.append(len(history))
        return alice(x, history, qubit)

    exact = bg.play(bg.optimal_strategy(), two_star_run)["win_rate"]
    result = bg.play((counting_alice, bob), two_star_run, rounds="run", seed=1)
    assert result["rounds"] == two_star_run["rounds"] == len(seen)
    assert seen == list(range(len(seen)))  # only played rounds count as history
    assert abs(result["win_rate"] - exact) < 4 * np.sqrt(exact * (1 - exact) / result["rounds"])


def test_replay_answers_missing_pairs_at_random():
    run = bg.run_network(bg.two_player_network(60), "Alice", "Bob", sim_time_s=0.2, seed=3, no_pair="random")
    assert run["rounds"] < run["questions"]
    assert bg.play(bg.optimal_strategy(), run, rounds="run", seed=1)["rounds"] == run["questions"]


def test_old_pairs_are_worse():
    net = bg.set_node(bg.two_player_network(20), coherence_time_ms=5)
    run = bg.run_network(net, "Alice", "Bob", questions_hz=500, sim_time_s=0.5, seed=1, pick="oldest")
    h = run["history"]
    used = h["had_pair"]
    assert np.corrcoef(h["age_ms"][used], bg.fidelity(h["states"][used]))[0, 1] < -0.5


def test_every_delivered_pair_is_counted():
    # On a direct link every pair SeQUeNCe makes reaches the players. Short-lived memories and
    # pick="oldest" make memories lose a stored pair and get a new one between two questions.
    net = bg.set_node(bg.two_player_network(20), coherence_time_ms=5)
    for pick in ("newest", "oldest"):
        run = bg.run_network(net, "Alice", "Bob", questions_hz=200, sim_time_s=0.5, seed=1, pick=pick)
        assert run["nodes"][0]["expired_hz"] > 0  # the situation this test is about did happen
        assert np.isclose(run["pairs_hz"], run["links"][0]["pairs_hz"]), pick


# ---------------------------------------------------------------- the referee


def _memory(name, partner):
    return SimpleNamespace(name=name, generation_time=0, entangled_memory={"memo_id": partner})


def _player(node, memories):
    return SimpleNamespace(
        node=SimpleNamespace(name=node), held={m.name: m for m in memories}, forget_lost=lambda partner: None
    )


def test_referee_only_pairs_halves_that_name_each_other():
    # Alice's a1 still names b1, but b1 has moved on to a pair with a2.
    a1, a2 = _memory("a1", "b1"), _memory("a2", "b1")
    b1 = _memory("b1", "a2")
    referee = _Referee(None, _player("Alice", [a1, a2]), _player("Bob", [b1]), period_ps=1, pick="newest")
    assert referee.shared_pairs() == [(a2, b1)]


# ---------------------------------------------------------------- run arguments


@pytest.mark.parametrize(
    "bad",
    [
        {"questions_hz": 0},
        {"questions_hz": -5},
        {"questions_hz": 2e12},
        {"questions_hz": float("nan")},
        {"questions_hz": "fast"},
        {"sim_time_s": 0},
        {"sim_time_s": -1},
        {"sim_time_s": float("inf")},
    ],
)
def test_bad_run_arguments_are_refused(bad):
    name = next(iter(bad))
    with pytest.raises(ValueError, match=name):
        bg.run_network(bg.two_player_network(5), "Alice", "Bob", seed=1, **{"sim_time_s": 0.01, **bad})


def test_fastest_question_rate_is_allowed():
    run = bg.run_network(bg.two_player_network(5), "Alice", "Bob", questions_hz=1e12, sim_time_s=1e-7, seed=1)
    assert run["questions"] > 0


# ---------------------------------------------------------------- routing and runs without rounds


def test_route_comes_from_sequence_forwarding_tables():
    # Two equally long ways from A to D: whichever SeQUeNCe picks, the run uses the same one.
    net = bg.from_edges([("A", "B", 5), ("B", "D", 5), ("A", "C", 5), ("C", "D", 5)])
    path = bg.expected_path(net, "A", "D")
    assert path in (["A", "B", "D"], ["A", "C", "D"])
    assert bg.run_network(net, "A", "D", sim_time_s=0.01, seed=1)["path"] == path


def test_run_with_no_rounds_has_no_state():
    # 100 km links and 0.1 ms: no pair is ready before the game ends
    run = bg.run_network(bg.two_player_network(100), "Alice", "Bob", questions_hz=1e5, sim_time_s=1e-4, seed=1)
    assert run["questions"] > 0 and run["rounds"] == 0
    assert np.all(np.isnan(run["state"])) and np.isnan(run["fidelity"])
    assert np.isnan(bg.play(bg.optimal_strategy(), run)["S"])
    assert np.isnan(bg.run_e91(run, rounds=None)["S"])
    # sampled play has no pairs to play: 0 rounds and NaN results, nothing raises
    sampled = bg.play(bg.optimal_strategy(), run, rounds=100, seed=1)
    assert sampled["rounds"] == 0 and np.isnan(sampled["win_rate"]) and np.isnan(sampled["S"])
    key = bg.run_e91(run, rounds=100, seed=1)
    assert key["rounds"] == key["key_length"] == 0 and np.isnan(key["qber"]) and not key["secure"]
    one_by_one = bg.play(bg.optimal_strategy(), run, rounds="run", seed=1)
    assert one_by_one["rounds"] == 0 and np.isnan(one_by_one["S"])


def test_random_answers_are_a_state_but_no_questions_are_not():
    net = bg.two_player_network(100)
    guessed = bg.run_network(net, "Alice", "Bob", questions_hz=1e5, sim_time_s=1e-4, seed=1, no_pair="random")
    assert guessed["rounds"] == 0 and np.isclose(guessed["fidelity"], 0.25)  # every answer was a guess
    unasked = bg.run_network(net, "Alice", "Bob", questions_hz=1, sim_time_s=0.5, seed=1, no_pair="random")
    assert unasked["questions"] == 0 and np.isnan(unasked["fidelity"])


def test_twisting_a_run_twists_every_round(two_star_run):
    strategy = bg.optimal_strategy()
    twisted = bg.misaligned_source(two_star_run, {"alice": [0, 0, 0], "bob": [0, 40, 0]})
    h = twisted["history"]
    assert np.allclose(twisted["state"], h["states"][h["had_pair"]].mean(axis=0))
    exact = bg.play(strategy, twisted)["win_rate"]
    assert exact < bg.play(strategy, two_star_run)["win_rate"] - 0.1
    played = bg.play(strategy, twisted, rounds="run", seed=1)
    assert abs(played["win_rate"] - exact) < 4 * np.sqrt(exact * (1 - exact) / played["rounds"])


def test_twisted_run_reports_its_own_fidelity(two_star_run):
    twisted = bg.misaligned_source(two_star_run, {"alice": [0, 0, 0], "bob": [0, 40, 0]})
    assert np.isclose(twisted["fidelity"], bg.fidelity(twisted["state"]))
    assert twisted["fidelity"] < two_star_run["fidelity"] - 0.2
    assert bg.play(bg.optimal_strategy(), twisted)["fidelity"] == twisted["fidelity"]
