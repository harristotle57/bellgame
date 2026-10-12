import itertools

import numpy as np
import pytest

import bellgame as bg


def test_optimal_strategy_on_ideal_pair():
    r = bg.play(bg.optimal_strategy(), bg.bell_pair())
    assert r["win_rate"] == pytest.approx(np.cos(np.pi / 8) ** 2)
    assert r["S"] == pytest.approx(2 * np.sqrt(2))


def test_best_classical_is_three_quarters():
    players = list(bg.CLASSICAL_PLAYERS.values())
    best = max(bg.play((a, b))["win_rate"] for a, b in itertools.product(players, players))
    assert best == pytest.approx(0.75)


@pytest.mark.parametrize("f", [0.25, 0.5, 0.8, 0.95, 1.0])
def test_werner_chsh(f):
    s = bg.play(bg.optimal_strategy(), bg.bell_pair(f))["S"]
    assert s == pytest.approx(2 * np.sqrt(2) * (4 * f - 1) / 3)


def test_win_rate_matches_s():
    t = bg.table_from_strategy(bg.optimal_strategy(), bg.bell_pair(0.9))
    assert bg.win_rate(t) == pytest.approx(0.5 + bg.chsh_value(t) / 8)


def test_bell_diagonal_order_matches_sequence():
    # SeQUeNCe's order is I, Z, X, Y: the error that turns Phi+ into each Bell state
    z_error = bg.apply_local(bg.bell_pair(), bob=np.diag([1, -1]))
    assert np.allclose(bg.bell_diagonal([0, 1, 0, 0]), z_error)
    assert np.allclose(bg.bell_diagonal([0.9, 0.1 / 3, 0.1 / 3, 0.1 / 3]), bg.bell_pair(0.9))


def test_misalignment_lowers_s_and_correction_restores():
    delta = [10, -15, 25]
    bad = bg.misalign(bg.bell_pair(), bob_deg=delta)
    assert bg.play(bg.optimal_strategy(), bad)["S"] < 2.7
    # undo: apply the inverse rotation as a correction
    u = bg.polarization_rotation(delta)
    fixed = bg.apply_local(bad, bob=u.conj().T)
    assert bg.play(bg.optimal_strategy(), fixed)["S"] == pytest.approx(2 * np.sqrt(2))


def test_sampled_mode_is_seeded_and_close():
    a = bg.play(bg.optimal_strategy(), bg.bell_pair(), rounds=20000, seed=3)
    b = bg.play(bg.optimal_strategy(), bg.bell_pair(), rounds=20000, seed=3)
    assert a["counts"] == b["counts"]
    assert a["win_rate"] == pytest.approx(0.8536, abs=0.01)


def test_friendly_errors():
    with pytest.raises(ValueError, match="alice"):
        bg.play({"bob": [0, 1]}, bg.bell_pair())
    with pytest.raises(ValueError, match="fidelity"):
        bg.bell_pair(1.5)


# ---------------------------------------------------------------- classical players with coins and memory


def coin(bit, history):
    return int(np.random.default_rng().integers(2))


def test_random_player_has_no_exact_answer():
    with pytest.raises(ValueError, match="doesn't always do the same"):
        bg.play((coin, bg.always_zero))


def test_history_player_has_no_exact_answer_if_it_uses_it():
    def last(x, history):
        return history[-1][1] if history else 0

    with pytest.raises(ValueError, match="doesn't always do the same"):
        bg.play((last, bg.always_zero))


def test_history_is_always_passed_but_may_be_ignored():
    def alice(x, history, bias=0.3):  # a default after history stays a default
        assert bias == 0.3
        return x

    assert bg.play((alice, bg.always_zero))["win_rate"] == 0.75
    assert bg.play((alice, bg.always_zero), rounds=10, seed=1)["rounds"] == 10


def test_players_without_history_get_a_clear_error():
    with pytest.raises(TypeError, match=r"alice\(bit, history\)"):
        bg.play((lambda bit: 0, bg.always_zero))
    with pytest.raises(TypeError, match=r"bob\(bit, history, qubit\)"):
        bg.play((bg.always_zero, lambda bit, history: 0), bg.bell_pair())


def test_coins_and_memory_cannot_beat_a_fair_referee():
    def alice(x, history):  # answer what Bob's question was last round, guessed from my own past
        return history[-1][0] if history else 0

    for a, b in [(coin, coin), (alice, bg.copy_bit), (alice, coin)]:
        result = bg.play((a, b), rounds=20000, seed=2)
        assert result["win_rate"] < 0.75 + 0.015  # 5 standard deviations


def test_history_is_each_players_own_past():
    seen = []

    def alice(x, history):
        seen.append(list(history))
        return x

    bg.play((alice, bg.always_zero), rounds=3, seed=1)
    assert [len(h) for h in seen] == [0, 1, 2]
    assert all(a == q for q, a in seen[-1])


def test_predictable_referee_can_be_beaten():
    order = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def referee(n):
        return order[n % 4]

    def alice(x, history):  # knows the referee's order, so knows Bob's question
        y = order[len(history) % 4][1]
        return x & y

    result = bg.play((alice, bg.always_zero), rounds=400, questions=referee)
    assert result["win_rate"] == 1.0


def test_s_needs_every_question_pair():
    # 3 rounds can't ask all four question pairs; S is NaN instead of a made-up number
    r = bg.play(bg.optimal_strategy(), bg.bell_pair(), rounds=3, seed=2)
    assert sum(sum(c.values()) == 0 for c in r["counts"].values()) >= 1
    assert np.isnan(r["S"]) and not np.isnan(r["win_rate"])


@pytest.mark.parametrize("bad", [0, -3, 2.5, "ten", float("nan")])
def test_bad_round_counts_are_refused(bad):
    with pytest.raises(ValueError, match="rounds"):
        bg.play(bg.optimal_strategy(), bg.bell_pair(), rounds=bad)
    with pytest.raises(ValueError, match="rounds"):
        bg.play((bg.strategies.always_zero, bg.strategies.always_zero), rounds=bad)
    with pytest.raises(ValueError, match="rounds"):
        bg.run_e91(bg.bell_pair(), rounds=bad)


def test_whole_number_floats_are_fine_as_round_counts():
    assert bg.play(bg.optimal_strategy(), bg.bell_pair(), rounds=100.0, seed=1)["rounds"] == 100


# ---------------------------------------------------------------- one interface for every player


def test_quantum_players_match_their_strategy_exactly():
    strategy = bg.with_correction(bg.optimal_strategy(), [5, 6, 7], "bob")
    twisted = bg.misalign(bg.bell_pair(0.9), bob_deg=[20, -35, 50])
    for source in (twisted, bg.link(distance_km=10)):
        players = bg.play(bg.quantum_players(strategy), source)
        direct = bg.play(strategy, source)
        assert np.allclose(players["table"], direct["table"])
    assert players["coincidence_prob"] == pytest.approx(direct["coincidence_prob"])


def test_quantum_players_round_by_round_match_the_exact_answer():
    exact = bg.play(bg.optimal_strategy(), bg.bell_pair(0.9))["win_rate"]
    played = bg.play(bg.quantum_players(bg.optimal_strategy()), bg.bell_pair(0.9), rounds=20000, seed=4)
    assert played["win_rate"] == pytest.approx(exact, abs=0.015)  # 5 standard deviations


def test_a_player_can_turn_its_outcome_into_any_answer():
    alice, bob = bg.quantum_players(bg.optimal_strategy())

    def contrary(y, history, qubit):  # flips its outcome, so the team wins when it should lose
        return 1 - bob(y, history, qubit)

    result = bg.play((alice, contrary), bg.bell_pair())
    assert result["win_rate"] == pytest.approx(1 - bg.TSIRELSON_WIN_RATE)


def test_classical_players_can_sit_on_a_source_and_ignore_it():
    def alice(x, history, qubit):
        return 0

    assert bg.play((alice, alice), bg.bell_pair())["win_rate"] == pytest.approx(0.75)
    assert bg.play((bg.copy_bit, bg.always_zero), bg.bell_pair())["win_rate"] == pytest.approx(0.75)
    played = bg.play((alice, alice), bg.bell_pair(), rounds=1000, seed=1)
    assert played["wins"] == 1000 - sum(played["counts"][1, 1].values())  # loses only on x = y = 1


def test_measured_halves_are_correlated_whoever_goes_first():
    # Phi+ measured at the same angle always agrees
    def same(bit, history, qubit):
        return qubit.measure(30)

    result = bg.play((same, same), bg.bell_pair(), rounds=200, seed=1)
    assert all(c["01"] == c["10"] == 0 for c in result["counts"].values())


def test_a_qubit_is_measured_once():
    def twice(bit, history, qubit):
        qubit.measure(0)
        return qubit.measure(0)

    with pytest.raises(ValueError, match="once"):
        bg.play((twice, twice), bg.bell_pair(), rounds=5, seed=1)


def test_a_quantum_player_can_use_its_history():
    # measures at the optimal angles, but only after its first round
    def warming_up(bit, history, qubit):
        return qubit.measure([0, 45][bit]) if history else 0

    _, bob = bg.quantum_players(bg.optimal_strategy())
    result = bg.play((warming_up, bob), bg.bell_pair(), rounds=5000, seed=2)
    assert result["win_rate"] == pytest.approx(bg.TSIRELSON_WIN_RATE, abs=0.025)
    with pytest.raises(ValueError, match="doesn't always do the same"):
        bg.play((warming_up, bob), bg.bell_pair())


def test_quantum_players_can_face_a_custom_referee():
    def referee(n):
        return 1, 1

    result = bg.play(bg.optimal_strategy(), bg.bell_pair(), rounds=1000, seed=1, questions=referee)
    assert result["counts"][0, 0] == {"00": 0, "01": 0, "10": 0, "11": 0}
    assert sum(result["counts"][1, 1].values()) == 1000


def test_adaptive_players_cannot_play_over_a_link():
    with pytest.raises(ValueError, match="photonic link"):
        bg.play((lambda x, history, qubit: int(np.random.default_rng().integers(2)), bg.copy_bit), bg.link(), rounds=10)


def test_bad_teams_and_sources_are_refused():
    with pytest.raises(TypeError, match="team"):
        bg.play(bg.always_zero)
    with pytest.raises(ValueError, match="needs a source"):
        bg.play(bg.optimal_strategy())
    with pytest.raises(ValueError, match="4x4"):
        bg.play(bg.optimal_strategy(), np.eye(2))
    with pytest.raises(ValueError, match="run"):
        bg.play(bg.optimal_strategy(), bg.bell_pair(), rounds="run")


@pytest.mark.parametrize("source", [bg.bell_pair(0.9), {"state": bg.bell_pair(0.8)}, bg.link(distance_km=5)])
def test_a_strategy_table_is_outcome_probabilities_at_its_angles(source):
    strategy = bg.with_correction(bg.optimal_strategy(), [5, -10, 20], "bob")
    table = bg.table_from_strategy(strategy, source)
    for x, y in itertools.product((0, 1), repeat=2):
        p = bg.outcome_probabilities(
            source, strategy["alice"][x], strategy["bob"][y], bob_correction=strategy["bob_correction"]
        )
        assert np.allclose(table[:, :, x, y], p)
        assert p.sum() == pytest.approx(1)


def test_outcome_probabilities_need_a_source_and_good_angles():
    with pytest.raises(ValueError, match="source of pairs"):
        bg.outcome_probabilities(None, 0, 0)
    with pytest.raises(ValueError, match="angle_deg"):
        bg.outcome_probabilities(bg.bell_pair(), [0, 1], 0)
    assert np.isnan(bg.outcome_probabilities(np.full((4, 4), np.nan), 0, 0)).all()


def test_count_rounds():
    counts = bg.count_rounds([0, 0, 1], [1, 1, 1], [0, 1, 1], [0, 1, 0])
    assert counts[(0, 1)] == {"00": 1, "01": 0, "10": 0, "11": 1} and counts[(1, 1)]["10"] == 1
    assert sum(counts[(0, 0)].values()) == 0
