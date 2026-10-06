import itertools

import numpy as np
import pytest

import bellgame as bg


def test_optimal_strategy_on_ideal_pair():
    r = bg.play_chsh(bg.optimal_strategy(), bg.bell_pair())
    assert r["win_rate"] == pytest.approx(np.cos(np.pi / 8) ** 2)
    assert r["S"] == pytest.approx(2 * np.sqrt(2))


def test_best_classical_is_three_quarters():
    players = list(bg.CLASSICAL_PLAYERS.values())
    best = max(bg.play_classical(a, b)["win_rate"] for a, b in itertools.product(players, players))
    assert best == pytest.approx(0.75)


@pytest.mark.parametrize("f", [0.25, 0.5, 0.8, 0.95, 1.0])
def test_werner_chsh(f):
    s = bg.play_chsh(bg.optimal_strategy(), bg.bell_pair(f))["S"]
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
    assert bg.play_chsh(bg.optimal_strategy(), bad)["S"] < 2.7
    # undo: apply the inverse rotation as a correction
    u = bg.polarization_rotation(delta)
    fixed = bg.apply_local(bad, bob=u.conj().T)
    assert bg.play_chsh(bg.optimal_strategy(), fixed)["S"] == pytest.approx(2 * np.sqrt(2))


def test_sampled_mode_is_seeded_and_close():
    a = bg.play_chsh(bg.optimal_strategy(), bg.bell_pair(), rounds=20000, seed=3)
    b = bg.play_chsh(bg.optimal_strategy(), bg.bell_pair(), rounds=20000, seed=3)
    assert a["counts"] == b["counts"]
    assert a["win_rate"] == pytest.approx(0.8536, abs=0.01)


def test_friendly_errors():
    with pytest.raises(ValueError, match="alice"):
        bg.play_chsh({"bob": [0, 1]}, bg.bell_pair())
    with pytest.raises(ValueError, match="fidelity"):
        bg.bell_pair(1.5)


# ---------------------------------------------------------------- classical players with coins and memory

def coin(bit):
    return int(np.random.default_rng().integers(2))


def test_random_player_has_no_exact_answer():
    with pytest.raises(ValueError, match="at random"):
        bg.play_classical(coin, bg.always_zero)


def test_history_player_has_no_exact_answer():
    def alice(x, history):
        return 0

    with pytest.raises(ValueError, match="past rounds"):
        bg.play_classical(alice, bg.always_zero)


def test_coins_and_memory_cannot_beat_a_fair_referee():
    def alice(x, history):  # answer what Bob's question was last round, guessed from my own past
        return history[-1][0] if history else 0

    for a, b in [(coin, coin), (alice, bg.copy_bit), (alice, coin)]:
        result = bg.play_classical(a, b, rounds=20000, seed=2)
        assert result["win_rate"] < 0.75 + 0.015  # 5 standard deviations


def test_history_is_each_players_own_past():
    seen = []

    def alice(x, history):
        seen.append(list(history))
        return x

    bg.play_classical(alice, bg.always_zero, rounds=3, seed=1)
    assert [len(h) for h in seen] == [0, 1, 2]
    assert all(a == q for q, a in seen[-1])


def test_predictable_referee_can_be_beaten():
    order = [(0, 0), (0, 1), (1, 0), (1, 1)]

    def referee(n):
        return order[n % 4]

    def alice(x, history):  # knows the referee's order, so knows Bob's question
        y = order[len(history) % 4][1]
        return x & y

    result = bg.play_classical(alice, bg.always_zero, rounds=400, questions=referee)
    assert result["win_rate"] == 1.0
