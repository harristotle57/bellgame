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


@pytest.mark.parametrize("f1,f2", [(1.0, 1.0), (0.9, 0.9), (0.95, 0.8)])
def test_swap_of_werner_links(f1, f2):
    v1, v2 = (4 * f1 - 1) / 3, (4 * f2 - 1) / 3
    out = bg.swap(bg.bell_pair(f1), bg.bell_pair(f2))
    assert bg.fidelity(out) == pytest.approx((3 * v1 * v2 + 1) / 4)
    assert np.trace(out).real == pytest.approx(1.0)


def test_swap_of_non_werner_states_keeps_phi_plus():
    rho = bg.misalign(bg.bell_pair(), bob_deg=[0, 0, 20])  # phase error on one link
    out = bg.swap(bg.bell_pair(), rho)
    assert bg.fidelity(out) == pytest.approx(bg.fidelity(rho))


def test_misalignment_lowers_s_and_correction_restores():
    delta = [10, -15, 25]
    bad = bg.misalign(bg.bell_pair(), bob_deg=delta)
    assert bg.play_chsh(bg.optimal_strategy(), bad)["S"] < 2.7
    # undo: apply the inverse rotation as a correction
    u = bg.polarization_rotation(delta)
    fixed = bg.apply_local(bad, bob=u.conj().T)
    assert bg.play_chsh(bg.optimal_strategy(), fixed)["S"] == pytest.approx(2 * np.sqrt(2))


def test_dephasing():
    rho = bg.dephase(bg.bell_pair(), wait_ms=[1.0, 1.0], coherence_time_ms=10.0)
    lam = np.exp(-0.1) ** 2
    assert bg.fidelity(rho) == pytest.approx((1 + lam) / 2)


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
