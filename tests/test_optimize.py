import numpy as np
import pytest
from scipy.optimize import minimize

import bellgame as bg


@pytest.mark.parametrize("method", ["Nelder-Mead", "COBYQA"])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_exact_mode_recovers_tsirelson(method, k):
    twisted = bg.misaligned_source(bg.bell_pair(), bg.random_misalignment(k))
    assert bg.play_chsh(bg.optimal_strategy(), twisted)["S"] < 2.8
    found = bg.find_alignment(twisted, method=method, seed=k)
    assert found["S"] > 2.8


def test_students_can_call_scipy_directly():
    twisted = bg.misaligned_source(bg.bell_pair(), bg.random_misalignment(5))
    res = minimize(bg.chsh_objective(twisted), x0=[0, 0, 0], method="Nelder-Mead")
    assert -res.fun > 2.8


def test_recorded_angles_are_the_simulated_angles():
    """The old COBYQA code recorded wrapped angles but simulated unwrapped ones."""
    twisted = bg.misaligned_source(bg.bell_pair(0.9), bg.random_misalignment(4))
    found = bg.find_alignment(twisted, method="COBYQA", rounds=2000, seed=3, max_evaluations=60)
    assert all(-90 < a <= 90 for a in found["correction"])
    assert found["S"] == pytest.approx(bg.play_chsh(found["strategy"], twisted)["S"])


def test_wrap_angles():
    assert np.allclose(bg.wrap_angles([270, -90, 90, 0.5]), [90, 90, 90, 0.5])


def test_history_is_recorded_with_times():
    history = []
    f = bg.chsh_objective(bg.bell_pair(), history=history)
    f([0, 0, 0])
    f([10, 0, 0])
    assert len(history) == 2 and history[1][0] >= history[0][0]


def test_compare_optimizers_shape_and_quality():
    out = bg.compare_optimizers(bg.bell_pair(), runs=2, rounds=2500, max_evaluations=120, seed=1)
    for method in ("Nelder-Mead", "COBYQA"):
        assert len(out[method]["histories"]) == 2
        assert min(out[method]["validated_S"]) > 2.6


def test_link_source_alignment():
    twisted = bg.misaligned_source(bg.link(), bg.random_misalignment(2))
    found = bg.find_alignment(twisted, seed=0, max_evaluations=150)
    best_possible = bg.play_chsh(bg.optimal_strategy(), bg.link())["S"]
    assert found["S"] == pytest.approx(best_possible, abs=0.01)
