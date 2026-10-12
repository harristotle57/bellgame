import numpy as np
import pytest
from scipy.optimize import minimize

import bellgame as bg


@pytest.mark.parametrize("method", ["Nelder-Mead", "COBYQA"])
@pytest.mark.parametrize("k", [0, 1, 2])
def test_exact_mode_recovers_tsirelson(method, k):
    twisted = bg.misaligned_source(bg.bell_pair(), bg.random_misalignment(k))
    assert bg.play(bg.optimal_strategy(), twisted)["S"] < 2.8
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
    assert found["S"] == pytest.approx(bg.play(found["strategy"], twisted)["S"])


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
    best_possible = bg.play(bg.optimal_strategy(), bg.link())["S"]
    assert found["S"] == pytest.approx(best_possible, abs=0.01)


@pytest.mark.parametrize("budget", [{"max_evaluations": 1}, {"time_limit_s": 1e-9}])
def test_tiny_budget_still_evaluates_one_point(budget):
    found = bg.find_alignment(bg.bell_pair(), seed=1, **budget)
    assert found["evaluations"] == 1 and found["best_seen_S"] > -np.inf
    assert found["S"] == pytest.approx(bg.play(found["strategy"], bg.bell_pair())["S"])


@pytest.mark.parametrize("bad", [{"max_evaluations": 0}, {"max_evaluations": 2.5}, {"time_limit_s": 0}])
def test_bad_budgets_are_refused(bad):
    with pytest.raises(ValueError, match=next(iter(bad))):
        bg.find_alignment(bg.bell_pair(), **bad)


def test_target_s_decides_restarts():
    noisy = bg.misaligned_source(bg.bell_pair(0.9), bg.random_misalignment(4))  # best S is about 2.26
    strict = bg.find_alignment(noisy, seed=1, restarts=3, max_evaluations=1000)
    relaxed = bg.find_alignment(noisy, seed=1, restarts=3, max_evaluations=1000, target_S=2.2)
    assert relaxed["evaluations"] < strict["evaluations"]
    assert relaxed["S"] > 2.2


@pytest.mark.parametrize("bad", [{"nodes": {}}, np.eye(2), [1, 2, 3]])
def test_misaligning_an_unknown_source_says_what_is_accepted(bad):
    with pytest.raises(ValueError, match="needs a 4x4 two-qubit state"):
        bg.misaligned_source(bad, bg.random_misalignment(1))


def test_a_source_with_no_data_gives_no_answer():
    found = bg.find_alignment(np.full((4, 4), np.nan), seed=1, max_evaluations=20)
    assert found["strategy"] is None and found["evaluations"] == 1  # stops at once: no angles can help
    assert np.isnan(found["S"]) and np.all(np.isnan(found["correction"]))


def test_random_misalignment_is_a_uniformly_random_twist():
    # A uniformly random twist sends H to every point of the polarization sphere equally often,
    # so each of the sphere's coordinates is spread evenly over [-1, 1].
    from bellgame.qubits import X, Y, Z

    points = []
    for k in range(2000):
        u = bg.polarization_rotation(bg.random_misalignment(k)["bob"])
        psi = u @ np.array([1, 0])
        points.append([np.real(psi.conj() @ p @ psi) for p in (X, Y, Z)])
    shares = np.array([np.histogram(axis, np.linspace(-1, 1, 5))[0] / len(points) for axis in np.array(points).T])
    assert np.all(np.abs(shares - 0.25) < 0.04), shares


def test_polarization_angles_undo_polarization_rotation():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        u = q[0] * np.eye(2) - 1j * (q[1] * bg.qubits.X + q[2] * bg.qubits.Y + q[3] * bg.qubits.Z)
        v = bg.polarization_rotation(bg.polarization_angles(u))
        assert np.isclose(abs(np.trace(u.conj().T @ v)), 2)  # equal up to a global phase
