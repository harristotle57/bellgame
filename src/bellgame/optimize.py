"""Undo fiber misalignment by searching for the right correction wave plates.

A twisted fiber rotates the photons' polarization (see ``bg.misalign``). One
player can undo it with three correction angles, if they can find them. The
only feedback is the CHSH number S, so this is a black-box optimization.

``bg.chsh_objective`` turns that into a function of three angles that you can
hand straight to ``scipy.optimize.minimize``::

    from scipy.optimize import minimize
    f = bg.chsh_objective(source)
    result = minimize(f, x0=[0, 0, 0], method="Nelder-Mead")

Example:
    >>> import bellgame as bg
    >>> twisted = bg.misalign(bg.bell_pair(), bob_deg=[20, -35, 50])
    >>> found = bg.find_alignment(twisted, seed=1)
    >>> found["S"] > 2.8
    True
"""

import time

import numpy as np
from scipy.optimize import minimize

from .game import play_chsh
from .qubits import misalign
from .strategies import optimal_strategy

ANGLE_PERIOD_DEG = 180.0  # the correction rotations repeat (up to a global phase) every 180 degrees
METHODS = ("Nelder-Mead", "COBYQA")
STEP_DEG = 45.0  # starting step size for the optimizers


class _OutOfBudget(Exception):
    """Raised inside the objective when the time or evaluation budget is used up."""


def wrap_angles(angles_deg):
    """Map angles to the range (-90, 90] degrees (one full period of a correction angle).

    Example:
        >>> wrap_angles([100, -90, 45]).tolist()
        [-80.0, 90.0, 45.0]
    """
    a = np.asarray(angles_deg, dtype=float)
    half = ANGLE_PERIOD_DEG / 2
    wrapped = (a + half) % ANGLE_PERIOD_DEG - half
    return np.where(np.isclose(wrapped, -half), half, wrapped)


def with_correction(strategy, angles_deg, player="alice"):
    """A copy of ``strategy`` with correction angles for one player.

    Example:
        >>> with_correction(optimal_strategy(), [1, 2, 3])["alice_correction"]
        [1.0, 2.0, 3.0]
    """
    if player not in ("alice", "bob"):
        raise ValueError("player must be 'alice' or 'bob'")
    out = dict(strategy)
    out[player + "_correction"] = [float(a) for a in wrap_angles(angles_deg)]
    return out


def chsh_objective(source, strategy=None, player="alice", rounds=None, seed=None, history=None):
    """A function f(angles_deg) = -S, ready for ``scipy.optimize.minimize``.

    * ``angles_deg``: the three correction angles ``player`` applies.
    * ``rounds=None`` uses the exact S. ``rounds=N`` simulates N rounds per call,
      so S is noisy, like in a real experiment.
    * Pass a list as ``history`` and every call appends ``(seconds, S)`` to it.

    It returns *minus* S because scipy minimizes.

    Example:
        >>> import bellgame as bg
        >>> f = chsh_objective(bg.bell_pair())
        >>> round(-f([0, 0, 0]), 3)
        2.828
    """
    if strategy is None:
        strategy = optimal_strategy()
    rng = np.random.default_rng(seed)
    start = time.perf_counter()

    def objective(angles_deg):
        trial = with_correction(strategy, angles_deg, player)
        round_seed = None if rounds is None else int(rng.integers(2**32))
        s = play_chsh(trial, source, rounds=rounds, seed=round_seed)["S"]
        if history is not None:
            history.append((time.perf_counter() - start, s))
        return -s

    return objective


def _run_method(objective, method, x0, max_evaluations, time_limit_s, start):
    """Run one scipy method, stopping when the budget runs out. Returns the evaluated points."""
    points = []

    def budgeted(x):
        if len(points) >= max_evaluations:
            raise _OutOfBudget
        if time_limit_s is not None and time.perf_counter() - start > time_limit_s:
            raise _OutOfBudget
        x = wrap_angles(x)  # simulate exactly the angles we record
        value = objective(x)
        points.append((value, x))
        return value

    x0 = np.asarray(x0, dtype=float)
    try:
        if method == "Nelder-Mead":
            simplex = [x0] + [x0 + STEP_DEG * np.eye(3)[i] for i in range(3)]
            minimize(budgeted, x0, method="Nelder-Mead",
                     options={"initial_simplex": np.array(simplex), "maxfev": max_evaluations,
                              "xatol": 1e-3, "fatol": 1e-6})
        elif method == "COBYQA":
            minimize(budgeted, x0, method="COBYQA",
                     options={"initial_tr_radius": STEP_DEG, "maxfev": max_evaluations})
        else:
            raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    except _OutOfBudget:
        pass
    return points


def find_alignment(source, strategy=None, player="alice", method="Nelder-Mead", rounds=None,
                   seed=None, max_evaluations=300, time_limit_s=None, restarts=3):
    """Search for the correction angles that give the biggest S.

    Starts from a random guess (set ``seed`` to make it repeatable) and, if it
    gets stuck below 2.8 in exact mode, tries again from new random guesses.

    Returns a dict with ``correction`` (3 angles in degrees), ``S`` (checked
    exactly at the end), ``strategy`` (ready to use), ``evaluations`` and ``history``.
    """
    if strategy is None:
        strategy = optimal_strategy()
    rng = np.random.default_rng(seed)
    history = []
    objective = chsh_objective(source, strategy, player, rounds=rounds,
                               seed=int(rng.integers(2**32)), history=history)
    start = time.perf_counter()
    best_value, best_x, evaluations = np.inf, np.zeros(3), 0
    for _ in range(max(1, restarts)):
        x0 = rng.uniform(-90, 90, size=3)
        points = _run_method(objective, method, x0, max_evaluations - evaluations, time_limit_s, start)
        evaluations += len(points)
        for value, x in points:
            if value < best_value:
                best_value, best_x = value, x
        if rounds is not None or -best_value > 2.8 or evaluations >= max_evaluations:
            break
    best_strategy = with_correction(strategy, best_x, player)
    return {
        "correction": [float(a) for a in wrap_angles(best_x)],
        "S": play_chsh(best_strategy, source)["S"],
        "best_seen_S": float(-best_value),
        "strategy": best_strategy,
        "evaluations": evaluations,
        "history": history,
    }


def random_misalignment(seed=None):
    """Random twists for both players: ``{"alice": [a, b, c], "bob": [a, b, c]}`` in degrees."""
    rng = np.random.default_rng(seed)
    return {
        "alice": [float(a) for a in rng.uniform(-90, 90, size=3)],
        "bob": [float(a) for a in rng.uniform(-90, 90, size=3)],
    }


def misaligned_source(source, twist):
    """Apply a misalignment ``{"alice": [...], "bob": [...]}`` to a qubit state or a link dict."""
    if isinstance(source, dict) and "mean_photon_number" in source:
        out = dict(source)
        out["alice_misalignment_deg"] = list(twist["alice"])
        out["bob_misalignment_deg"] = list(twist["bob"])
        return out
    if isinstance(source, dict) and "state" in source:
        out = dict(source)
        out["state"] = misalign(source["state"], twist["alice"], twist["bob"])
        return out
    return misalign(source, twist["alice"], twist["bob"])


def compare_optimizers(source, methods=METHODS, runs=5, rounds=2500, seed=0,
                       max_evaluations=200, time_limit_s=None):
    """Race optimizers at undoing random misalignment, like the original study.

    Each run draws a new random twist for both players (the same twists for
    every method), then each method searches for Alice's correction. In sampled
    mode (``rounds=N``) S is noisy; at the end every answer is checked with the
    exact S (``validated_S``).

    Returns ``{method: {"histories": [...], "validated_S": [...], "corrections": [...]}}``.
    Plot the histories with ``bg.plot_convergence``.
    """
    rng = np.random.default_rng(seed)
    out = {m: {"histories": [], "validated_S": [], "corrections": []} for m in methods}
    for _ in range(runs):
        twisted = misaligned_source(source, random_misalignment(int(rng.integers(2**32))))
        run_seed = int(rng.integers(2**32))
        for method in methods:
            found = find_alignment(twisted, method=method, rounds=rounds, seed=run_seed,
                                   max_evaluations=max_evaluations, time_limit_s=time_limit_s,
                                   restarts=1)
            out[method]["histories"].append(found["history"])
            out[method]["validated_S"].append(found["S"])
            out[method]["corrections"].append(found["correction"])
    return out
