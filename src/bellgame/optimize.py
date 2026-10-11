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

from .defaults import _as_number, _in_range
from .game import play_chsh
from .qubits import X, Y, Z, fidelity, misalign, polarization_angles
from .strategies import optimal_strategy

ANGLE_PERIOD_DEG = 180.0  # the correction rotations repeat (up to a global phase) every 180 degrees
METHODS = ("Nelder-Mead", "COBYQA")
STEP_DEG = 45.0  # starting step size for the optimizers


class _OutOfBudget(Exception):
    """Raised inside the objective when the time or evaluation budget is used up."""


class _NoData(Exception):
    """Raised inside the objective when S is NaN: the source has no rounds to measure."""


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
        # the first point is always evaluated, so even a tiny budget gives a real answer
        if points and (len(points) >= max_evaluations or _out_of_time(start, time_limit_s)):
            raise _OutOfBudget
        x = wrap_angles(x)  # simulate exactly the angles we record
        value = objective(x)
        points.append((value, x))
        if np.isnan(value):
            raise _NoData
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
    except (_OutOfBudget, _NoData):
        pass
    return points


def _out_of_time(start, time_limit_s):
    return time_limit_s is not None and time.perf_counter() - start > time_limit_s


def find_alignment(source, strategy=None, player="alice", method="Nelder-Mead", rounds=None,
                   seed=None, max_evaluations=300, time_limit_s=None, restarts=3, target_S=2.8):
    """Search for the correction angles that give the biggest S.

    Starts from a random guess (set ``seed`` to make it repeatable). In exact mode
    (``rounds=None``), if the best S found is below ``target_S``, it tries
    again from a new random guess, up to ``restarts`` searches in all. All the
    searches share one budget: ``max_evaluations`` calls and ``time_limit_s``
    seconds. The first call is always made, however small the budget.

    The default 2.8 suits nearly perfect pairs: no state with fidelity below
    about 0.99 can reach S = 2.8 (0.9925 if its noise is white), so for those
    every search runs and uses the whole budget. Lower it for noisier sources.

    If S is NaN, the source has no rounds to measure (e.g. a network run that got
    no pairs). The correction angles only change how Alice measures, not whether
    pairs arrive, so no other angles would do better: the search stops, and there
    is no answer (``correction`` and ``S`` are NaN, ``strategy`` is None). Run the
    network longer, or with another seed, to get rounds.

    Returns a dict with ``correction`` (3 angles in degrees), ``S`` (checked
    exactly at the end), ``strategy`` (ready to use), ``evaluations`` and ``history``.
    """
    max_evaluations = _as_number("max_evaluations", max_evaluations, int)
    if max_evaluations < 1:
        raise ValueError(f"max_evaluations must be at least 1, got {max_evaluations}")
    if time_limit_s is not None and not _in_range(_as_number("time_limit_s", time_limit_s, float), "(0, inf]"):
        raise ValueError(f"time_limit_s must be more than 0 (or None), got {time_limit_s}")
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
        if np.isnan(points[-1][0]):  # no rounds to measure
            return {"correction": [float("nan")] * 3, "S": float("nan"), "best_seen_S": float("nan"),
                    "strategy": None, "evaluations": evaluations, "history": history}
        for value, x in points:
            if value < best_value:
                best_value, best_x = value, x
        if (rounds is not None or -best_value >= target_S or evaluations >= max_evaluations
                or _out_of_time(start, time_limit_s)):
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
    """Random twists for both players: ``{"alice": [a, b, c], "bob": [a, b, c]}`` in degrees.

    A long fiber's twist is one net polarization change, built up from many small
    random ones, so every polarization change is equally likely. Each twist is
    drawn that way (a uniformly random unit quaternion) and written as the three
    angles ``bg.polarization_rotation`` takes. Drawing the three angles uniformly
    instead would favour some twists over others.
    """
    rng = np.random.default_rng(seed)
    return {player: [float(a) for a in polarization_angles(_random_unitary(rng))] for player in ("alice", "bob")}


def _random_unitary(rng):
    """A uniformly random 2x2 polarization change, from a uniformly random unit quaternion."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    return q[0] * np.eye(2) - 1j * (q[1] * X + q[2] * Y + q[3] * Z)


def misaligned_source(source, twist):
    """Apply a misalignment ``{"alice": [...], "bob": [...]}`` to a 4x4 state, a link dict or a network run."""
    if isinstance(source, dict) and "mean_photon_number" in source:
        out = dict(source)
        out["alice_misalignment_deg"] = list(twist["alice"])
        out["bob_misalignment_deg"] = list(twist["bob"])
        return out
    if isinstance(source, dict) and "state" in source:
        out = dict(source)
        out["state"] = misalign(source["state"], twist["alice"], twist["bob"])
        if "fidelity" in source:
            out["fidelity"] = fidelity(out["state"])
        if "history" in source:  # a network run: twist every round's pair too
            history = source["history"]
            out["history"] = {**history, "states": misalign(history["states"], twist["alice"], twist["bob"])}
        return out
    if isinstance(source, dict) or np.shape(source) != (4, 4):
        got = f"a dict with keys {sorted(source)}" if isinstance(source, dict) else f"shape {np.shape(source)}"
        raise ValueError("misaligned_source needs a 4x4 two-qubit state, a link from bg.link, or a run from "
                         f"bg.run_network; got {got}")
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
