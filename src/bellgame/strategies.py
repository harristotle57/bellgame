"""Strategies for the CHSH game.

A **quantum strategy** is a dict of measurement angles in degrees::

    {"alice": [angle for x=0, angle for x=1],
     "bob":   [angle for y=0, angle for y=1]}

It may also carry correction wave plates (three angles each, see
``bg.polarization_rotation``) that each player applies before measuring::

    {"alice": [0, 45], "bob": [22.5, -22.5],
     "alice_correction": [0, 0, 0], "bob_correction": [0, 0, 0]}

A **player** is a Python function. It gets the referee's bit and its own past
rounds (a list of ``(question, answer)``), and returns an answer bit::

    def alice(bit, history):
        return 0

On a source of pairs, it also gets its half of this round's pair, which it can
measure once at an angle of its choice (see ``bg.Qubit``)::

    def alice(bit, history, qubit):
        return qubit.measure([0, 45][bit])

A quantum strategy dict is shorthand for two such players (``bg.quantum_players``).
The built-in classical players below take an optional ``qubit`` they ignore,
so they can play on any source.

Example:
    >>> import bellgame as bg
    >>> bg.optimal_strategy()
    {'alice': [0, 45], 'bob': [22.5, -22.5]}
"""

import numpy as np

from .qubits import check_angles, polarization_rotation, ry


def optimal_strategy():
    """The best quantum strategy for a Phi+ pair: wins cos^2(22.5 deg) = 85.4% of rounds.

    Example:
        >>> optimal_strategy()["bob"]
        [22.5, -22.5]
    """
    return {"alice": [0, 45], "bob": [22.5, -22.5]}


def always_zero(bit, history, qubit=None):
    """Classical player who ignores the question and answers 0."""
    return 0


def always_one(bit, history, qubit=None):
    """Classical player who ignores the question and answers 1."""
    return 1


def copy_bit(bit, history, qubit=None):
    """Classical player who answers with the bit they were given."""
    return bit


def flip_bit(bit, history, qubit=None):
    """Classical player who answers with the opposite of the bit they were given."""
    return 1 - bit


CLASSICAL_PLAYERS = {
    "always_zero": always_zero,
    "always_one": always_one,
    "copy_bit": copy_bit,
    "flip_bit": flip_bit,
}


def check_strategy(strategy):
    """Raise a friendly error if a quantum strategy dict is malformed.

    Example:
        >>> check_strategy({"alice": [0, 45], "bob": [22.5, -22.5]})
    """
    if not isinstance(strategy, dict):
        raise TypeError("a quantum strategy is a dict like {'alice': [0, 45], 'bob': [22.5, -22.5]}")
    allowed = {"alice", "bob", "alice_correction", "bob_correction"}
    extra = set(strategy) - allowed
    if extra:
        raise ValueError(f"unknown strategy keys {sorted(extra)}; allowed keys are {sorted(allowed)}")
    for player in ("alice", "bob"):
        if player not in strategy:
            raise ValueError(f"strategy is missing '{player}': it needs two angles in degrees")
        check_angles(strategy[player], 2, f"strategy['{player}'] (one angle per question)")
        key = player + "_correction"
        if key in strategy:
            check_angles(strategy[key], 3, f"strategy['{key}']")


def quantum_players(strategy):
    """The two players ``(alice, bob)`` a quantum strategy dict stands for.

    Each measures its qubit at the strategy's angle for its question (after its
    correction, if any) and answers with the outcome. Use them to mix a quantum
    player with one of your own.

    Example:
        >>> import bellgame as bg
        >>> alice, _ = quantum_players(bg.optimal_strategy())
        >>> mine = lambda bit, history, qubit: qubit.measure(22.5 if bit == 0 else -22.5)
        >>> round(bg.play((alice, mine), bg.bell_pair())["win_rate"], 4)
        0.8536
    """
    check_strategy(strategy)

    def player(name):
        angles, correction = strategy[name], strategy.get(name + "_correction")

        def measure(bit, history, qubit):
            return qubit.measure(angles[bit], correction)

        measure.__name__ = name
        return measure

    return player("alice"), player("bob")


def measurement_setting(angle_deg, correction_deg=None):
    """A measurement setting, checked, as ``(angle, correction angles or None)`` in floats.

    The result can be used as a dict key, so equal settings can share their projectors.

    Example:
        >>> measurement_setting(45, [0, 10, 0])
        (45.0, (0.0, 10.0, 0.0))
    """
    try:
        angle = float(angle_deg)
    except (TypeError, ValueError):
        raise TypeError(f"a measurement angle is one number in degrees, got {angle_deg!r}") from None
    if correction_deg is None:
        return angle, None
    return angle, check_angles(correction_deg, 3, "a correction")


def projectors(angle_deg, correction_deg=None):
    """[P0, P1]: projectors for outcome 0 and outcome 1 when measuring at ``angle_deg``.

    The correction wave plates (three angles, if any) act first, then the analyzer
    turns polarization ``angle_deg`` onto H (outcome 0) and the polarization at
    90 degrees from it onto V (outcome 1).

    Example:
        >>> p0, p1 = projectors(0)   # a plain H/V polarizer
        >>> np.real(p0).tolist()
        [[1.0, 0.0], [0.0, 0.0]]
    """
    angle, correction = measurement_setting(angle_deg, correction_deg)
    u = _unitary(angle, correction)
    back = u.conj().T  # undoes the analyzer
    seen_as_h = back @ [1, 0]  # the polarization the analyzer sends to H: outcome 0
    seen_as_v = back @ [0, 1]  # the polarization the analyzer sends to V: outcome 1
    return [np.outer(s, s.conj()) for s in (seen_as_h, seen_as_v)]


def measurement_projectors(strategy, player, bit):
    """[P0, P1] for ``player`` ("alice" or "bob") answering question ``bit`` with ``strategy``.

    Example:
        >>> p0, p1 = measurement_projectors(optimal_strategy(), "alice", 0)
        >>> np.real(p1).tolist()
        [[0.0, 0.0], [0.0, 1.0]]
    """
    return projectors(strategy[player][bit], strategy.get(player + "_correction"))


def _unitary(angle_deg, correction_deg=None):
    """Correction wave plates (if any), then turn the measurement angle onto H."""
    # ry(-2 angle) turns the polarization cos(angle) H + sin(angle) V onto H
    turn = ry(-2 * angle_deg)
    if correction_deg is None:
        return turn
    return turn @ polarization_rotation(correction_deg)
