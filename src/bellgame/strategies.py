"""Strategies for the CHSH game.

A **quantum strategy** is a dict of measurement angles in degrees::

    {"alice": [angle for x=0, angle for x=1],
     "bob":   [angle for y=0, angle for y=1]}

It may also carry correction wave plates (three angles each, see
``bg.polarization_rotation``) that each player applies before measuring::

    {"alice": [0, 45], "bob": [22.5, -22.5],
     "alice_correction": [0, 0, 0], "bob_correction": [0, 0, 0]}

A **classical strategy** is just a pair of Python functions, one per player,
that turn the referee's bit into an answer bit::

    def alice(x):
        return 0

Example:
    >>> import bellgame as bg
    >>> bg.optimal_strategy()
    {'alice': [0, 45], 'bob': [22.5, -22.5]}
"""

import numpy as np

from .qubits import polarization_rotation, ry


def optimal_strategy():
    """The best quantum strategy for a Phi+ pair: wins cos^2(22.5 deg) = 85.4% of rounds.

    Example:
        >>> optimal_strategy()["bob"]
        [22.5, -22.5]
    """
    return {"alice": [0, 45], "bob": [22.5, -22.5]}


def always_zero(bit):
    """Classical player who ignores the question and answers 0."""
    return 0


def always_one(bit):
    """Classical player who ignores the question and answers 1."""
    return 1


def copy_bit(bit):
    """Classical player who answers with the bit they were given."""
    return bit


def flip_bit(bit):
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
    for player in ("alice", "bob"):
        if player not in strategy:
            raise ValueError(f"strategy is missing '{player}': it needs two angles in degrees")
        if len(strategy[player]) != 2:
            raise ValueError(f"strategy['{player}'] needs exactly two angles (one per input bit)")
        key = player + "_correction"
        if key in strategy and len(strategy[key]) != 3:
            raise ValueError(f"strategy['{key}'] needs exactly three angles in degrees")
    allowed = {"alice", "bob", "alice_correction", "bob_correction"}
    extra = set(strategy) - allowed
    if extra:
        raise ValueError(f"unknown strategy keys {sorted(extra)}; allowed keys are {sorted(allowed)}")


def measurement_unitary(strategy, player, bit):
    """The 2x2 unitary a player applies before measuring H (outcome 0) vs V (outcome 1).

    It is the correction wave plates (if any) followed by turning the chosen
    measurement angle onto H.

    Example:
        >>> U = measurement_unitary(optimal_strategy(), "alice", 1)
        >>> U.shape
        (2, 2)
    """
    angle = strategy[player][bit]
    correction = strategy.get(player + "_correction", (0, 0, 0))
    # ry(-2 angle) turns the polarization cos(angle) H + sin(angle) V onto H
    return ry(-2 * angle) @ polarization_rotation(correction)


def measurement_projectors(strategy, player, bit):
    """[P0, P1]: projectors for outcome 0 and outcome 1 on one qubit."""
    u = measurement_unitary(strategy, player, bit)
    projs = []
    for outcome in (0, 1):
        e = np.zeros(2)
        e[outcome] = 1
        projs.append(u.conj().T @ np.outer(e, e) @ u)
    return projs
