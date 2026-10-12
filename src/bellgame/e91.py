"""The E91 quantum key distribution protocol (Ekert, 1991).

Alice and Bob each measure every photon pair at one of three polarizer angles,
picked at random. Afterwards they announce which angles they used (not the
results):

* when they used the **same** angle, their bits should be equal: those rounds
  become the secret **key**
* rounds with the right **different** angles are used to compute the CHSH
  number S. S > 2 shows the pairs are entangled; if S <= 2, throw the key away.
  How much of the key an eavesdropper could know shrinks as S approaches
  2 sqrt(2) = 2.83: S > 2 alone does not make the key secret.

Example:
    >>> import bellgame as bg
    >>> r = bg.run_e91(bg.bell_pair(), rounds=3000, seed=1)
    >>> r["qber"], r["S"] > 2.7
    (0.0, True)
"""

import numpy as np

from .defaults import check
from .game import chsh_value, correlations, count_rounds, outcome_probabilities, table_from_counts

# Polarizer angles in degrees. Index i in "alice" is Alice's basis i.
E91_SETTINGS = {
    "alice": [0.0, 22.5, 45.0],
    "bob": [22.5, 45.0, 67.5],
    # (Alice basis, Bob basis) pairs with the same angle: these make the key
    "key_pairs": [(1, 0), (2, 1)],
    # (Alice basis, Bob basis) -> CHSH question pair (x, y), for S = E00 + E01 + E10 - E11
    "chsh_pairs": {(2, 0): (0, 0), (2, 2): (0, 1), (0, 0): (1, 0), (0, 2): (1, 1)},
}

# The angles used by the original CHSH-game-introduction code. They have no
# matching pair of angles, so the "key" rounds disagree 14.6% of the time even
# with perfect equipment. Kept so you can compare.
E91_OLD_SETTINGS = {
    "alice": [0.0, 45.0, 90.0],
    "bob": [22.5, 67.5, 112.5],
    "key_pairs": [(0, 0)],
    "chsh_pairs": {(1, 1): (0, 0), (1, 0): (0, 1), (2, 1): (1, 0), (2, 0): (1, 1)},
}


def run_e91(source, rounds=10000, seed=None, settings=None, corrections=None):
    """Run E91 on a source of entangled pairs.

    * ``source``: anything ``bg.play`` accepts (a 4x4 state, a link, a network run)
    * ``rounds``: number of pairs measured; ``None`` gives the exact expected values
    * ``settings``: the angles and pairings (default ``bg.E91_SETTINGS``)
    * ``corrections``: ``{"alice_correction": [a, b, c], "bob_correction": [...]}`` (either
      or both), e.g. from ``bg.find_alignment``

    With ``rounds=N``, returns ``key_alice``, ``key_bob`` (lists of bits), ``key_length``,
    ``qber`` (fraction of key bits that disagree), ``S``, ``E``, ``secure``, ``rounds`` and
    ``chsh_counts``. With ``rounds=None`` there are no key bits: it returns the expected
    ``qber``, ``S``, ``E`` and ``secure``, and ``key_fraction``, the share of rounds that
    become key. ``secure`` is only the S > 2 test (see the module notes).

    On a network run, the rounds are drawn from the run's average delivered pair. That
    gives the same statistics as measuring each recorded pair in turn, since every round's
    angles are random; ``bg.play(..., rounds="run")`` replays the pairs one by one.

    A source with no data (a network run in which no round was played) gives no
    key: ``rounds`` is 0, the key is empty, ``qber`` and ``S`` are NaN, and it is
    not secure.
    """
    if settings is None:
        settings = E91_SETTINGS
    corrections = dict(corrections or {})
    unknown = set(corrections) - {"alice_correction", "bob_correction"}
    if unknown:
        raise ValueError(f"corrections can only have 'alice_correction' and 'bob_correction', got {sorted(unknown)}")
    probs = {
        (i, j): outcome_probabilities(source, alice_deg, bob_deg, **corrections)
        for i, alice_deg in enumerate(settings["alice"])
        for j, bob_deg in enumerate(settings["bob"])
    }
    n_a, n_b = len(settings["alice"]), len(settings["bob"])

    if rounds is None:
        return _exact(probs, settings)

    rounds = check("rounds", rounds)
    if any(np.isnan(p).any() for p in probs.values()):  # no pairs to measure
        rounds = 0
    rng = np.random.default_rng(seed)
    a_basis = rng.integers(0, n_a, size=rounds)
    b_basis = rng.integers(0, n_b, size=rounds)
    a_bits = np.zeros(rounds, dtype=int)
    b_bits = np.zeros(rounds, dtype=int)
    for (i, j), p in probs.items():
        mask = (a_basis == i) & (b_basis == j)
        n = int(mask.sum())
        if n == 0:
            continue
        flat = p.flatten() / p.sum()
        outcome = rng.choice(4, size=n, p=flat)  # 0=00, 1=01, 2=10, 3=11
        a_bits[mask] = outcome // 2
        b_bits[mask] = outcome % 2

    key_mask = np.zeros(rounds, dtype=bool)
    for i, j in settings["key_pairs"]:
        key_mask |= (a_basis == i) & (b_basis == j)
    key_alice = a_bits[key_mask].tolist()
    key_bob = b_bits[key_mask].tolist()
    errors = sum(ka != kb for ka, kb in zip(key_alice, key_bob))

    # the CHSH rounds, each with the question pair (x, y) its two angles stand for
    chsh = np.zeros(rounds, dtype=bool)
    x, y = np.zeros(rounds, dtype=int), np.zeros(rounds, dtype=int)
    for (i, j), (qx, qy) in settings["chsh_pairs"].items():
        mask = (a_basis == i) & (b_basis == j)
        chsh |= mask
        x[mask], y[mask] = qx, qy
    counts = count_rounds(x[chsh], y[chsh], a_bits[chsh], b_bits[chsh])
    table = table_from_counts(counts)
    s = chsh_value(table)
    return {
        "key_alice": key_alice,
        "key_bob": key_bob,
        "key_length": len(key_alice),
        "qber": errors / len(key_alice) if key_alice else float("nan"),
        "S": s,
        "E": np.round(correlations(table), 6).tolist(),
        "secure": s > 2,
        "rounds": rounds,
        "chsh_counts": counts,
    }


def _exact(probs, settings):
    """Expected QBER and S, with no sampling noise."""
    qbers = [p[0, 1] + p[1, 0] for (i, j), p in probs.items() if (i, j) in settings["key_pairs"]]
    table = np.zeros((2, 2, 2, 2))
    for (i, j), (x, y) in settings["chsh_pairs"].items():
        table[:, :, x, y] = probs[(i, j)]
    s = chsh_value(table)
    n_pairs = len(settings["alice"]) * len(settings["bob"])
    return {
        "qber": float(np.mean(qbers)),
        "S": s,
        "E": np.round(correlations(table), 6).tolist(),
        "secure": s > 2,
        "key_fraction": len(settings["key_pairs"]) / n_pairs,
        "rounds": None,
    }
