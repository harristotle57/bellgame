"""The E91 quantum key distribution protocol (Ekert, 1991).

Alice and Bob each measure every photon pair at one of three polarizer angles,
picked at random. Afterwards they announce which angles they used (not the
results):

* when they used the **same** angle, their bits should be equal: those rounds
  become the secret **key**
* rounds with the right **different** angles are used to compute the CHSH
  number S. If S > 2, the pairs really were entangled, so no eavesdropper can
  know the key. If S <= 2, throw the key away.

Example:
    >>> import bellgame as bg
    >>> r = bg.run_e91(bg.bell_pair(), rounds=3000, seed=1)
    >>> r["qber"], r["S"] > 2.7
    (0.0, True)
"""

import numpy as np

from .game import chsh_value, correlations, table_from_counts, table_from_strategy

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


def basis_pair_table(source, alice_deg, bob_deg, corrections=None):
    """p[a, b]: outcome probabilities when Alice measures at ``alice_deg`` and Bob at ``bob_deg``."""
    strategy = {"alice": [alice_deg, alice_deg], "bob": [bob_deg, bob_deg]}
    if corrections:
        strategy.update(corrections)
    return table_from_strategy(strategy, source)[:, :, 0, 0]


def run_e91(source, rounds=10000, seed=None, settings=None, corrections=None):
    """Run E91 on a source of entangled pairs.

    * ``source``: anything ``bg.play_chsh`` accepts (a 4x4 state, a link, a path result)
    * ``rounds``: number of pairs measured; ``None`` gives the exact expected values
    * ``settings``: the angles and pairings (default ``bg.E91_SETTINGS``)
    * ``corrections``: e.g. ``{"alice_correction": [a, b, c]}`` from ``bg.find_alignment``

    Returns a dict with ``key_alice``, ``key_bob`` (lists of bits), ``key_length``,
    ``qber`` (fraction of key bits that disagree), ``S``, ``E``, and ``secure`` (S > 2).
    """
    if settings is None:
        settings = E91_SETTINGS
    n_a, n_b = len(settings["alice"]), len(settings["bob"])
    probs = {}
    for i in range(n_a):
        for j in range(n_b):
            probs[(i, j)] = basis_pair_table(source, settings["alice"][i], settings["bob"][j], corrections)

    if rounds is None:
        return _exact(probs, settings)

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

    counts = {(x, y): {"00": 0, "01": 0, "10": 0, "11": 0} for x in (0, 1) for y in (0, 1)}
    for (i, j), (x, y) in settings["chsh_pairs"].items():
        mask = (a_basis == i) & (b_basis == j)
        for a, b in zip(a_bits[mask], b_bits[mask]):
            counts[(x, y)][f"{a}{b}"] += 1
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
