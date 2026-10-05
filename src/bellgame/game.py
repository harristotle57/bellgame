"""The CHSH game.

The referee sends Alice a random bit x and Bob a random bit y. Alice answers a,
Bob answers b. They win when ``a XOR b == x AND y``.

Everything in bellgame boils down to a **probability table**
``table[a, b, x, y] = p(a, b | x, y)``, a numpy array of shape (2, 2, 2, 2).
Strategies plus sources make tables; the game reads tables.

Example:
    >>> import bellgame as bg
    >>> result = bg.play_chsh(bg.optimal_strategy(), bg.bell_pair())
    >>> round(result["win_rate"], 4)
    0.8536
"""

import numpy as np

from .strategies import check_strategy, measurement_projectors

CLASSICAL_WIN_RATE = 0.75
TSIRELSON_WIN_RATE = (2 + np.sqrt(2)) / 4  # = cos^2(pi/8) ~ 0.854
CLASSICAL_S = 2.0
TSIRELSON_S = 2 * np.sqrt(2)


def referee_wins(x, y, a, b):
    """True when the answers a, b win for questions x, y.

    Example:
        >>> referee_wins(1, 1, 0, 1)
        True
    """
    return (a ^ b) == (x & y)


def table_from_classical(alice, bob):
    """Probability table for two classical player functions.

    Example:
        >>> from bellgame.strategies import always_zero
        >>> win_rate(table_from_classical(always_zero, always_zero))
        0.75
    """
    table = np.zeros((2, 2, 2, 2))
    for x in (0, 1):
        for y in (0, 1):
            a, b = alice(x), bob(y)
            if a not in (0, 1) or b not in (0, 1):
                raise ValueError(f"players must answer 0 or 1, got a={a}, b={b}")
            table[a, b, x, y] = 1.0
    return table


def table_from_state(strategy, rho):
    """Probability table for a quantum strategy measured on a 4x4 two-qubit state.

    Example:
        >>> from bellgame.qubits import bell_pair
        >>> from bellgame.strategies import optimal_strategy
        >>> table_from_state(optimal_strategy(), bell_pair()).shape
        (2, 2, 2, 2)
    """
    check_strategy(strategy)
    rho = np.asarray(rho)
    if rho.shape != (4, 4):
        raise ValueError(f"a two-qubit state must be a 4x4 matrix, got shape {rho.shape}")
    table = np.zeros((2, 2, 2, 2))
    for x in (0, 1):
        pa = measurement_projectors(strategy, "alice", x)
        for y in (0, 1):
            pb = measurement_projectors(strategy, "bob", y)
            for a in (0, 1):
                for b in (0, 1):
                    table[a, b, x, y] = np.real(np.trace(np.kron(pa[a], pb[b]) @ rho))
    return np.clip(table, 0, 1)


def table_from_strategy(strategy, source):
    """Probability table for a quantum strategy and any source of entanglement.

    ``source`` can be:

    * a 4x4 two-qubit density matrix (e.g. ``bg.bell_pair(0.9)``)
    * the dict returned by ``bg.end_to_end`` (uses its ``"state"``)
    * a photonic link dict from ``bg.link(...)`` (Fock model, see ``bg.link_table``)

    Example:
        >>> from bellgame.qubits import bell_pair
        >>> from bellgame.strategies import optimal_strategy
        >>> t = table_from_strategy(optimal_strategy(), bell_pair())
    """
    if isinstance(source, dict) and "mean_photon_number" in source:
        from .link import link_table

        return link_table(source, strategy)
    if isinstance(source, dict) and "state" in source:
        return table_from_state(strategy, source["state"])
    if isinstance(source, dict):
        raise ValueError("source dict must be a link (bg.link) or a path result (bg.end_to_end)")
    return table_from_state(strategy, source)


def win_rate(table):
    """Chance of winning when x and y are uniformly random.

    Example:
        >>> from bellgame.strategies import always_zero, copy_bit
        >>> win_rate(table_from_classical(copy_bit, always_zero))
        0.75
    """
    total = 0.0
    for x in (0, 1):
        for y in (0, 1):
            for a in (0, 1):
                for b in (0, 1):
                    if referee_wins(x, y, a, b):
                        total += table[a, b, x, y] / 4
    return float(total)


def correlations(table):
    """2x2 array E[x, y] = p(a == b) - p(a != b) for each pair of questions.

    Example:
        >>> from bellgame.qubits import bell_pair
        >>> from bellgame.strategies import optimal_strategy
        >>> np.round(correlations(table_from_state(optimal_strategy(), bell_pair())), 3)
        array([[ 0.707,  0.707],
               [ 0.707, -0.707]])
    """
    signs = np.array([[1, -1], [-1, 1]])  # +1 when a == b
    return np.einsum("ab,abxy->xy", signs, table)


def chsh_value(table):
    """The CHSH number S = E00 + E01 + E10 - E11, with this one fixed sign pattern.

    Classical players can't beat S = 2. Quantum players reach 2*sqrt(2) ~ 2.83.
    (Win rate = 1/2 + S/8.)

    Example:
        >>> from bellgame.qubits import bell_pair
        >>> from bellgame.strategies import optimal_strategy
        >>> round(chsh_value(table_from_state(optimal_strategy(), bell_pair())), 4)
        2.8284
    """
    e = correlations(table)
    return float(e[0, 0] + e[0, 1] + e[1, 0] - e[1, 1])


def sample_counts(table, rounds, seed=None):
    """Play ``rounds`` random rounds from a table, return counts per question pair.

    The result looks like ``{(x, y): {"00": n, "01": n, "10": n, "11": n}}``,
    where the string is the answers ``a`` then ``b``.

    Example:
        >>> from bellgame.strategies import always_zero
        >>> counts = sample_counts(table_from_classical(always_zero, always_zero), 100, seed=1)
        >>> sum(sum(c.values()) for c in counts.values())
        100
    """
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")
    rng = np.random.default_rng(seed)
    counts = {(x, y): {"00": 0, "01": 0, "10": 0, "11": 0} for x in (0, 1) for y in (0, 1)}
    xs = rng.integers(0, 2, size=rounds)
    ys = rng.integers(0, 2, size=rounds)
    for x in (0, 1):
        for y in (0, 1):
            n = int(np.sum((xs == x) & (ys == y)))
            if n == 0:
                continue
            p = table[:, :, x, y].flatten()
            draws = rng.multinomial(n, p / p.sum())
            for k, ab in enumerate(["00", "01", "10", "11"]):
                counts[(x, y)][ab] = int(draws[k])
    return counts


def table_from_counts(counts):
    """Turn sampled counts back into an (estimated) probability table."""
    table = np.zeros((2, 2, 2, 2))
    for (x, y), c in counts.items():
        n = sum(c.values())
        if n == 0:
            table[:, :, x, y] = 0.25
            continue
        for ab, k in c.items():
            table[int(ab[0]), int(ab[1]), x, y] = k / n
    return table


def summarize(table, rounds=None, seed=None):
    """Result dict for a table: exact (``rounds=None``) or estimated from sampled rounds.

    Example:
        >>> from bellgame.strategies import always_zero
        >>> summarize(table_from_classical(always_zero, always_zero))["win_rate"]
        0.75
    """
    result = {}
    if rounds is None:
        used = table
    else:
        counts = sample_counts(table, rounds, seed=seed)
        used = table_from_counts(counts)
        wins = 0
        for (x, y), c in counts.items():
            for ab, k in c.items():
                if referee_wins(x, y, int(ab[0]), int(ab[1])):
                    wins += k
        result["rounds"] = rounds
        result["wins"] = wins
        result["counts"] = counts
    e = correlations(used)
    if rounds is None:
        result["win_rate"] = win_rate(used)
    else:
        result["win_rate"] = result["wins"] / rounds
    result["S"] = chsh_value(used)
    result["E"] = np.round(e, 6).tolist()
    result["table"] = table
    return result


def play_classical(alice, bob, rounds=None, seed=None):
    """Play the CHSH game with two classical player functions.

    With ``rounds=None`` you get the exact win rate; with a number you get a
    simulated experiment with that many rounds.

    Example:
        >>> from bellgame.strategies import always_zero
        >>> play_classical(always_zero, always_zero)["win_rate"]
        0.75
    """
    return summarize(table_from_classical(alice, bob), rounds=rounds, seed=seed)


def play_chsh(strategy, source, rounds=None, seed=None):
    """Play the CHSH game with a quantum strategy on some source of entanglement.

    Returns a dict with ``win_rate``, ``S``, the correlations ``E``, and the
    probability ``table``. With ``rounds=N`` it also has ``counts`` and ``wins``.
    When the source reports a herald or pair rate, that is passed through too.

    Example:
        >>> import bellgame as bg
        >>> r = bg.play_chsh(bg.optimal_strategy(), bg.bell_pair(0.9), rounds=1000, seed=7)
        >>> r["rounds"]
        1000
    """
    if isinstance(source, dict) and "mean_photon_number" in source:
        from .link import link_outcomes

        table, coincidence_prob = link_outcomes(source, strategy)
        result = summarize(table, rounds=rounds, seed=seed)
        result["coincidence_prob"] = coincidence_prob
        return result
    table = table_from_strategy(strategy, source)
    result = summarize(table, rounds=rounds, seed=seed)
    if isinstance(source, dict):
        for key in ("coincidence_prob", "rate_hz", "fidelity"):
            if key in source:
                result[key] = source[key]
    return result
