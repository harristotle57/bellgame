"""The CHSH game.

The referee flips two fair coins and sends Alice one bit x and Bob another bit
y: ordinary classical bits, independent and uniformly random. Alice answers a
bit a, Bob a bit b, without talking. They win when ``a XOR b == x AND y``.

Everything in bellgame boils down to a **probability table**
``table[a, b, x, y] = p(a, b | x, y)``, a numpy array of shape (2, 2, 2, 2).
Strategies plus sources make tables; the game reads tables.

Example:
    >>> import bellgame as bg
    >>> result = bg.play_chsh(bg.optimal_strategy(), bg.bell_pair())
    >>> round(result["win_rate"], 4)
    0.8536
"""

import inspect

import numpy as np

from .qubits import bell_diagonal
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


def uses_history(player):
    """True when a classical player takes the past rounds as a second argument: ``player(bit, history)``."""
    try:
        return len(inspect.signature(player).parameters) >= 2
    except (TypeError, ValueError):
        return False


def answer(player, bit, history=()):
    """Ask a classical player for its answer bit (passing ``history`` if it wants it)."""
    a = player(bit, list(history)) if uses_history(player) else player(bit)
    if a not in (0, 1):
        raise ValueError(f"players must answer 0 or 1, got {a!r} for question {bit}")
    return int(a)


def table_from_classical(alice, bob):
    """Exact probability table for two classical players who always give the same answer to the same question.

    Players who flip coins, or look at past rounds, have no single table;
    play them for a number of rounds instead (``bg.play_classical(..., rounds=N)``).

    Example:
        >>> from bellgame.strategies import always_zero
        >>> win_rate(table_from_classical(always_zero, always_zero))
        0.75
    """
    answers = {}
    for name, player in (("alice", alice), ("bob", bob)):
        if uses_history(player):
            raise ValueError(f"{name} looks at past rounds, so there is no exact answer; "
                             "play an experiment instead: bg.play_classical(alice, bob, rounds=10000)")
        for bit in (0, 1):
            seen = {answer(player, bit) for _ in range(16)}
            if len(seen) > 1:
                raise ValueError(f"{name} answers question {bit} at random, so there is no exact answer; "
                                 "play an experiment instead: bg.play_classical(alice, bob, rounds=10000)")
            answers[name, bit] = seen.pop()
    table = np.zeros((2, 2, 2, 2))
    for x in (0, 1):
        for y in (0, 1):
            table[answers["alice", x], answers["bob", y], x, y] = 1.0
    return table


def play_rounds(alice, bob, rounds, seed=None, questions=None):
    """Play ``rounds`` rounds one at a time with two classical players; return counts like ``sample_counts``.

    Each round the referee picks the questions, then each player answers. A
    player can be ``player(bit)`` or ``player(bit, history)``, where ``history``
    is that player's own past rounds as a list of ``(question, answer)``. Players
    may also flip their own coins. ``questions`` replaces the fair referee with
    your own: a function of the round number (0, 1, 2, ...) that returns ``(x, y)``.

    Example:
        >>> from bellgame.strategies import always_zero
        >>> counts = play_rounds(always_zero, always_zero, 100, seed=1)
        >>> sum(sum(c.values()) for c in counts.values())
        100
    """
    if rounds < 1:
        raise ValueError(f"rounds must be at least 1, got {rounds}")
    rng = np.random.default_rng(seed)
    counts = {(x, y): {"00": 0, "01": 0, "10": 0, "11": 0} for x in (0, 1) for y in (0, 1)}
    alice_past, bob_past = [], []
    for n in range(rounds):
        if questions is None:
            x, y = (int(q) for q in rng.integers(0, 2, size=2))
        else:
            x, y = questions(n)
            if x not in (0, 1) or y not in (0, 1):
                raise ValueError(f"questions must be bits, got x={x!r}, y={y!r} in round {n}")
        a, b = answer(alice, x, alice_past), answer(bob, y, bob_past)
        alice_past.append((x, a))
        bob_past.append((y, b))
        counts[(x, y)][f"{a}{b}"] += 1
    return counts


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
    * the dict returned by ``bg.run_network`` (uses its ``"state"``)
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
        raise ValueError("source dict must be a link (bg.link) or a network run (bg.run_network)")
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
    if rounds is not None:
        result = summarize_counts(sample_counts(table, rounds, seed=seed))
        result["table"] = table
        return result
    return {"win_rate": win_rate(table), "S": chsh_value(table), "E": np.round(correlations(table), 6).tolist(),
            "table": table}


def summarize_counts(counts):
    """Result dict for an experiment's counts: win rate, S and correlations as measured."""
    rounds = sum(sum(c.values()) for c in counts.values())
    wins = sum(k for (x, y), c in counts.items() for ab, k in c.items()
               if referee_wins(x, y, int(ab[0]), int(ab[1])))
    table = table_from_counts(counts)
    return {"rounds": rounds, "wins": wins, "counts": counts, "win_rate": wins / rounds, "S": chsh_value(table),
            "E": np.round(correlations(table), 6).tolist(), "table": table}


def play_classical(alice, bob, rounds=None, seed=None, questions=None):
    """Play the CHSH game with two classical player functions.

    With ``rounds=None`` you get the exact win rate (only for players who always
    give the same answer to the same question). With a number, the players
    really play that many rounds, one at a time, so they may flip coins or look
    at their own past rounds (``player(bit, history)``, see ``bg.play_rounds``).
    ``questions`` swaps in your own referee.

    Example:
        >>> from bellgame.strategies import always_zero
        >>> play_classical(always_zero, always_zero)["win_rate"]
        0.75
        >>> import random
        >>> coin = lambda bit: random.randint(0, 1)
        >>> play_classical(coin, coin, rounds=10000, seed=1)["win_rate"] < 0.75
        True
    """
    if rounds is None:
        if questions is not None:
            raise ValueError("a custom referee (questions=) needs an experiment: pass rounds=N")
        return summarize(table_from_classical(alice, bob))
    return summarize_counts(play_rounds(alice, bob, rounds, seed=seed, questions=questions))


def play_chsh(strategy, source, rounds=None, seed=None):
    """Play the CHSH game with a quantum strategy on some source of entanglement.

    Returns a dict with ``win_rate``, ``S``, the correlations ``E``, and the
    probability ``table``. With ``rounds=N`` it also has ``counts`` and ``wins``.
    When the source reports a herald or pair rate, that is passed through too.
    To play a network run's own rounds one at a time instead, use ``bg.play_history``.

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
        for key in ("coincidence_prob", "rounds_hz", "fidelity"):
            if key in source:
                result[key] = source[key]
    return result


def play_history(strategy, run, seed=None):
    """Play each question of a network run as its own round, on the pair it actually used.

    ``bg.play_chsh(strategy, run)`` uses the run's average state, which gives the
    exact win rate. This plays the recorded rounds one by one instead: the referee
    flips its coins, and the players measure the pair that question got, in the
    state it had decayed to. So you can ask questions an average can't answer,
    like whether old pairs lose more often.

    Questions without a pair are skipped when the run used ``no_pair="discard"``,
    and answered at random with ``no_pair="random"``.

    Returns the same dict as ``bg.summarize_counts``, plus ``history``: the run's
    history (``t_ms``, ``had_pair``, ``age_ms``, ``weights``) with, for every
    question, ``x``, ``y``, ``a``, ``b`` (-1 when not played), ``played`` and ``win``.

    Example:
        >>> import bellgame as bg
        >>> run = bg.run_network(bg.two_player_network(5), "Alice", "Bob", sim_time_s=0.2, seed=1)
        >>> result = play_history(bg.optimal_strategy(), run, seed=1)
        >>> result["rounds"] == run["rounds"]
        True
    """
    if not isinstance(run, dict) or "history" not in run:
        raise ValueError("play_history needs the dict returned by bg.run_network")
    h = run["history"]
    n = len(h["t_ms"])
    rng = np.random.default_rng(seed)
    x, y = rng.integers(0, 2, size=(2, n))
    played = np.array(h["had_pair"], dtype=bool)
    weights = np.where(played[:, None], h["weights"], 0.25)  # no pair: random answers
    if run.get("no_pair", "discard") == "random":
        played = np.ones(n, dtype=bool)
    if not played.any():
        raise ValueError("no question in this run had a pair to play on")
    # a table is linear in the state, so each round's table mixes the four Bell states' tables
    bell_tables = np.stack([table_from_state(strategy, bell_diagonal(np.eye(4)[k])) for k in range(4)])
    probs = np.einsum("nk,kabn->nab", weights, bell_tables[:, :, :, x, y]).reshape(n, 4)
    probs /= probs.sum(axis=1, keepdims=True)
    outcome = np.minimum((rng.random(n)[:, None] > np.cumsum(probs, axis=1)).sum(axis=1), 3)
    a, b = np.where(played, outcome // 2, -1), np.where(played, outcome % 2, -1)
    win = played & ((a ^ b) == (x & y))
    counts = {(i, j): {"00": 0, "01": 0, "10": 0, "11": 0} for i in (0, 1) for j in (0, 1)}
    for i, j, ai, bi in zip(x[played], y[played], a[played], b[played]):
        counts[int(i), int(j)][f"{ai}{bi}"] += 1
    result = summarize_counts(counts)
    result["history"] = {**h, "x": x, "y": y, "a": a, "b": b, "played": played, "win": win}
    return result
