"""The CHSH game.

The referee flips two fair coins and sends Alice one bit x and Bob another bit
y: ordinary classical bits, independent and uniformly random. Alice answers a
bit a, Bob a bit b, without talking. They win when ``a XOR b == x AND y``.

**Players.** Every player is a function ``player(bit, history)`` that returns an
answer bit, where ``history`` is that player's own past rounds, a list of
``(question, answer)``. On a source of pairs, each player also gets its half of
the round's pair, ``player(bit, history, qubit)``, and may measure it once with
``qubit.measure(angle_deg)`` (see ``Qubit``). A quantum strategy dict, like
``bg.optimal_strategy()``, is shorthand for two players that measure at its
angles (``bg.quantum_players``).

**Tables.** Everything in bellgame boils down to a probability table
``table[a, b, x, y] = p(a, b | x, y)``, a numpy array of shape (2, 2, 2, 2).
The win rate, the correlations and S are read off it.

**Playing.** ``bg.play(team, source)`` plays a team, two player functions
``(alice, bob)`` or a strategy dict, on a source: nothing for a classical game,
or a two-qubit state, a photonic link or a network run.

A source with no data (a network run in which no round was played, whose state
is NaN) gives NaN results and 0 rounds played. Nothing raises, so a parameter
scan carries on past the empty points.

Example:
    >>> import bellgame as bg
    >>> result = bg.play(bg.optimal_strategy(), bg.bell_pair())
    >>> round(result["win_rate"], 4)
    0.8536
"""

import inspect
import itertools

import numpy as np

from .defaults import check
from .strategies import check_strategy, measurement_projectors, measurement_setting, projectors

CLASSICAL_WIN_RATE = 0.75
TSIRELSON_WIN_RATE = (2 + np.sqrt(2)) / 4  # = cos^2(pi/8) ~ 0.854
CLASSICAL_S = 2.0
TSIRELSON_S = 2 * np.sqrt(2)
PROBES = 16  # how many times each player answers each question to show it always does the same


def referee_wins(x, y, a, b):
    """True when the answers a, b win for questions x, y.

    Example:
        >>> referee_wins(1, 1, 0, 1)
        True
    """
    return (a ^ b) == (x & y)


def _no_counts():
    return {(x, y): {"00": 0, "01": 0, "10": 0, "11": 0} for x in (0, 1) for y in (0, 1)}


# ---------------------------------------------------------------- tables


def outcome_probabilities(source, alice_deg, bob_deg, alice_correction=None, bob_correction=None):
    """p[a, b]: the chance of outcomes a, b when Alice measures at ``alice_deg`` and Bob at ``bob_deg``.

    Each player's correction wave plates (three angles, if any) act before their
    polarizer. ``source`` is any source of pairs:

    * a 4x4 two-qubit density matrix (e.g. ``bg.bell_pair(0.9)``)
    * the dict returned by ``bg.run_network`` (its average delivered ``"state"``)
    * a photonic link dict from ``bg.link(...)`` (Fock model, detectors included)

    Every quantum table in bellgame is built from this: a CHSH table is it at four
    pairs of angles, E91 at nine. A source with no data gives NaN.

    Example:
        >>> import bellgame as bg
        >>> np.round(outcome_probabilities(bg.bell_pair(), 0, 0), 3).tolist()
        [[0.5, 0.0], [0.0, 0.5]]
    """
    alice = measurement_setting(alice_deg, alice_correction)
    bob = measurement_setting(bob_deg, bob_correction)
    kind = _source_kind(source)
    if kind is None:
        raise ValueError("outcome probabilities need a source of pairs, e.g. bg.bell_pair()")
    if kind == "link":
        from .link import link_outcome_probabilities

        return link_outcome_probabilities(source, alice, bob)
    rho = np.asarray(source["state"] if kind == "dict" else source)
    pa, pb = projectors(*alice), projectors(*bob)
    p = np.array([[np.real(np.trace(np.kron(pa[a], pb[b]) @ rho)) for b in (0, 1)] for a in (0, 1)])
    return np.clip(p, 0, 1)


def table_from_strategy(strategy, source):
    """Probability table ``table[a, b, x, y]`` for a quantum strategy on any source of pairs.

    It is ``outcome_probabilities`` at the strategy's angles for each pair of questions.

    Example:
        >>> import bellgame as bg
        >>> table_from_strategy(bg.optimal_strategy(), bg.bell_pair()).shape
        (2, 2, 2, 2)
    """
    check_strategy(strategy)
    table = np.zeros((2, 2, 2, 2))
    for x, y in itertools.product((0, 1), repeat=2):
        table[:, :, x, y] = outcome_probabilities(
            source,
            strategy["alice"][x],
            strategy["bob"][y],
            strategy.get("alice_correction"),
            strategy.get("bob_correction"),
        )
    return table


# ---------------------------------------------------------------- numbers from tables and counts


def win_rate(table):
    """Chance of winning when x and y are uniformly random.

    Example:
        >>> win_rate(np.full((2, 2, 2, 2), 0.25))
        0.5
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
        >>> import bellgame as bg
        >>> np.round(correlations(table_from_strategy(bg.optimal_strategy(), bg.bell_pair())), 3)
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
        >>> import bellgame as bg
        >>> round(chsh_value(table_from_strategy(bg.optimal_strategy(), bg.bell_pair())), 4)
        2.8284
    """
    e = correlations(table)
    return float(e[0, 0] + e[0, 1] + e[1, 0] - e[1, 1])


def sample_counts(table, rounds, seed=None):
    """Play ``rounds`` random rounds from a table, return counts per question pair.

    The result looks like ``{(x, y): {"00": n, "01": n, "10": n, "11": n}}``,
    where the string is the answers ``a`` then ``b``.

    Example:
        >>> counts = sample_counts(np.full((2, 2, 2, 2), 0.25), 100, seed=1)
        >>> sum(sum(c.values()) for c in counts.values())
        100
    """
    rounds = check("rounds", rounds)
    rng = np.random.default_rng(seed)
    xs, ys = _questions(rounds, rng, None)
    return _counts_from_table(table, xs, ys, rng)


def _counts_from_table(table, xs, ys, rng):
    """Counts for the questions ``xs``, ``ys``, with answers drawn from the table."""
    counts = _no_counts()
    if np.isnan(table).any():  # a source with no data: there are no pairs to play
        return counts
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


def count_rounds(xs, ys, a, b):
    """Counts ``{(x, y): {"00": n, "01": n, "10": n, "11": n}}`` of played rounds.

    ``xs``, ``ys`` are each round's questions and ``a``, ``b`` its answers.

    Example:
        >>> count_rounds([0, 1], [1, 1], [0, 1], [1, 1])[(1, 1)]
        {'00': 0, '01': 0, '10': 0, '11': 1}
    """
    counts = _no_counts()
    for x, y, ai, bi in zip(xs, ys, a, b):
        counts[int(x), int(y)][f"{ai}{bi}"] += 1
    return counts


def table_from_counts(counts):
    """Turn sampled counts back into an (estimated) probability table.

    A pair of questions that was never asked has no estimate: its part of the
    table is NaN, and so is anything computed from it, like S.
    """
    table = np.zeros((2, 2, 2, 2))
    for (x, y), c in counts.items():
        n = sum(c.values())
        if n == 0:
            table[:, :, x, y] = np.nan
            continue
        for ab, k in c.items():
            table[int(ab[0]), int(ab[1]), x, y] = k / n
    return table


def summarize_counts(counts):
    """Result dict for an experiment's counts: win rate, S and correlations as measured.

    S needs every pair of questions to have been asked at least once; until then it
    is NaN. With no rounds at all, the win rate is NaN too.
    """
    rounds = sum(sum(c.values()) for c in counts.values())
    wins = sum(k for (x, y), c in counts.items() for ab, k in c.items() if referee_wins(x, y, int(ab[0]), int(ab[1])))
    table = table_from_counts(counts)
    return {
        "rounds": rounds,
        "wins": wins,
        "counts": counts,
        "win_rate": wins / rounds if rounds else float("nan"),
        "S": chsh_value(table),
        "E": np.round(correlations(table), 6).tolist(),
        "table": table,
    }


def _summarize_table(table):
    return {
        "win_rate": win_rate(table),
        "S": chsh_value(table),
        "E": np.round(correlations(table), 6).tolist(),
        "table": table,
    }


# ---------------------------------------------------------------- players


class Qubit:
    """One player's half of this round's pair.

    ``qubit.measure(angle_deg, correction_deg=None)`` measures it with a polarizer
    at ``angle_deg`` degrees, after optional correction wave plates (three angles,
    see ``bg.polarization_rotation``), and returns the outcome, 0 or 1. Each half
    can be measured once. The outcomes of the two halves are correlated as the
    pair's state says, whoever measures first.
    """

    def __init__(self, pair, side):
        self._pair, self._side = pair, side

    def measure(self, angle_deg, correction_deg=None):
        return self._pair.measure(self._side, angle_deg, correction_deg)


class _Pair:
    """A two-qubit state that collapses as its halves are measured."""

    def __init__(self, rho, rng, projectors):
        self.rho, self.rng, self.projectors = rho, rng, projectors
        self.measured = [False, False]

    def measure(self, side, angle_deg, correction_deg):
        if self.measured[side]:
            raise ValueError("a qubit can only be measured once per round")
        self.measured[side] = True
        ops = [
            np.kron(p, np.eye(2)) if side == 0 else np.kron(np.eye(2), p)
            for p in self.projectors(angle_deg, correction_deg)
        ]
        p0 = np.real(np.trace(ops[0] @ self.rho))
        outcome = int(self.rng.random() >= p0)
        after = ops[outcome] @ self.rho @ ops[outcome]
        self.rho = after / np.real(np.trace(after))
        return outcome


def _projector_cache():
    """``projectors``, remembering each setting's projectors for the rest of a game."""
    cache = {}

    def cached(angle_deg, correction_deg=None):
        key = measurement_setting(angle_deg, correction_deg)
        if key not in cache:
            cache[key] = projectors(*key)
        return cache[key]

    return cached


def _check_player(name, player, quantum):
    args = ("bit", "history", "qubit") if quantum else ("bit", "history")
    wanted = f"{name}({', '.join(args)})"
    if not callable(player):
        raise TypeError(f"{name} must be a function {wanted}, got {player!r}")
    try:
        signature = inspect.signature(player)
    except (TypeError, ValueError):
        return  # no signature to check (some built-ins); calling it will tell
    try:
        signature.bind(*args)
    except TypeError:
        raise TypeError(
            f"{name} must take {len(args)} arguments, {wanted}: the question, its own past rounds"
            f"{' and its half of the pair' if quantum else ''}. Got {name}{signature}"
        ) from None


def _answer(name, player, bit, history, *qubit):
    """Ask a player for its answer bit, giving it a copy of its history (and its qubit, on a source)."""
    a = player(bit, list(history), *qubit)
    if a not in (0, 1):
        raise ValueError(f"{name} must answer 0 or 1, got {a!r} for question {bit}")
    return int(a)


class _Probe:
    """A stand-in qubit that notes how it is measured and gives a set outcome."""

    def __init__(self, outcome):
        self.outcome, self.setting = outcome, None

    def measure(self, angle_deg, correction_deg=None):
        if self.setting is not None:
            raise ValueError("a qubit can only be measured once per round")
        self.setting = measurement_setting(angle_deg, correction_deg)
        return self.outcome


def _fixed_play(name, player, quantum, link):
    """For each question, ``(setting, [answer if outcome 0, answer if outcome 1])``.

    Only a player that always does the same for the same question has one. It is
    asked each question ``PROBES`` times, with made-up histories, and must always
    measure the same way and answer the same; otherwise this is a ValueError.
    A player that is random only rarely (say 1% of the time) usually gets through.
    """
    rng = np.random.default_rng(0)
    plays = []
    for bit in (0, 1):
        seen = set()
        for _ in range(PROBES):
            history = [(int(q), int(a)) for q, a in rng.integers(0, 2, size=(int(rng.integers(0, 6)), 2))]
            if not quantum:
                answer = _answer(name, player, bit, history)
                seen.add((None, None, answer, answer))
                continue
            probes = [_Probe(0), _Probe(1)]
            answers = [_answer(name, player, bit, history, p) for p in probes]
            seen.add((probes[0].setting, probes[1].setting, *answers))
        if len(seen) > 1:
            fix = (
                "over a photonic link, only such players can play (bellgame computes a link's statistics "
                "for fixed measurement settings)"
                if link
                else "play rounds instead: bg.play(team, source, rounds=10000)"
            )
            raise ValueError(
                f"{name} doesn't always do the same for question {bit} (it flips coins or looks "
                f"at its past rounds), so there is no exact answer; {fix}"
            )
        setting, other, *answers = seen.pop()
        if setting != other:
            raise ValueError(f"{name} measures its qubit differently depending on the outcome it got")
        plays.append((setting or (0.0, None), answers))
    return plays


def _players_table(alice, bob, source, kind):
    """(table, extras): the exact table for two players who always do the same for the same question."""
    quantum, link = kind is not None, kind == "link"
    plays_a = _fixed_play("alice", alice, quantum, link)
    plays_b = _fixed_play("bob", bob, quantum, link)
    table = np.zeros((2, 2, 2, 2))
    if not quantum:
        for x in (0, 1):
            for y in (0, 1):
                table[plays_a[x][1][0], plays_b[y][1][0], x, y] = 1.0
        return table, {}
    for x, y in itertools.product((0, 1), repeat=2):
        (angle_a, correction_a), map_a = plays_a[x]
        (angle_b, correction_b), map_b = plays_b[y]
        outcomes = outcome_probabilities(source, angle_a, angle_b, correction_a, correction_b)
        for oa, ob in itertools.product((0, 1), repeat=2):
            table[map_a[oa], map_b[ob], x, y] += outcomes[oa, ob]
    return table, _link_extras(source, kind)


def _strategy_table(strategy, source, kind):
    """(table, extras) for a strategy dict on a source."""
    return table_from_strategy(strategy, source), _link_extras(source, kind)


def _link_extras(source, kind):
    """What a result adds on a photonic link: ``coincidence_prob`` (see ``bg.coincidence_prob``)."""
    if kind != "link":
        return {}
    from .link import coincidence_prob

    return {"coincidence_prob": coincidence_prob(source)}


def _questions(n, rng, questions):
    """The referee's questions for ``n`` rounds: fair coins, or ``questions(round number)``."""
    if questions is None:
        return rng.integers(0, 2, size=(2, n))
    xs, ys = np.zeros(n, dtype=int), np.zeros(n, dtype=int)
    for i in range(n):
        x, y = questions(i)
        if x not in (0, 1) or y not in (0, 1):
            raise ValueError(f"questions must be bits, got x={x!r}, y={y!r} in round {i}")
        xs[i], ys[i] = x, y
    return xs, ys


def _play_rounds(alice, bob, xs, ys, rng, state_of=None):
    """Answers ``a``, ``b`` from two player functions, round by round.

    ``state_of(i)`` is the pair for round ``i`` (None for a classical game).
    """
    cached_projectors = _projector_cache()
    a, b = np.zeros(len(xs), dtype=int), np.zeros(len(xs), dtype=int)
    alice_past, bob_past = [], []
    for i, (x, y) in enumerate(zip(xs, ys)):
        x, y = int(x), int(y)
        qubits = ((), ())
        if state_of is not None:
            pair = _Pair(state_of(i), rng, cached_projectors)
            qubits = ((Qubit(pair, 0),), (Qubit(pair, 1),))
        a[i] = _answer("alice", alice, x, alice_past, *qubits[0])
        b[i] = _answer("bob", bob, y, bob_past, *qubits[1])
        alice_past.append((x, int(a[i])))
        bob_past.append((y, int(b[i])))
    return a, b


def _source_kind(source):
    if source is None:
        return None
    if isinstance(source, dict):
        if "mean_photon_number" in source:
            return "link"
        if "state" in source:
            return "dict"
        raise ValueError("source dict must be a link (bg.link) or a network run (bg.run_network)")
    if np.shape(source) != (4, 4):
        raise ValueError(f"a two-qubit state must be a 4x4 matrix, got shape {np.shape(source)}")
    return "state"


def _team(team):
    """(strategy, alice, bob): a strategy dict, or two player functions."""
    if isinstance(team, dict):
        check_strategy(team)
        return team, None, None
    if isinstance(team, (tuple, list)) and len(team) == 2:
        return None, team[0], team[1]
    raise TypeError(
        "a team is a quantum strategy dict (like bg.optimal_strategy()) "
        f"or two player functions (alice, bob), got {team!r}"
    )


# ---------------------------------------------------------------- playing


def play(team, source=None, rounds=None, seed=None, questions=None):
    """Play the CHSH game.

    ``team`` is two player functions ``(alice, bob)`` or a quantum strategy dict
    (shorthand for ``bg.quantum_players(strategy)``). ``source`` is what the
    players share:

    * nothing (``None``): a classical game, players are ``player(bit, history)``
    * a 4x4 two-qubit state, e.g. ``bg.bell_pair(0.9)``
    * a photonic link from ``bg.link(...)``
    * a network run from ``bg.run_network(...)`` (its average pair, ``"state"``)

    On a source, players are ``player(bit, history, qubit)``: they may measure
    their half of the round's pair (``bg.Qubit``), or ignore it.

    ``rounds`` picks how the game is played:

    * ``None``: the exact win rate and S. Only for players who always do the same
      for the same question (each player is asked each question several times
      to check). A player that is random only rarely (say 1% of the time) usually
      gets through and is then given a wrong exact table, so play any player that
      uses randomness for a number of rounds.
    * ``N``: an experiment of N rounds, played one at a time, so players may flip
      coins or look at their past rounds. A strategy dict, which does neither, is
      drawn straight from its table instead, which is much faster. Over a
      photonic link only players who always do the same can play.
    * ``"run"``: replay a network run's own questions, each on the pair it
      actually got, in the state it had decayed to. Questions without a pair are
      skipped when the run used ``no_pair="discard"``, and answered on a useless
      (maximally mixed) pair with ``no_pair="random"``. The result also has
      ``history``: the run's history (``t_ms``, ``had_pair``, ``age_ms``,
      ``states``) with, for every question, ``x``, ``y``, ``a``, ``b`` (-1 when
      not played), ``played`` and ``win``.

    ``questions`` swaps the fair referee for your own: a function of the round
    number (0, 1, 2, ...) that returns ``(x, y)``. It needs ``rounds``.

    Returns a dict with ``win_rate``, ``S``, the correlations ``E`` and the
    probability ``table``. Experiments add ``rounds``, ``wins`` and ``counts``,
    and their ``table`` is the measured one. A link adds ``coincidence_prob``, a
    network run passes on its ``rounds_hz`` and ``fidelity``.

    Example:
        >>> import bellgame as bg
        >>> bg.play((bg.always_zero, bg.always_zero))["win_rate"]
        0.75
        >>> import random
        >>> coin = lambda bit, history: random.randint(0, 1)
        >>> bg.play((coin, coin), rounds=10000, seed=1)["win_rate"] < 0.75
        True
        >>> bg.play(bg.optimal_strategy(), bg.bell_pair(0.9), rounds=1000, seed=7)["rounds"]
        1000
    """
    strategy, alice, bob = _team(team)
    kind = _source_kind(source)
    if strategy is not None and kind is None:
        raise ValueError("a quantum strategy needs a source of pairs, e.g. bg.play(strategy, bg.bell_pair())")
    if strategy is None:
        _check_player("alice", alice, kind is not None)
        _check_player("bob", bob, kind is not None)
    if rounds is None:
        if questions is not None:
            raise ValueError("a custom referee (questions=) needs an experiment: pass rounds=N")
        table, extras = (
            _strategy_table(strategy, source, kind)
            if strategy is not None
            else _players_table(alice, bob, source, kind)
        )
        result = _summarize_table(table)
    elif isinstance(rounds, str) and rounds == "run":
        if kind != "dict" or "history" not in source:
            raise ValueError(
                'rounds="run" replays a network run\'s own questions: source must be the dict '
                "returned by bg.run_network"
            )
        result, extras = _replay(strategy, alice, bob, source, np.random.default_rng(seed), questions), {}
    else:
        rounds = check("rounds", rounds)
        rng = np.random.default_rng(seed)
        if strategy is not None or kind == "link":  # fixed settings: draw from the table
            table, extras = (
                _strategy_table(strategy, source, kind)
                if strategy is not None
                else _players_table(alice, bob, source, kind)
            )
            xs, ys = _questions(rounds, rng, questions)
            counts = _counts_from_table(table, xs, ys, rng)
        else:
            extras = {}
            state = None if kind is None else np.asarray(source["state"] if kind == "dict" else source)
            xs, ys = _questions(rounds, rng, questions)
            if state is not None and np.isnan(state).any():  # a source with no data
                counts = _no_counts()
            else:
                a, b = _play_rounds(alice, bob, xs, ys, rng, None if state is None else lambda i: state)
                counts = count_rounds(xs, ys, a, b)
        result = summarize_counts(counts)
    result.update(extras)
    if kind == "dict":
        for key in ("coincidence_prob", "rounds_hz", "fidelity"):
            if key in source:
                result[key] = source[key]
    return result


def _replay(strategy, alice, bob, run, rng, questions):
    """Play a network run's recorded questions, each on its own pair (``rounds="run"``)."""
    h = run["history"]
    n = len(h["t_ms"])
    x, y = _questions(n, rng, questions)
    played = np.array(h["had_pair"], dtype=bool)
    states = np.where(played[:, None, None], h["states"], np.eye(4) / 4)  # no pair: a useless one
    if run.get("no_pair", "discard") == "random":
        played = np.ones(n, dtype=bool)
    a, b = np.full(n, -1), np.full(n, -1)
    if strategy is not None:
        # P(a, b | x, y) = Tr(rho (Pa_x (x) Pb_y)) for each round's own state and questions.
        # ops[x, y, a, b] is the 4x4 operator Pa_x (x) Pb_y.
        ops = np.zeros((2, 2, 2, 2, 4, 4), dtype=complex)
        for qx, qy in itertools.product((0, 1), repeat=2):
            pa, pb = measurement_projectors(strategy, "alice", qx), measurement_projectors(strategy, "bob", qy)
            for oa, ob in itertools.product((0, 1), repeat=2):
                ops[qx, qy, oa, ob] = np.kron(pa[oa], pb[ob])
        probs = np.real(np.einsum("nij,nabji->nab", states, ops[x, y])).reshape(n, 4)
        probs = np.clip(probs, 0, None)
        probs /= probs.sum(axis=1, keepdims=True)
        outcome = np.minimum((rng.random(n)[:, None] > np.cumsum(probs, axis=1)).sum(axis=1), 3)
        a[played], b[played] = outcome[played] // 2, outcome[played] % 2
    else:
        rows = np.flatnonzero(played)
        a[rows], b[rows] = _play_rounds(alice, bob, x[rows], y[rows], rng, lambda i: states[rows[i]])
    win = played & ((a ^ b) == (x & y))
    result = summarize_counts(count_rounds(x[played], y[played], a[played], b[played]))
    result["history"] = {**h, "x": x, "y": y, "a": a, "b": b, "played": played, "win": win}
    return result
