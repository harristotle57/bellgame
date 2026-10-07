# %% [markdown]
# # Can a classical team beat 75%?
#
# A classical player is a function from the question bit to an answer bit. Here
# we give the players more to work with (coins, shared random numbers, memory
# of past rounds) and see what, if anything, gets them past 75%. Last, a quantum
# team over a network, round by round, for comparison.

# %%
import itertools

import numpy as np

import bellgame as bg

ROUNDS = 20_000  # about +-0.003 statistical wobble on a win rate

# %% [markdown]
# ## 1. Fixed answers
#
# There are four functions from one bit to one bit, so sixteen teams. The best
# win 75% exactly.

# %%
best = max(bg.play_classical(a, b)["win_rate"]
           for a, b in itertools.product(bg.CLASSICAL_PLAYERS.values(), repeat=2))
print("best fixed team:", best)

# %% [markdown]
# ## 2. Coins
#
# Each player flips their own coin, or both read the same list of random bits
# (shared randomness, agreed before the game). Either way, each round they're
# playing one of the sixteen fixed teams, chosen at random, so the average can't
# beat the best of them.

# %%
own = np.random.default_rng(101)  # not the referee's seed: see part 4


def coin(bit):
    return int(own.integers(2))


shared_a, shared_b = np.random.default_rng(7), np.random.default_rng(7)  # same seed: same random bits


def alice_shared(x):
    return int(shared_a.integers(2))


def bob_shared(y):
    return int(shared_b.integers(2))


print("own coins:   ", bg.play_classical(coin, coin, rounds=ROUNDS, seed=1)["win_rate"])
print("shared coins:", bg.play_classical(alice_shared, bob_shared, rounds=ROUNDS, seed=1)["win_rate"])

# %% [markdown]
# ## 3. Memory
#
# A player written as `player(bit, history)` sees their own past rounds as
# `(question, answer)` pairs. The questions are fresh fair coin flips every
# round, so nothing in the past says anything about this round's questions.
# Memory can't help on average (it can make a short experiment wobble more,
# which is why real Bell tests worry about it: the *memory loophole*).


# %%
def last_question(x, history):  # answer with my previous question
    return history[-1][0] if history else 0


def streak(x, history):  # copy my question, but flip after three 1s in a row
    recent = [q for q, _ in history[-3:]]
    return x ^ (recent == [1, 1, 1])


for name, (a, b) in {"last_question + copy_bit": (last_question, bg.copy_bit),
                     "streak + always_zero": (streak, bg.always_zero),
                     "streak + streak": (streak, streak)}.items():
    print(f"{name:26s}", bg.play_classical(a, b, rounds=ROUNDS, seed=2)["win_rate"])

# %% [markdown]
# ## 4. A referee you can predict
#
# The 75% limit assumes the questions are fair, independent coin flips. If the
# players can predict them, Alice knows Bob's question and they win every round.
# A *biased* referee helps too: if x = y = 1 is rare, "always answer 0" nearly
# always wins. That is why Bell tests use the best random number generators they can.
# (Try it: seed the players' coins in part 2 with the referee's seed, 1. The
# "coins" then repeat the referee's questions, and the win rate changes.)

# %%
order = [(0, 0), (0, 1), (1, 0), (1, 1)]


def cycling_referee(n):
    return order[n % 4]


def alice_knows_order(x, history):
    y = order[len(history) % 4][1]  # the round number tells her Bob's question
    return x & y


biased = np.random.default_rng(3)


def biased_referee(n):
    x, y = (int(b) for b in biased.integers(2, size=2))
    return (0, 0) if x == y == 1 and biased.random() < 0.8 else (x, y)  # makes (1, 1) rarer


print("predictable referee:", bg.play_classical(alice_knows_order, bg.always_zero, rounds=ROUNDS,
                                                 questions=cycling_referee)["win_rate"])
print("biased referee:     ", bg.play_classical(bg.always_zero, bg.always_zero, rounds=ROUNDS,
                                                 questions=biased_referee)["win_rate"])

# %% [markdown]
# ## 5. Quantum players, round by round
#
# For comparison, a quantum team playing across a network: Alice and Bob 20 km
# apart, memories that decay in 2 ms. `bg.play_history` plays every question the
# referee asked during the run, on the pair that question actually got, in the
# state it had decayed to by then. Sorting the rounds by how long their pair had
# waited shows where the quantum advantage goes: fresh pairs beat 75%, old ones
# don't. (No pair is older than 2 ms: SeQUeNCe throws pairs away at the coherence time.)

# %%
net = bg.two_player_network(20)
bg.set_node(net, coherence_time_ms=2)
run = bg.run_network(net, "Alice", "Bob", sim_time_s=2.0, seed=1)
quantum = bg.play_history(bg.optimal_strategy(), run, seed=1)
print(f"quantum team: {quantum['rounds']} rounds, win rate {quantum['win_rate']:.3f}")

history = quantum["history"]
played = history["played"]
age_ms, win, weights = history["age_ms"][played], history["win"][played], history["weights"][played]
for lo, hi in [(0, 0.4), (0.4, 0.6), (0.6, 2.0)]:
    in_bin = (age_ms >= lo) & (age_ms < hi)
    # the exact win rate on the average pair in this bin, for comparison
    expected = bg.play_chsh(bg.optimal_strategy(), bg.bell_diagonal(weights[in_bin].mean(axis=0)))
    print(f"pairs {lo:.1f}-{hi:.1f} ms old: won {win[in_bin].mean():.3f} of {in_bin.sum()} rounds "
          f"(expected {expected['win_rate']:.3f})")
