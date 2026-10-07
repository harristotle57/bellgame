# bellgame

The CHSH game, played over simulated photonic links and quantum networks.

Classical players win at most 75% of the rounds; players sharing entangled
pairs can win 85.4%. bellgame plays the game over realistic links and over
networks simulated by [SeQUeNCe](https://github.com/sequence-toolbox/SeQUeNCe)
(entanglement generation, swapping, decaying memories), shows what breaks the
quantum advantage, and uses the same test to build a secret key (E91).

```python
import bellgame as bg

result = bg.play_chsh(bg.optimal_strategy(), bg.bell_pair())
print(result["win_rate"], result["S"])         # 0.854, 2.83
```

## Install

You need [uv](https://docs.astral.sh/uv/getting-started/installation/). Then, in this folder:

```
uv sync
```

That creates `.venv/` with Python 3.13, bellgame, SeQUeNCe, numpy, scipy,
matplotlib, Jupyter, and the Spyder kernel.

Run a part of the tutorial, or an example:

```
uv run python tutorial/01_classical_game.py
uv run python examples/two_star_network.py
```

### Jupyter

The tutorial and examples are plain Python files with `# %%` cells. To open one as a notebook:

```
uv run jupyter lab
```

then right-click a `.py` file, **Open With > Notebook** (jupytext does the conversion).

### Spyder (6.x)

Spyder runs its own Python, so point it at this project's environment:
**Tools > Preferences > Python interpreter > Use the following interpreter**, and choose

* Linux/macOS: `<this folder>/.venv/bin/python`
* Windows: `<this folder>\.venv\Scripts\python.exe`

Restart the console. The `# %%` lines become cells you can run with Shift+Enter.

### VS Code

Open the folder, pick the `.venv` interpreter, and click **Run Cell** above any `# %%`.

## The game

Each round, a referee flips two fair coins and sends one result to each player:
Alice gets a bit **x**, Bob a bit **y**. These questions are ordinary classical
bits, independent of each other and of everything that came before. They are
not qubits, and not halves of a Bell pair. Alice answers a bit **a**, Bob a bit
**b**, without talking to each other. They win when

    a XOR b  ==  x AND y

that is, their answers should differ only when both questions are 1.

Where entanglement comes in: before the questions arrive, the players can share
something. Classical players can share a plan, including shared random numbers;
quantum players also share a Bell pair, one qubit each (in a network, a pair
the network has just delivered). A quantum player turns the question into a
measurement setting (x picks one of two polarizer angles), measures their qubit,
and answers with the outcome (which detector clicked). The questions are what
make it a test: the players don't know in advance which measurement they'll make.

Classical players can be any Python function from the question to an answer.
They can flip coins, or look at their own past rounds (`player(bit, history)`).
None of that beats 75% as long as the questions really are fair coin flips; a
referee whose questions can be predicted can be beaten
(`examples/classical_strategies.py`).

## Probability tables

Every experiment ends in a **probability table** `p(a, b | x, y)`: the chance
of answers a, b for questions x, y. A strategy plus a source of entanglement
makes a table; the game reads the table.

| Source | How to make it |
|---|---|
| a classical team | two functions, `bg.play_classical(alice, bob)` (add `rounds=N` for coins or memory) |
| an ideal or noisy qubit pair | `bg.bell_pair(fidelity)` |
| a photonic link (Fock model) | `bg.link(distance_km=..., mean_photon_number=..., ...)` |
| a game played across a network | `bg.run_network(net, "A1", "B2")` |

`bg.play_chsh(strategy, source)` gives the exact answer; add `rounds=N, seed=...`
to simulate a finite experiment. For a network run, `rounds=run["rounds"]` is the
experiment the network actually delivered.

## Build a network

A network is a plain dict of nodes (quantum routers with memories) and fiber
links, every parameter at its default (`bg.DEFAULTS`). There are several ways to
make one; the last four below build exactly the same Alice–Repeater–Bob chain.

```python
import networkx as nx
import numpy as np
import bellgame as bg

# Ready-made shapes
net = bg.two_player_network(distance_km=20)                  # Alice -- Bob
net = bg.star("Hub", ["A", "B", "C"], leaf_km=5)              # every leaf linked to Hub
net = bg.two_star("HubA", ["A1", "A2"], "HubB", ["B1", "B2"], leaf_km=5, hub_km=20)

# Node by node
net = bg.empty_network()
for name in ["Alice", "Repeater", "Bob"]:
    bg.add_node(net, name)
bg.connect(net, "Alice", "Repeater", 10)                      # km
bg.connect(net, "Repeater", "Bob", 15)

# A list of (node, node, km)
net = bg.from_edges([("Alice", "Repeater", 10), ("Repeater", "Bob", 15)])

# A symmetric distance matrix in km (0 or np.inf = no link)
dist_km = np.array([[0, 10, 0],
                    [10, 0, 15],
                    [0, 15, 0]])
net = bg.from_matrix(["Alice", "Repeater", "Bob"], dist_km)

# A networkx graph with a `length` (km) on each edge
G = nx.Graph()
G.add_edge("Alice", "Repeater", length=10)
G.add_edge("Repeater", "Bob", length=15)
net = bg.from_networkx(G)
```

`from_networkx` also accepts graphs from SeQUeNCe's builders in
`sequence.utils.graphs` (their `attenuation` becomes `loss_db_per_km`). To go
back, `bg.to_matrix(net)` and `bg.to_networkx(net)`; to draw it, `bg.plot_network(net)`.

Change knobs with `set_node` and `set_link`; `bg.parameters(net)` prints every
knob with its unit and meaning.

```python
bg.set_node(net, coherence_time_ms=100)                       # every node
bg.set_node(net, "Repeater", memory_size=20, gate_fidelity=0.99)
bg.set_link(net, "Alice", "Repeater", raw_fidelity=0.95)
bg.set_all_links(net, detector_efficiency=0.9)
```

## Run a simulation

`bg.run_network` simulates the network in SeQUeNCe while a referee asks Alice
and Bob questions, then hands back the pairs they used. Play the game on it like
any other source:

```python
run = bg.run_network(net, "Alice", "Bob", sim_time_s=0.5, seed=1)
print(run["path"])                                            # ['Alice', 'Repeater', 'Bob']
print(run["questions_hz"], run["pairs_hz"], run["rounds_hz"]) # ~1000, ~3990, ~990 per second
print(run["fidelity"], run["pair_age_ms"])                    # ~0.91, ~0.75 ms

result = bg.play_chsh(bg.optimal_strategy(), run)             # exact, from the average pair
print(result["win_rate"], result["S"])                        # ~0.81, ~2.48

finite = bg.play_chsh(bg.optimal_strategy(), run, rounds=run["rounds"], seed=1)
```

`run["links"]` and `run["nodes"]` give each link's pair rate and each node's swap
and memory-expiry rates; `bg.play_history(strategy, run)` replays the rounds one
by one. The same `seed` gives the same run. A sweep is a loop over this: see
`examples/distance_sweep.py` and `examples/two_star_network.py`.

## Tutorial

Step-by-step notebooks (plain Python files with `# %%` cells), from the
classical game to networks. Work through them in order.

| Part | Topic |
|---|---|
| 01 classical game | no classical team beats 75% |
| 02 many rounds | finite experiments only estimate probabilities |
| 03 quantum strategy | entanglement wins 85.4% |
| 04 photons as Fock modes | real sources make 0, 1, 2... pairs |
| 05 parameter studies | loss, dark counts, the detection loophole |
| 06 misalignment | undoing polarization twists with scipy.optimize |
| 07 E91 | a secret key from a Bell test |
| 08 networks | playing across a network: swapping, waiting memories, missing pairs |

## Examples

Short, self-contained scripts, one per part of bellgame: a reference for how
to use it.

| Example | What it shows |
|---|---|
| `distance_sweep.py` | S and rate against distance, over a network and over a bare photonic link |
| `two_star_network.py` | two swaps across a two-star network; sweeps of hub distance, memories and gate quality |
| `classical_strategies.py` | coins, shared randomness and memory don't beat 75%; a predictable referee does; quantum rounds over a network, by pair age (`play_history`) |
| `optimizer_comparison.py` | scipy optimizers undoing a polarization twist |
| `e91_key.py` | key rate from a network run |
| `eavesdropper.py` | an intercept-resend Eve against E91: which angle she should pick, why she can't learn the bases, and how much key survives |

## The physics model, and its assumptions

* **Networks are simulated by SeQUeNCe**, in its Bell-diagonal formalism.
  `bg.run_network` lets SeQUeNCe route a path, make pairs on every link
  (single-heralded generation), swap them at the middle nodes, and keep them in
  memories that decay gradually (`coherence_time_ms`, `memory_errors`). A
  referee asks questions `questions_hz` times per second; the players measure a
  pair they share at that moment, in the state SeQUeNCe says it has decayed to.
  Questions with no pair are discarded, or answered at random (`no_pair`).
* **Supply and demand:** a run reports `questions_hz` (the referee's demand),
  `pairs_hz` (pairs the network delivered: the supply) and `rounds_hz` (rounds
  actually played), which can't beat either of the other two. To find the
  bottleneck, `run["links"]` gives each link's pair-generation rate and
  `run["nodes"]` each node's swap and memory-expiry rates (from SeQUeNCe's own
  event metrics).
* **Memories:** SeQUeNCe reserves the same number of memories on every link of
  the path (`run["memories"]`). The players need one per pair, a middle node one
  per side, so a middle node with `memory_size` 10 allows 5.
* The run's `"state"` is the average pair the players measured. Their
  measurements don't change the network, and outcome probabilities are linear
  in the state, so this gives exactly the statistics of the whole experiment.
  `run["history"]` keeps every question (time, pair age, pair state), and
  `bg.play_history(strategy, run)` plays those rounds one by one, on the pair
  each one actually got.
* **A link's fresh pair** comes from its `link_model`: `"fixed"` (`raw_fidelity`
  at any distance), `"analytic"` (default: dark counts fake a growing share of
  heralds as the fiber gets longer), or `"fock"` (below).
* **Photonic links** (`bg.link`, and `link_model="fock"`) are modeled in Fock
  (photon-number) space: an SPDC source in the middle (two two-mode squeezed
  vacua, so multi-pair emission is included), fiber loss, polarization
  misalignment, polarizers, and threshold detectors with efficiency and dark
  counts. This reuses SeQUeNCe's Fock quantum manager. A network link keeps only
  the Bell-diagonal part of that state, which is all SeQUeNCe can store.
* **Assumptions:** in a network, the Fock source stands in for whatever makes a
  link's pairs (SeQUeNCe's own rate model, memory emission meeting in the middle,
  decides how often pairs appear). Swaps twirl pairs into Werner form
  (SeQUeNCe's default). Polarization twists act on the delivered pair, at the players.
* `no_click="discard"` (the default) keeps only rounds where both sides click:
  the fair-sampling assumption. Tutorial 05 shows what happens without it.

## Using SeQUeNCe and scipy directly

`bg.to_sequence(net)` returns SeQUeNCe's `RouterNetTopo` for your network. To
run it yourself, do so inside `with bg.bell_diagonal_mode():`, which sets
SeQUeNCe's protocols the way bellgame uses them and puts them back afterwards.
`bg.chsh_objective(source)` is an ordinary function you can pass to
`scipy.optimize.minimize`. Tutorial parts 06 and 08 show both.

## Layout

```
src/bellgame/   the package (import bellgame as bg)
tutorial/       step-by-step notebooks, in order
examples/       short reference scripts, one per part of bellgame (sweeps, two-star, E91, ...)
projects/       studies that build to a conclusion, one folder each (copy projects/_template)
tests/          pytest tests: uv run pytest
```
