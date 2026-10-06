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

Run a lab or example:

```
uv run python labs/01_classical_game.py
uv run python examples/two_star_network.py
```

### Jupyter

The labs are plain Python files with `# %%` cells. To open one as a notebook:

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

## Labs

Step-by-step notebooks (plain Python files with `# %%` cells), from the
classical game to networks.

| Lab | Topic |
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

| Example | What it shows |
|---|---|
| `distance_sweep.py` | S and rate against distance, over a network and over a bare photonic link |
| `two_star_network.py` | two swaps across a two-star network; sweeps of hub distance, memories and gate quality |
| `classical_strategies.py` | coins, shared randomness and memory don't beat 75%; a predictable referee does |
| `optimizer_comparison.py` | scipy optimizers undoing a polarization twist |
| `e91_key.py` | key rate from a network run |

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
  actually played), which can't beat either of the other two.
* The run's `"state"` is the average pair the players measured. Their
  measurements don't change the network, and outcome probabilities are linear
  in the state, so this gives exactly the statistics of the whole experiment.
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
  the fair-sampling assumption. Lab 05 shows what happens without it.

## Using SeQUeNCe and scipy directly

`bg.to_sequence(net)` returns SeQUeNCe's `RouterNetTopo` for your network. To
run it yourself, do so inside `with bg.bell_diagonal_mode():`, which sets
SeQUeNCe's protocols the way bellgame uses them and puts them back afterwards.
`bg.chsh_objective(source)` is an ordinary function you can pass to
`scipy.optimize.minimize`. Labs 06 and 08 show both.

## Layout

```
src/bellgame/   the package (import bellgame as bg)
labs/           step-by-step notebooks
examples/       worked studies (sweeps, two-star network, classical strategies, optimizers, E91)
students/       one folder per contributor (copy students/_template)
tests/          pytest tests: uv run pytest
```
