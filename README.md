# bellgame

The CHSH game over simulated quantum networks, for students learning Python.

Alice and Bob get random questions from a referee and must answer without
talking. Classical players win at most 75% of the rounds; players sharing
entangled photons can win 85.4%. bellgame lets you play that game over
realistic photonic links and quantum networks, find out what breaks it, and
use the same test to build a secret key (E91).

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

## The one idea to hold on to

Every experiment ends in a **probability table** `p(a, b | x, y)`: the chance
of answers a, b for questions x, y. A strategy plus a source of entanglement
makes a table; the game reads the table.

| Source | How to make it |
|---|---|
| a classical team | two functions, `bg.play_classical(alice, bob)` |
| an ideal or noisy qubit pair | `bg.bell_pair(fidelity)` |
| a photonic link (Fock model) | `bg.link(distance_km=..., mean_photon_number=..., ...)` |
| a path through a network | `bg.end_to_end(net, "A1", "B2")` |

`bg.play_chsh(strategy, source)` gives the exact answer; add `rounds=N, seed=...`
to simulate a finite experiment.

## Labs

| Lab | Python skill | Physics idea |
|---|---|---|
| 01 classical game | functions | no classical team beats 75% |
| 02 many rounds | loops, lists | experiments only estimate probabilities |
| 03 quantum strategy | dicts | entanglement wins 85.4% |
| 04 photons as Fock modes | numpy arrays | real sources make 0, 1, 2... pairs |
| 05 parameter studies | sweeps, matplotlib | loss, dark counts, the detection loophole |
| 06 misalignment | scipy.optimize | undoing polarization twists |
| 07 E91 | lists of bits | a secret key from a Bell test |
| 08 networks | dicts, numpy matrices | entanglement swapping, the two-star network |

## The physics model, and its assumptions

* **Links** are modeled in Fock (photon-number) space: an SPDC source in the
  middle (two two-mode squeezed vacua, so multi-pair emission is included),
  fiber loss, polarization misalignment, polarizers, and threshold detectors
  with efficiency and dark counts. This reuses SeQUeNCe's Fock quantum manager.
* **Networks** are hybrid. SeQUeNCe runs the network (routing, Barrett-Kok
  generation, swapping at the hubs) and reports rates and how long qubits waited
  in memory. bellgame takes each link's state from the Fock model, lets stored
  qubits dephase for as long as they waited (`coherence_time_ms`), and joins the
  links with exact entanglement swaps.
* **Assumption:** each link's photon pair is stored in quantum memories when it
  is heralded. The link's state is the Fock model's one-photon-at-each-end state.
* `no_click="discard"` (the default) keeps only rounds where both sides click:
  the fair-sampling assumption. Lab 05 shows what happens without it.

## Using SeQUeNCe and scipy directly

Nothing is hidden. `bg.build(net)` returns SeQUeNCe's `RouterNetTopo` for your
network, and `bg.chsh_objective(source)` is an ordinary function you can pass to
`scipy.optimize.minimize`. Labs 06 and 08 show both.

## Layout

```
src/bellgame/   the package (import bellgame as bg)
labs/           eight guided labs
examples/       longer worked examples (two-star network, sweeps, optimizers, E91)
students/       one folder per student (copy students/_template)
tests/          pytest tests: uv run pytest
```
