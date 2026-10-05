# Plan: `bellgame` — CHSH over simulated quantum networks, for students

## Context

`CHSH-game-introduction/E_91/*/network_module_E91.py` simulates the CHSH game, E91, and a misalignment-correction
optimizer study (Nelder-Mead / COBYQA, after the Brewster paper) on an old SeQUeNCe API. Problems with it:

- three near-duplicate copies of the module, plus one notebook per parameter value
- it breaks on SeQUeNCe 1.2.0 (qutip-4 imports; `EntanglementGenerationA` is now abstract)
- knobs that don't do what they say: memory "fidelity" is metadata only, distance only changes runtime
- a full discrete-event rebuild for every round

**Goal:** a new peer repo `QuantumNetworking/bellgame/` that keeps the scope to **CHSH** (the game, misalignment +
optimizers, E91). It moves the physics to **polarization photons modeled in Fock (photon-number) modes**, so students can
study every key network parameter. It also ships a **two-star network** example (two hub-and-leaf stars joined hub-to-hub), which the advisor wants to build toward.

**Audience:** freshmen learning Python through this project. The default path is `import bellgame as bg` plus plain Python, numpy and matplotlib.
`scipy.optimize.minimize` and SeQUeNCe are reachable, not hidden: the docs and labs show students how to use them directly.

**Distribution:** a GitHub repo with a uv lockfile. `uv sync` installs everything, with `sequence` from PyPI. Each student gets a
`students/<name>/` folder and may extend `src/bellgame/`.

## First action on approval

Write this plan to `QuantumNetworking/bellgame/PLAN.md`. This creates the peer directory; it's not a git repo yet. That file is the
hand-off document for a fresh session. Milestone 0 then scaffolds around it.

## Key findings (so a fresh session doesn't have to re-derive them)

- **Paths:**
  - old repo: `QuantumNetworking/CHSH-game-introduction/` (main code in `E_91/Distance_Aware/network_module_E91.py`; the other copies differ only in channel params)
  - SeQUeNCe 1.2.0 checkout: `QuantumNetworking/SeQUeNCe/` (Python >=3.12, <3.15; qutip 5 + qutip-qip)
- **Sandbox quirk:** Bash sees the project folder as empty unless you use absolute paths. The Read tool works normally.
  `.venv/bin/python` links into `~/.local/share/uv/python/`, which the sandbox must be allowed to read.
- **SeQUeNCe 1.2.0 facts this plan relies on:**
  - `Circuit` has no rx/ry/rz; the ket backend uses `gate_matrix()` and rejects unknown gates
  - `EntanglementGenerationA` is abstract; use `.create(...)` or `BarretKokA`
  - Barrett-Kok treats memory fidelity as metadata only
  - `coherence_time` is a hard expiry, not gradual decoherence
  - Fock support is component-level only (`QuantumManagerDensityFock`, `SPDCSource`, `FockDetector`, `QSDetectorFock*`, `FockBeamSplitter*`,
    Fock loss in `QuantumChannel`); the router stack (`RouterNetTopo`, `RequestApp`, swapping) is qubit-only, hence the hybrid design
  - `RouterNetTopo` accepts a config dict with a `"formalism"` key
  - graph tooling already exists: `sequence/utils/graphs.py` (`build_star`, `build_linear`, `build_ring`, `build_tree`, …, returning networkx graphs) and
    `sequence/utils/nx_converter.py` (networkx → `RouterNetTopo` config; its default template uses `single_heralded`/Bell-diagonal, so we
    supply our own Barrett-Kok + `density_matrix` template)
- **Old-code bugs not to carry over:**
  - `play_round` with `deltas=None` crashes
  - S is taken as the max over sign patterns (biased upward)
  - COBYQA records wrapped angles but simulates unwrapped ones
  - no seeding
- **User decisions:**
  - name `bellgame`, but scope stays CHSH
  - v1 = CHSH + noise, misalignment + optimizers, E91
  - uv environment with `sequence` from PyPI
  - polarization encoded in Fock modes
  - hybrid two-star design
  - `# %%` scripts + jupytext, not .ipynb and not marimo
  - students may use scipy and SeQUeNCe directly, but don't have to
  - Python 3.13 (uv-managed interpreter); spyder-kernels 3.0.* for Spyder 6
  - local git commits per milestone, no remote yet

## Network description: one dict, many ways in

The canonical representation stays a single dict that is a valid `RouterNetTopo` config (see Physics layer 2). Students can **build it
whichever way suits them**, and every route produces the same dict:

- **Connectivity / distance matrix** (recommended teaching path, because it's pure numpy): `bg.from_matrix(names, dist_km)`.
  `dist_km[i, j]` is the fiber length; 0 means no link. It must be symmetric, and the function checks that with a friendly error. All other link
  parameters come from `DEFAULTS` and can be overridden per link with `bg.set_link`. The two-star network is a block matrix: two star
  blocks plus one hub-to-hub entry, which makes a nice numpy exercise.
- **Edge list:** `bg.from_edges([("A1", "HubA", 5), ("HubA", "HubB", 20), ...])`
- **networkx:** `bg.from_networkx(G)`, using edge attribute `length` in km. This reuses SeQUeNCe's `graphs.py` builders (e.g.
  `build_star`) for students who want them.
- **Builders:** `bg.star(...)`, `bg.two_star(...)`, `bg.connect(...)` for quick in-code edits.
- **Going the other way:** `bg.to_matrix(net)` returns (names, distance matrix) for inspection or `plt.imshow`. `bg.parameters(net)` lists all knobs.

A matrix alone can't hold the per-link Fock parameters (μ, efficiencies, dark counts) without parallel matrices. That's why it's an
*input* format rather than the storage format.

## Core idea: everything becomes a probability table

Every experiment reduces to **p(a, b | x, y)**, a numpy array of shape (2, 2, 2, 2): outputs a, b given referee inputs x, y.

- Classical strategies, ideal qubit pairs, Fock photonic links, and multi-hop network paths all *produce* such a table.
- The CHSH game, S, E91 key rate/QBER, and the plots all *consume* it.
- Exact mode uses the table directly: deterministic and instant, so optimizers are fast.
  Sampled mode draws `rounds=N` from it with a seeded rng, reproducing the shot noise of the old study.

This gives students one clear concept to hold onto: a strategy plus a network gives a probability table, which gives a win rate.

## Physics layers

1. **Fock link model** (`fock.py`, `link.py`): one link with a polarization-entangled SPDC source in the middle.
   - **Modes:** A_H, A_V, B_H, B_V, with photon-number truncation `truncation` (default 2, giving an 81-dim density matrix, which captures double pairs).
   - **Source:** type-II SPDC, two two-mode-squeezed vacua (A_H·B_V and A_V·B_H) with `mean_photon_number`.
     Same TMSV amplitude formula as SeQUeNCe's `SPDCSource._generate_tmsv_state` (`sequence/components/light_source.py:128`).
   - **Fiber + detector loss:** a pure-loss channel per mode, using `QuantumManagerDensityFock.add_loss`
     (`sequence/kernel/quantum_manager/fock_density_matrix.py:299`), with transmissivity from `distance_km` and `loss_db_per_km`.
   - **Polarization analyzer and misalignment:** a passive 2-mode unitary on (H, V), exp of the beam-splitter generator built from
     `build_ladder()` (exact for total photon number ≤ truncation). Misalignment and alignment correction use the same 3-angle family.
   - **Detectors:** threshold POVMs with `detector_efficiency` and `dark_count_prob` (rate × coincidence window), the same
     construction as `QSDetectorFockDirect._generate_povms` (`sequence/components/detector.py:353`).
   - **Click → bit policy (a teaching knob):** `no_click="discard" | "random" | "zero"` and `double_click="random"`.
     `discard` is post-selection on coincidences, i.e. the fair-sampling assumption. The other two show the detection loophole
     (violation needs η > 2(√2−1) ≈ 82.8%).
   - **Outputs:** the probability table, herald/coincidence probability, and an **effective 2-qubit state**
     (projected onto one photon per side and normalized) with its fidelity, for use in networks.

2. **Network layer, hybrid** (`network.py`, `paths.py`):
   - **Topology** is a plain dict that *is* a valid SeQUeNCe `RouterNetTopo` config (`nodes`, `qconnections`, `cconnections`,
     `templates`, `"formalism": "density_matrix"`), stored in SeQUeNCe-native units. Each `qconnection` also carries a
     `"photonics"` sub-dict of Fock link parameters, which is stripped before handing the dict to SeQUeNCe.
   - **Rates and timing** come from SeQUeNCe's router stack, run on the same distances, attenuation and efficiencies:
     `RouterNetTopo(config)` + a `RequestApp` from the end node, giving real routing, Barrett-Kok generation, swapping at the hubs,
     wait times and throughput. A small private `RequestApp` subclass records timestamps and wait times in `get_memory`
     (`sequence/app/request_app.py:120`).
   - **State quality** comes from the Fock model. Each link's effective qubit state is composed along the path by an exact
     numpy entanglement swap (Bell measurement + Pauli correction, averaged over outcomes, on 16-dim states), with memory dephasing over the
     SeQUeNCe-reported wait times (SeQUeNCe's `coherence_time` is only a hard-expiry cutoff, `memory.py:307-369`).
     The end-to-end state goes to the qubit analyzer, which produces the probability table.
   - **Stated assumption:** heralded storage of link photons in memories. The README and lab 08 name it explicitly.

3. **Qubit shortcut** (`qubits.py`): `bell_pair(fidelity)`, `werner`, `rx/ry/rz`, `swap`, `dephase`. Used in lab 03 before photons are
   introduced, and internally for path composition.

## Q2 recap: old code disposition

| Old | Fate | New |
|---|---|---|
| `Timeline`, `QuantumChannel`, `ClassicalChannel`, `BSMNode`, `Memory`, Barrett-Kok, routing/swapping | **Reuse SeQUeNCe as-is** | via `RouterNetTopo(config_dict)`; `bg.build(net)` returns the topo object for students to explore |
| `RequestApp.get_memory` | **Repurpose** | one private subclass for timing capture (the only subclass in the package) |
| SeQUeNCe Fock pieces (`QuantumManagerDensityFock`, TMSV formula, detector POVMs, `add_loss`) | **Reuse** | wrapped in `fock.py` |
| `CustomCircuit`, qutip gate funcs, `PlayerNode`, `Manager`, `generate_entanglement`, `pair_protocol`, `getNodeFromName`, `MsgType`/`Player`, shared circuit | **Drop** | numpy gates and analyzers |
| `RefereeProtocol`, `Player/Alice/BobProtocol` | **Replace** | `referee_wins(x, y, a, b)`, strategy dicts, student-written strategy functions |
| `Game` | **Split** | `play_chsh`, `run_e91`, network helpers |
| `compute_chsh_s` + max over sign patterns | **Rewrite** | `correlations(table)`, `chsh_value` with a **fixed** sign pattern (removes the upward bias) |
| Notebook optimizers, `wrap_angles`, `pad_with_last_value` | **Move into package** | `optimize.py`. BayesOpt is dropped (it was commented out). |
| Notebook plotting cells | **Move into package** | `plots.py` |

## Repo layout

```
bellgame/
  pyproject.toml  uv.lock  .python-version (3.13)  jupytext.toml (py:percent)  README.md  CONTRIBUTING.md
  .github/workflows/ci.yml
  src/bellgame/
    __init__.py      flat API: bg.<name>
    defaults.py      DEFAULTS dict: every tunable parameter, realistic value, unit, one-line meaning
    qubits.py        bell_pair, werner, rx/ry/rz, swap, dephase
    fock.py          spdc_state, apply_loss, analyzer_unitary, threshold_povms (SeQUeNCe Fock manager inside)
    link.py          link_table(link_params, strategy) → p(a,b|x,y); link_state(link_params) → (rho, herald_prob)
    network.py       dict builders: two_player_network, add_node, connect, star, two_star; from_matrix, from_edges,
                     from_networkx, to_matrix; set_link, set_memory, set_source, set_detectors;
                     parameters(net) prints every knob; build(net) → RouterNetTopo
    _sequence_adapter.py  strip photonics, run router stack, capture timings (private)
    paths.py         end_to_end(net, "A1", "B2") → {"state", "rate_hz", "wait_ms", "links": [...]}
    strategies.py    optimal_strategy(), classical strategies, angle convention (degrees)
    game.py          referee_wins, table_from_strategy, play_classical, play_chsh(strategy, source, rounds=None, seed=None)
    e91.py           run_e91(source, rounds, seed) → key bits, QBER, S
    optimize.py      chsh_objective(...) → f(angles) usable with scipy.optimize.minimize; find_alignment; compare_optimizers
    plots.py         (below)
  labs/          01_classical_game.py … 08_networks.py   (# %% cells; open in Spyder/VS Code, or as notebooks via jupytext)
  examples/      two_star_network.py, distance_sweep.py, optimizer_comparison.py, e91_key.py
  students/      README.md, _template/starter.py
  tests/
```

**Dependencies:** `sequence==1.2.*`, `numpy`, `scipy`, `matplotlib`.
**Dev group** (installed by default by `uv sync`): `jupyterlab`, `jupytext`, `ipykernel`, `spyder-kernels` (pinned to match the class's Spyder
version; the README shows how to point Spyder at `.venv`), `pytest`, `ruff`.

## Two-star example (`examples/two_star_network.py`)

```python
net = bg.two_star(hub_a="HubA", leaves_a=["A1", "A2", "A3"],
                  hub_b="HubB", leaves_b=["B1", "B2", "B3"],
                  leaf_km=5, hub_km=20)          # leaves only connect to their own hub
# same network from a distance matrix (shown side by side in the example):
# names = ["HubA","A1","A2","A3","HubB","B1","B2","B3"]; dist = np.zeros((8, 8)); fill star blocks + dist[0, 4] = dist[4, 0] = 20
# net = bg.from_matrix(names, dist)
bg.plot_network(net)
path = bg.end_to_end(net, "A1", "B2", seed=1)    # A1→HubA→HubB→B2: 3 links, 2 swaps (SeQUeNCe routing)
print(bg.play_chsh(bg.optimal_strategy(), path))
# sweep hub_km, plot S and rate vs distance, with classical/Tsirelson reference lines
```

## Plotting (Q3)

Every function takes results or tables, accepts an optional `ax=`, returns `ax`, and draws the **classical (75% / S=2)** and **Tsirelson
(85.4% / 2√2)** reference lines.

- `plot_results`: counts per (x,y), colored by win or loss
- `plot_correlations`: 2×2 heatmap of E(x,y)
- `plot_sweep(xs, results, metric=...)`
- `plot_angles(strategy)`: measurement directions on a circle
- `plot_convergence`: best-so-far S vs. time, mean ± std
- `plot_network(net)`
- `plot_photon_numbers(link)`: photon-number distribution, showing what multi-pair emission looks like

## Freshman-friendly rules (Q4)

- **Data and parameters:** inputs and outputs are dicts and lists, so `print()` shows everything.
  Strategies are `{"alice": [0, 45], "bob": [22.5, -22.5]}` in degrees. Classical strategies are student-written functions (`def alice(x): return 0`).
- **Package source** avoids ABCs, metaclasses, `**kwargs` pass-through and decorators. Functions stay short, each with a docstring and a runnable example.
- **Units in parameter names** (`distance_km`, `loss_db_per_km`, `mean_photon_number`, `detector_efficiency`, `dark_count_rate_hz`,
  `coherence_time_ms`, `angle_deg`). Plain-English validation errors. `seed=` on everything random.
- **`bg.parameters(net)`** lists every tunable knob with value, unit and meaning, so students can see what's available to study.
- **Each lab** opens with a "Python skill" and a "Physics idea":
  1. classical game (functions)
  2. many rounds (loops, lists)
  3. ideal quantum strategy (dicts, `plot_angles`)
  4. photons as Fock modes (numpy arrays)
  5. parameter studies (sweeps, matplotlib, detection loophole)
  6. misalignment: call `scipy.optimize.minimize` directly on `bg.chsh_objective`, then `bg.find_alignment`
  7. E91
  8. networks: dict building, two-star, then `bg.build(net)` to explore the SeQUeNCe objects underneath
- **`students/<name>/`**: copied from `_template/`. CONTRIBUTING covers branch-per-student and how to add a package function with a test.

## Milestones

0. **Scaffold:** `uv init --package bellgame` (src layout), `uv add sequence numpy scipy matplotlib` (done: PyPI `sequence` resolved to 1.2.0 on Python 3.13; pyproject written by hand with hatchling), dev group, jupytext.toml, README install steps, `git init`.
1. **Qubit + game core:** qubits.py, strategies.py, game.py, the probability-table machinery. Tests below.
2. **Fock link:** fock.py, link.py, defaults.py. Tests below.
3. **Network:** start with a **spike** to confirm three things:
   - (a) `RouterNetTopo` accepts our dict (with `photonics` stripped) and the `density_matrix` formalism
   - (b) `RequestApp` from A1 to B2 routes through both hubs, and we can capture wait times
   - (c) runtime is acceptable

   Fallback if (b) fails: per-link SeQUeNCe two-node runs for rates, composed by `paths.py`. Then the dict builders,
   `end_to_end`, numpy swap composition.
4. **Plots.**
5. **Optimize:** `chsh_objective`, `find_alignment`, `compare_optimizers` (Nelder-Mead, COBYQA).
6. **E91:** bases A {0, 45, 90}°, B {22.5, 67.5, 112.5}°, using the old `E91_CHSH_MAP` with fixed signs.
7. **Labs 01–08, examples (including two-star), students template, CONTRIBUTING.**
8. **CI:** `uv sync`, `uv run pytest`, and executing every `labs/*.py` and `examples/*.py` as a smoke test.

## Verification

- **Analytic checks (pytest):**
  - ideal Φ+ with the optimal strategy gives win = cos²(π/8) ≈ 0.8536 and S = 2√2
  - the best classical strategy gives 0.75
  - a Werner state gives S = 2√2·(4F−1)/3
  - a swap of two Werner links gives the known composed fidelity
- **Fock checks:**
  - μ→0, no loss, perfect detectors, `discard` gives S → 2√2
  - S decreases monotonically with μ, distance (with dark counts), and misalignment
  - with `no_click="zero"`, violation disappears below η ≈ 0.828
  - truncation 2 vs 3 agree within 1e-3 at μ = 0.01
- **Network checks:** `from_matrix`, `from_edges`, `from_networkx` and `two_star` all build the identical dict for the two-star network, and
  `to_matrix` round-trips it. Rate falls with `hub_km`. The two-star A1→B2 path runs end to end and its S is below a single link's.
- **Optimizer:** random misalignment, exact mode recovers S > 2.8 in seconds. Sampled mode (`rounds=2500`) reproduces the old Nelder-Mead vs. COBYQA convergence plot qualitatively.
- **Fresh clone:** `uv sync && uv run python examples/two_star_network.py` produces the plot. Every lab runs top to bottom (`uv run python labs/0X_*.py`) and opens in Jupyter via jupytext.
