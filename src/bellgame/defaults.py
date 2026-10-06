"""Every knob you can turn, with a realistic default, its unit, and what it means.

Print it to see what you can study::

    >>> import bellgame as bg
    >>> bg.DEFAULTS["loss_db_per_km"]["value"]
    0.2

Each entry is ``{"value": ..., "unit": ..., "meaning": ..., "group": ...}``. The groups:

* ``fiber`` and ``detector``: on every link, in a network or on its own
* ``pair``: how good a network link's freshly made pair is (``link_model`` picks the physics)
* ``source``: the SPDC photon source, used by ``link_model="fock"`` and by ``bg.link``
* ``photonic``: extras that only a stand-alone photonic link (``bg.link``) has
* ``node``: a network node's quantum memories and swapping hardware
"""

DEFAULTS = {
    # --- fiber ---
    "distance_km": {
        "value": 10.0, "unit": "km", "group": "fiber",
        "meaning": "fiber length between the two ends of the link (the pair is made in the middle)",
    },
    "loss_db_per_km": {
        "value": 0.2, "unit": "dB/km", "group": "fiber",
        "meaning": "fiber attenuation (0.2 dB/km is typical telecom fiber at 1550 nm)",
    },
    # --- detectors ---
    "detector_efficiency": {
        "value": 0.8, "unit": "", "group": "detector",
        "meaning": "chance a detector clicks when a photon arrives (0 to 1)",
    },
    "dark_count_rate_hz": {
        "value": 100.0, "unit": "Hz", "group": "detector",
        "meaning": "false clicks per second from a detector with no light",
    },
    "coincidence_window_ns": {
        "value": 1.0, "unit": "ns", "group": "detector",
        "meaning": "how long a detector listens per attempt; dark-count chance = rate x window",
    },
    # --- pair quality on a network link ---
    "link_model": {
        "value": "analytic", "unit": "", "group": "pair",
        "meaning": "how a link's pair quality is worked out: 'fixed' (just raw_fidelity), 'analytic' "
                   "(raw_fidelity spoiled by dark counts as the fiber gets longer), or 'fock' (full photon "
                   "simulation of an SPDC source, including double pairs)",
    },
    "raw_fidelity": {
        "value": 0.98, "unit": "", "group": "pair",
        "meaning": "fidelity of a freshly made pair before any distance effects (the source's own quality)",
    },
    "raw_errors": {
        "value": [1 / 3, 1 / 3, 1 / 3], "unit": "", "group": "pair",
        "meaning": "how the missing fidelity splits into X, Y, Z errors (must add up to 1)",
    },
    # --- photon source (fock model) ---
    "mean_photon_number": {
        "value": 0.01, "unit": "photons", "group": "source",
        "meaning": "average photons per mode per pulse from the SPDC source (bigger = more pairs, "
                   "but more double pairs that spoil the correlations)",
    },
    "truncation": {
        "value": 2, "unit": "photons", "group": "source",
        "meaning": "most photons we keep track of in one mode (2 captures double pairs)",
    },
    "heralded": {
        "value": False, "unit": "", "group": "source",
        "meaning": "if True, only count pulses where the source really emitted a pair "
                   "(an idealized event-ready source; needed to see the detection loophole)",
    },
    # --- stand-alone photonic link only ---
    "alice_misalignment_deg": {
        "value": [0.0, 0.0, 0.0], "unit": "deg", "group": "photonic",
        "meaning": "polarization twist the fiber applies on Alice's side (3 angles, see "
                   "bg.polarization_rotation)",
    },
    "bob_misalignment_deg": {
        "value": [0.0, 0.0, 0.0], "unit": "deg", "group": "photonic",
        "meaning": "polarization twist the fiber applies on Bob's side (3 angles)",
    },
    "no_click": {
        "value": "discard", "unit": "", "group": "photonic",
        "meaning": "what a player answers when neither detector clicks: 'discard' (throw the "
                   "round away = fair-sampling assumption), 'random', or 'zero'",
    },
    "double_click": {
        "value": "random", "unit": "", "group": "photonic",
        "meaning": "what a player answers when both detectors click: 'random' or 'zero'",
    },
    # --- network node ---
    "memory_size": {
        "value": 10, "unit": "memories", "group": "node",
        "meaning": "number of quantum memories in the node",
    },
    "memory_frequency_hz": {
        "value": 2e4, "unit": "Hz", "group": "node",
        "meaning": "how often a memory can be re-excited (limits how fast links make pairs)",
    },
    "memory_efficiency": {
        "value": 1.0, "unit": "", "group": "node",
        "meaning": "chance a memory successfully emits its photon when asked",
    },
    "coherence_time_ms": {
        "value": 100.0, "unit": "ms", "group": "node",
        "meaning": "how long a memory keeps its qubit: stored pairs decay gradually on this time scale, "
                   "and are thrown away after it (inf = perfect memory)",
    },
    "memory_errors": {
        "value": [0.0, 0.0, 1.0], "unit": "", "group": "node",
        "meaning": "what kind of decay a stored qubit suffers, as X, Y, Z error shares "
                   "([0, 0, 1] = pure dephasing, [1/3, 1/3, 1/3] = depolarizing)",
    },
    "gate_fidelity": {
        "value": 1.0, "unit": "", "group": "node",
        "meaning": "quality of the two-qubit gate a node uses for entanglement swapping",
    },
    "measurement_fidelity": {
        "value": 1.0, "unit": "", "group": "node",
        "meaning": "quality of the measurements a node uses for entanglement swapping",
    },
}

LINK_GROUPS = ("fiber", "detector", "pair", "source")       # a link inside a network
PHOTONIC_GROUPS = ("fiber", "detector", "source", "photonic")  # bg.link, a link on its own


def default_values(*groups):
    """Plain {name: value} dict of the defaults in the given groups.

    Example:
        >>> default_values("fiber")
        {'distance_km': 10.0, 'loss_db_per_km': 0.2}
    """
    out = {}
    for name, entry in DEFAULTS.items():
        if entry["group"] in groups:
            value = entry["value"]
            out[name] = list(value) if isinstance(value, list) else value
    return out


def describe(name):
    """One line: name = value unit  -- meaning."""
    entry = DEFAULTS[name]
    return f"{name} = {entry['value']} {entry['unit']}  -- {entry['meaning']}"
