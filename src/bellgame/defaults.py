"""Every knob you can turn, with a realistic default, its unit, and what it means.

Print it to see what you can study::

    >>> import bellgame as bg
    >>> bg.DEFAULTS["loss_db_per_km"]["value"]
    0.2

Each entry is ``{"value": ..., "unit": ..., "meaning": ..., "group": ...}``. Network
parameters also say which values are allowed: a ``range`` such as ``"(0, 1]"``, a tuple
of ``choices``, or a ``total`` that a list of shares must add up to. The groups:

* ``fiber`` and ``detector``: on every link, in a network or on its own
* ``pair``: how good a network link's freshly made pair is (``link_model`` picks the physics)
* ``source``: the SPDC photon source, used by ``link_model="fock"`` and by ``bg.link``
* ``photonic``: extras that only a stand-alone photonic link (``bg.link``) has
* ``node``: a network node's quantum memories and swapping hardware
"""

import numpy as np

DEFAULTS = {
    # --- fiber ---
    "distance_km": {
        "value": 10.0, "unit": "km", "group": "fiber", "range": "(0, inf)",
        "meaning": "fiber length between the two ends of the link (the pair is made in the middle)",
    },
    "loss_db_per_km": {
        "value": 0.2, "unit": "dB/km", "group": "fiber", "range": "[0, inf)",
        "meaning": "fiber attenuation (0.2 dB/km is typical telecom fiber at 1550 nm)",
    },
    # --- detectors ---
    "detector_efficiency": {
        "value": 0.8, "unit": "", "group": "detector", "range": "[0, 1]",
        "meaning": "chance a detector clicks when a photon arrives (0 to 1)",
    },
    "dark_count_rate_hz": {
        "value": 100.0, "unit": "Hz", "group": "detector", "range": "[0, inf)",
        "meaning": "false clicks per second from a detector with no light",
    },
    "coincidence_window_ns": {
        "value": 1.0, "unit": "ns", "group": "detector", "range": "[0, inf)",
        "meaning": "how long a detector listens per attempt; dark-count chance = rate x window",
    },
    # --- pair quality on a network link ---
    "link_model": {
        "value": "analytic", "unit": "", "group": "pair", "choices": ("fixed", "analytic", "fock"),
        "meaning": "how a link's pair quality is worked out: 'fixed' (just raw_fidelity), 'analytic' "
                   "(raw_fidelity spoiled by dark counts as the fiber gets longer), or 'fock' (full photon "
                   "simulation of an SPDC source, including double pairs)",
    },
    "raw_fidelity": {
        "value": 0.98, "unit": "", "group": "pair", "range": "[0.25, 1]",
        "meaning": "fidelity of a freshly made pair before any distance effects (the source's own quality)",
    },
    "raw_errors": {
        "value": [1 / 3, 1 / 3, 1 / 3], "unit": "", "group": "pair", "range": "[0, 1]", "total": 1,
        "meaning": "how the missing fidelity splits into X, Y, Z errors (must add up to 1)",
    },
    # --- photon source (fock model) ---
    "mean_photon_number": {
        "value": 0.01, "unit": "photons", "group": "source", "range": "(0, inf)",
        "meaning": "average photons per mode per pulse from the SPDC source (bigger = more pairs, "
                   "but more double pairs that spoil the correlations)",
    },
    "truncation": {
        "value": 2, "unit": "photons", "group": "source", "range": "[1, inf)",
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
        "value": 10, "unit": "memories", "group": "node", "range": "[1, inf)",
        "meaning": "number of quantum memories in the node",
    },
    "memory_frequency_hz": {
        "value": 2e4, "unit": "Hz", "group": "node", "range": "(0, inf)",
        "meaning": "how often a memory can be re-excited (limits how fast links make pairs)",
    },
    "memory_efficiency": {
        "value": 1.0, "unit": "", "group": "node", "range": "[0, 1]",
        "meaning": "chance a memory successfully emits its photon when asked",
    },
    "coherence_time_ms": {
        "value": 100.0, "unit": "ms", "group": "node", "range": "(0, inf]",
        "meaning": "how long a memory keeps its qubit: stored pairs decay gradually on this time scale, "
                   "and are thrown away after it (inf = perfect memory)",
    },
    "memory_errors": {
        "value": [0.0, 0.0, 1.0], "unit": "", "group": "node", "range": "[0, 1]", "total": 1,
        "meaning": "what kind of decay a stored qubit suffers, as X, Y, Z error shares "
                   "([0, 0, 1] = pure dephasing, [1/3, 1/3, 1/3] = depolarizing)",
    },
    "gate_fidelity": {
        "value": 1.0, "unit": "", "group": "node", "range": "[0, 1]",
        "meaning": "quality of the two-qubit gate a node uses for entanglement swapping",
    },
    "measurement_fidelity": {
        "value": 1.0, "unit": "", "group": "node", "range": "[0, 1]",
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


def clean_value(name, value, where=None):
    """``value`` as bellgame stores parameter ``name``, or a ValueError saying what is wrong.

    Numbers become floats (ints where the default is an int) and lists become new lists
    of floats. NaN is never allowed; ``range``, ``choices`` and ``total`` are checked when
    the entry has them. ``where`` (e.g. ``"node Alice"``) starts the error message.

    Example:
        >>> clean_value("memory_size", 6.0)
        6
        >>> clean_value("detector_efficiency", float("nan"), where="link A -- B")
        Traceback (most recent call last):
        ...
        ValueError: link A -- B: detector_efficiency is NaN
    """
    entry, default = DEFAULTS[name], DEFAULTS[name]["value"]
    try:
        if isinstance(default, list):
            value = _as_list(name, value, len(default))
        elif isinstance(default, bool):
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be True or False, got {value!r}")
            value = bool(value)
        elif isinstance(default, str):
            if "choices" in entry and value not in entry["choices"]:
                raise ValueError(f"{name} must be one of {entry['choices']}, got {value!r}")
        else:
            value = _as_number(name, value, type(default))
        for x in value if isinstance(value, list) else [value]:
            if "range" in entry and not _in_range(x, entry["range"]):
                raise ValueError(f"{name} must be in {entry['range']}, got {value}")
        if "total" in entry and not np.isclose(sum(value), entry["total"]):
            raise ValueError(f"{name} must add up to {entry['total']}, got {value}")
    except ValueError as err:
        raise ValueError(f"{where}: {err}" if where else str(err)) from None
    return value


def _as_number(name, value, kind):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        x = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}") from None
    if np.isnan(x):
        raise ValueError(f"{name} is NaN")
    if kind is int:
        if not x.is_integer():
            raise ValueError(f"{name} must be a whole number, got {value}")
        return int(x)
    return x


def _as_list(name, value, length):
    if isinstance(value, (str, dict)) or not hasattr(value, "__len__") or len(value) != length:
        raise ValueError(f"{name} must be a list of {length} numbers, got {value!r}")
    return [_as_number(name, x, float) for x in value]


def _in_range(x, interval):
    """Is ``x`` inside ``interval``, written like ``"(0, inf]"``?"""
    low, high = (float(end) for end in interval[1:-1].split(","))
    above = low <= x if interval[0] == "[" else low < x
    below = x <= high if interval[-1] == "]" else x < high
    return above and below


def describe(name):
    """One line: name = value unit  -- meaning."""
    entry = DEFAULTS[name]
    return f"{name} = {entry['value']} {entry['unit']}  -- {entry['meaning']}"
