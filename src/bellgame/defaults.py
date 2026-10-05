"""Every knob you can turn, with a realistic default, its unit, and what it means.

Print it to see what you can study::

    >>> import bellgame as bg
    >>> bg.DEFAULTS["loss_db_per_km"]["value"]
    0.2

Each entry is ``{"value": ..., "unit": ..., "meaning": ..., "group": ...}``.
The group says where the knob lives: on a ``link`` (photon source, fiber and
detectors), or on a network ``node`` (quantum memory).
"""

DEFAULTS = {
    # --- link: photon source ---
    "mean_photon_number": {
        "value": 0.01, "unit": "photons", "group": "link",
        "meaning": "average photons per mode per pulse from the SPDC source (bigger = more pairs, "
                   "but more double pairs that spoil the correlations)",
    },
    "truncation": {
        "value": 2, "unit": "photons", "group": "link",
        "meaning": "most photons we keep track of in one mode (2 captures double pairs)",
    },
    "heralded": {
        "value": False, "unit": "", "group": "link",
        "meaning": "if True, only count pulses where the source really emitted a pair "
                   "(an idealized event-ready source; needed to see the detection loophole)",
    },
    # --- link: fiber ---
    "distance_km": {
        "value": 10.0, "unit": "km", "group": "link",
        "meaning": "fiber length between the two ends of the link (source sits in the middle)",
    },
    "loss_db_per_km": {
        "value": 0.2, "unit": "dB/km", "group": "link",
        "meaning": "fiber attenuation (0.2 dB/km is typical telecom fiber at 1550 nm)",
    },
    "alice_misalignment_deg": {
        "value": [0.0, 0.0, 0.0], "unit": "deg", "group": "link",
        "meaning": "polarization twist the fiber applies on Alice's side (3 angles, see "
                   "bg.polarization_rotation)",
    },
    "bob_misalignment_deg": {
        "value": [0.0, 0.0, 0.0], "unit": "deg", "group": "link",
        "meaning": "polarization twist the fiber applies on Bob's side (3 angles)",
    },
    # --- link: detectors ---
    "detector_efficiency": {
        "value": 0.8, "unit": "", "group": "link",
        "meaning": "chance a detector clicks when a photon arrives (0 to 1)",
    },
    "dark_count_rate_hz": {
        "value": 100.0, "unit": "Hz", "group": "link",
        "meaning": "false clicks per second from a detector with no light",
    },
    "coincidence_window_ns": {
        "value": 1.0, "unit": "ns", "group": "link",
        "meaning": "how long a detector listens per pulse; dark-count chance = rate x window",
    },
    "no_click": {
        "value": "discard", "unit": "", "group": "link",
        "meaning": "what a player answers when neither detector clicks: 'discard' (throw the "
                   "round away = fair-sampling assumption), 'random', or 'zero'",
    },
    "double_click": {
        "value": "random", "unit": "", "group": "link",
        "meaning": "what a player answers when both detectors click: 'random' or 'zero'",
    },
    # --- node: quantum memory ---
    "memory_efficiency": {
        "value": 1.0, "unit": "", "group": "node",
        "meaning": "chance a memory successfully emits its photon when asked",
    },
    "memory_frequency_hz": {
        "value": 2e3, "unit": "Hz", "group": "node",
        "meaning": "how often a memory can be re-excited (limits the attempt rate)",
    },
    "coherence_time_ms": {
        "value": 100.0, "unit": "ms", "group": "node",
        "meaning": "how long a memory keeps its phase; stored qubits dephase as exp(-t / T)",
    },
    "memory_size": {
        "value": 10, "unit": "memories", "group": "node",
        "meaning": "number of quantum memories in each router node",
    },
}


def default_values(group):
    """Plain {name: value} dict of the defaults in one group ('link' or 'node').

    Example:
        >>> default_values("link")["distance_km"]
        10.0
    """
    out = {}
    for name, entry in DEFAULTS.items():
        if entry["group"] == group:
            value = entry["value"]
            out[name] = list(value) if isinstance(value, list) else value
    return out


def describe(name):
    """One line: name = value unit  -- meaning."""
    entry = DEFAULTS[name]
    return f"{name} = {entry['value']} {entry['unit']}  -- {entry['meaning']}"
