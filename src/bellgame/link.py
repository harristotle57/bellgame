"""One photonic link: an SPDC source in the middle, fiber to each side, detectors at the ends.

A link is a plain dict of parameters (see ``bg.DEFAULTS``)::

    >>> import bellgame as bg
    >>> my_link = bg.link(distance_km=20, mean_photon_number=0.05)
    >>> my_link["distance_km"]
    20

Put it into ``bg.play_chsh(strategy, my_link)`` like any other source.

Example:
    >>> r = bg.play_chsh(bg.optimal_strategy(), bg.link(distance_km=0))
    >>> r["S"] > 2.7
    True
"""

import numpy as np

from . import fock
from .defaults import DEFAULTS, default_values
from .strategies import check_strategy

NO_CLICK_POLICIES = ("discard", "random", "zero")
DOUBLE_CLICK_POLICIES = ("random", "zero")

_STATE_CACHE = {}


def link(**changes):
    """A link dict with the default parameters, plus any you change.

    Example:
        >>> link(detector_efficiency=0.9)["detector_efficiency"]
        0.9
    """
    params = default_values("link")
    for name, value in changes.items():
        if name not in params:
            raise ValueError(f"'{name}' is not a link parameter. Link parameters are: "
                             f"{', '.join(sorted(params))}")
        params[name] = value
    check_link(params)
    return params


def check_link(params):
    """Raise a friendly error if a link dict has a bad value."""
    for name in default_values("link"):
        if name not in params:
            raise ValueError(f"link is missing '{name}' (make links with bg.link(...))")
    if params["no_click"] not in NO_CLICK_POLICIES:
        raise ValueError(f"no_click must be one of {NO_CLICK_POLICIES}, got {params['no_click']!r}")
    if params["double_click"] not in DOUBLE_CLICK_POLICIES:
        raise ValueError(f"double_click must be one of {DOUBLE_CLICK_POLICIES}, "
                         f"got {params['double_click']!r}")
    if int(params["truncation"]) < 1:
        raise ValueError("truncation must be at least 1")
    for key in ("alice_misalignment_deg", "bob_misalignment_deg"):
        if len(params[key]) != 3:
            raise ValueError(f"{key} needs three angles in degrees")


def dark_count_prob(params):
    """Chance of a dark count in one detection window.

    Example:
        >>> round(dark_count_prob(link(dark_count_rate_hz=1000, coincidence_window_ns=2)), 9)
        2e-06
    """
    return params["dark_count_rate_hz"] * params["coincidence_window_ns"] * 1e-9


def arm_transmissivity(params):
    """Fraction of photons surviving the fiber from the source (in the middle) to one end."""
    return fock.fiber_transmissivity(params["distance_km"] / 2, params["loss_db_per_km"])


def _state_key(params):
    return (
        float(params["mean_photon_number"]), int(params["truncation"]), bool(params["heralded"]),
        float(params["distance_km"]), float(params["loss_db_per_km"]),
        tuple(float(a) for a in params["alice_misalignment_deg"]),
        tuple(float(a) for a in params["bob_misalignment_deg"]),
    )


def arriving_state(params):
    """Four-mode density matrix of the light reaching the two detectors (or memories).

    Source, then fiber loss, then fiber misalignment. Results are cached, so
    calling this again with the same link is instant.
    """
    check_link(params)
    key = _state_key(params)
    if key in _STATE_CACHE:
        return _STATE_CACHE[key]
    t = int(params["truncation"])
    rho = fock.spdc_state(params["mean_photon_number"], t)
    if params["heralded"]:
        rho = rho.copy()
        rho[0, :] = 0  # index 0 is |0000>: no pair emitted
        rho[:, 0] = 0
        rho = rho / np.trace(rho).real
    rho = fock.apply_loss(rho, arm_transmissivity(params), modes=[0, 1, 2, 3], truncation=t)
    ua = fock.polarization_unitary(params["alice_misalignment_deg"], t)
    ub = fock.polarization_unitary(params["bob_misalignment_deg"], t)
    u = np.kron(ua, ub)
    rho = u @ rho @ u.conj().T
    if len(_STATE_CACHE) > 256:
        _STATE_CACHE.clear()
    _STATE_CACHE[key] = rho
    return rho


def click_probabilities(params, strategy, x, y):
    """p[cA0, cA1, cB0, cB1]: chance of each click pattern for questions x, y.

    ``cA0`` is 1 when Alice's outcome-0 detector clicks, and so on.
    """
    t = int(params["truncation"])
    rho = arriving_state(params)
    ua = fock.analyzer_unitary(strategy["alice"][x], strategy.get("alice_correction", (0, 0, 0)), t)
    ub = fock.analyzer_unitary(strategy["bob"][y], strategy.get("bob_correction", (0, 0, 0)), t)
    u = np.kron(ua, ub)
    d = t + 1
    photons = np.real(np.einsum("ij,jk,ik->i", u, rho, u.conj())).reshape(d, d, d, d)
    no_click = fock.threshold_povm_diagonal(params["detector_efficiency"], dark_count_prob(params), t)
    w = np.array([no_click, 1 - no_click])  # w[click, n]
    return np.einsum("abcd,ia,jb,kc,ld->ijkl", photons, w, w, w, w)


def _bit_probabilities(c0, c1, params):
    """[p(bit=0), p(bit=1)] for one player's click pattern, or None to discard the round."""
    if c0 and not c1:
        return [1.0, 0.0]
    if c1 and not c0:
        return [0.0, 1.0]
    if c0 and c1:
        return [0.5, 0.5] if params["double_click"] == "random" else [1.0, 0.0]
    if params["no_click"] == "discard":
        return None
    return [0.5, 0.5] if params["no_click"] == "random" else [1.0, 0.0]


def link_outcomes(params, strategy):
    """(table, coincidence_prob): the CHSH table for this link, and how often both sides click.

    ``coincidence_prob`` is per pulse, averaged over the four question pairs.
    """
    check_strategy(strategy)
    table = np.zeros((2, 2, 2, 2))
    coincidence = 0.0
    for x in (0, 1):
        for y in (0, 1):
            clicks = click_probabilities(params, strategy, x, y)
            kept = 0.0
            for ca0 in (0, 1):
                for ca1 in (0, 1):
                    bits_a = _bit_probabilities(ca0, ca1, params)
                    for cb0 in (0, 1):
                        for cb1 in (0, 1):
                            p = clicks[ca0, ca1, cb0, cb1]
                            if (ca0 or ca1) and (cb0 or cb1):
                                coincidence += p / 4
                            bits_b = _bit_probabilities(cb0, cb1, params)
                            if bits_a is None or bits_b is None:
                                continue
                            kept += p
                            table[:, :, x, y] += p * np.outer(bits_a, bits_b)
            if kept > 0:
                table[:, :, x, y] /= kept
            else:
                table[:, :, x, y] = 0.25
    return table, float(coincidence)


def link_table(params, strategy):
    """p(a, b | x, y) for a quantum strategy played over this photonic link.

    Example:
        >>> import bellgame as bg
        >>> t = link_table(link(), bg.optimal_strategy())
        >>> round(bg.chsh_value(t), 2)
        2.8
    """
    return link_outcomes(params, strategy)[0]


def link_state(params):
    """(state, probability): the link's effective two-qubit state, for use in networks.

    We keep only the pulses that leave exactly one photon at each end (this is
    what a heralded quantum memory would store), and write that photon pair as a
    4x4 qubit density matrix. ``probability`` is the chance per pulse.

    Example:
        >>> import bellgame as bg
        >>> state, p = link_state(link(distance_km=50, mean_photon_number=0.05))
        >>> round(bg.fidelity(state), 3)
        0.968
    """
    t = int(params["truncation"])
    return fock.one_photon_each_state(arriving_state(params), t)


def photon_numbers(params):
    """p[nA, nB]: chance of nA photons reaching Alice and nB reaching Bob (before detection)."""
    return fock.photon_number_distribution(arriving_state(params), int(params["truncation"]))


def describe_link(params):
    """Print every link parameter with its unit and meaning."""
    for name, value in params.items():
        entry = DEFAULTS.get(name, {"unit": "", "meaning": ""})
        print(f"  {name:24s} = {value!s:18s} {entry['unit']:8s} {entry['meaning']}")
