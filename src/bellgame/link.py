"""One photonic link: an SPDC source in the middle, fiber to each side, detectors at the ends.

A link is a plain dict of parameters (see ``bg.DEFAULTS``)::

    >>> import bellgame as bg
    >>> my_link = bg.link(distance_km=20, mean_photon_number=0.05)
    >>> my_link["distance_km"]
    20.0

Put it into ``bg.play(strategy, my_link)`` like any other source.

Example:
    >>> r = bg.play(bg.optimal_strategy(), bg.link(distance_km=0))
    >>> r["S"] > 2.7
    True
"""

import itertools

import numpy as np

from . import fock
from .defaults import DEFAULTS, PHOTONIC_GROUPS, clean_value, default_values
from .strategies import check_strategy

_STATE_CACHE = {}


def link(**changes):
    """A link dict with the default parameters, plus any you change.

    Example:
        >>> link(detector_efficiency=0.9)["detector_efficiency"]
        0.9
    """
    params = default_values(*PHOTONIC_GROUPS)
    for name, value in changes.items():
        if name not in params:
            raise ValueError(f"'{name}' is not a link parameter. Link parameters are: "
                             f"{', '.join(sorted(params))}")
        params[name] = clean_value(name, value, where="link")
    return params


def photonic(params):
    """A full photonic link dict from any link-like dict (e.g. a network link), filling gaps with defaults.

    Example:
        >>> photonic({"distance_km": 3})["no_click"]
        'discard'
    """
    full = default_values(*PHOTONIC_GROUPS)
    full.update({k: v for k, v in params.items() if k in full})
    return full


def check_link(params):
    """Raise a friendly error if a link dict is missing a parameter or has a bad value.

    Each value is checked against its entry in ``bg.DEFAULTS`` (type, range, choices).
    """
    for name in default_values(*PHOTONIC_GROUPS):
        if name not in params:
            raise ValueError(f"link is missing '{name}' (make links with bg.link(...))")
        clean_value(name, params[name], where="link")


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
    rho.flags.writeable = False  # shared by every caller with the same link: nobody may change it
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
    ua = fock.analyzer_unitary(strategy["alice"][x], strategy.get("alice_correction"), t)
    ub = fock.analyzer_unitary(strategy["bob"][y], strategy.get("bob_correction"), t)
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

    ``coincidence_prob`` is the chance, per pulse of the source, that both players see a
    click (from either of their two detectors), dark counts and double pairs included.
    It is exact, not sampled, and it doesn't depend on the measurement angles. Multiply
    by the source's pulse rate for coincidences per second. With ``no_click="discard"``
    it is also the fraction of pulses that become rounds: ``N`` rounds take about
    ``N / coincidence_prob`` pulses.

    A question pair whose rounds are all discarded has NaN in the table: there is no
    data for it.
    """
    check_strategy(strategy)
    # one player's click pattern: (outcome-0 detector clicked, outcome-1 detector clicked)
    patterns = list(itertools.product((0, 1), repeat=2))
    bits = {c: _bit_probabilities(*c, params) for c in patterns}  # None: the round is discarded
    table = np.zeros((2, 2, 2, 2))
    for x, y in itertools.product((0, 1), repeat=2):
        clicks = click_probabilities(params, strategy, x, y)
        kept = 0.0
        for ca, cb in itertools.product(patterns, repeat=2):
            if bits[ca] is None or bits[cb] is None:
                continue
            kept += clicks[ca + cb]
            table[:, :, x, y] += clicks[ca + cb] * np.outer(bits[ca], bits[cb])
        table[:, :, x, y] = table[:, :, x, y] / kept if kept > 0 else np.nan
    # Whether a side clicks at all depends only on how many photons reach it; the angle
    # only picks which of its two detectors fires. So any question pair gives the same
    # coincidence probability, and the last one's clicks will do.
    coincidence = sum(clicks[ca + cb] for ca, cb in itertools.product(patterns, repeat=2) if any(ca) and any(cb))
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


def photon_numbers(params):
    """p[nA, nB]: chance of nA photons reaching Alice and nB reaching Bob (before detection)."""
    params = photonic(params)
    return fock.photon_number_distribution(arriving_state(params), int(params["truncation"]))


def describe_link(params):
    """Print every link parameter with its unit and meaning."""
    for name, value in params.items():
        entry = DEFAULTS.get(name, {"unit": "", "meaning": ""})
        print(f"  {name:24s} = {value!s:18s} {entry['unit']:8s} {entry['meaning']}")


# Measuring ZZ, XX and YY on the pair. YY uses a correction wave plate that turns
# circular polarization onto H/V (the sign flip is the same for both players, so it cancels).
_PAULI_SETTINGS = {
    "zz": {"alice": [0, 0], "bob": [0, 0]},
    "xx": {"alice": [45, 45], "bob": [45, 45]},
    "yy": {"alice": [0, 0], "bob": [0, 0], "alice_correction": [45, 0, 0], "bob_correction": [45, 0, 0]},
}


def fock_pair_weights(params):
    """(weights, coincidence_prob): the Bell-state make-up of the pairs this link delivers.

    This is how the Fock model feeds a network. We look only at pulses where
    both ends click (what heralds a stored pair), measure the ZZ, XX and YY
    correlations, and turn them into weights ``[Phi+, Phi-, Psi+, Psi-]`` (the
    first one is the fidelity). ``params`` can be a network link (the photonic
    extras are filled in with no misalignment).

    A link with a polarization twist is refused: four Bell-state weights (how SeQUeNCe
    stores a pair) can't describe a twisted pair. Twist the delivered pairs afterwards
    instead, with ``bg.misalign`` or ``bg.misaligned_source``.

    Example:
        >>> w, p = fock_pair_weights(link(distance_km=0))
        >>> round(w[0], 3)  # not 1: sometimes the source makes two pairs at once
        0.995
    """
    full = photonic(params)
    for key in ("alice_misalignment_deg", "bob_misalignment_deg"):
        if np.any(np.asarray(full[key]) != 0):
            raise ValueError(f"fock_pair_weights needs an untwisted link, got {key}={full[key]}: four Bell-state "
                             "weights can't describe a twisted pair. Leave the twist out here and add it to the "
                             "delivered pairs with bg.misalign or bg.misaligned_source.")
    full["no_click"] = "discard"
    t = {}
    for name, strategy in _PAULI_SETTINGS.items():
        table, coincidence = link_outcomes(full, strategy)
        signs = np.array([[1, -1], [-1, 1]])
        t[name] = float(np.sum(signs * table[:, :, 0, 0]))
    weights = np.array([
        1 + t["xx"] - t["yy"] + t["zz"],  # Phi+
        1 - t["xx"] + t["yy"] + t["zz"],  # Phi-
        1 + t["xx"] + t["yy"] - t["zz"],  # Psi+
        1 - t["xx"] - t["yy"] - t["zz"],  # Psi-
    ]) / 4
    weights = np.clip(weights, 0, None)
    return (weights / weights.sum()).tolist(), coincidence
