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
from .defaults import PHOTONIC_GROUPS, check, default_values, describe

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
            raise ValueError(f"'{name}' is not a link parameter. Link parameters are: {', '.join(sorted(params))}")
        params[name] = check(name, value, where="link")
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
        check(name, params[name], where="link")


def dark_count_prob(params):
    """Chance of at least one dark count in one detection window.

    Dark counts come at random times at ``dark_count_rate_hz`` (a Poisson process, as in
    SeQUeNCe's detectors), so the chance is ``1 - exp(-rate * window)``: about
    ``rate * window`` when that is small, and never more than 1.

    Example:
        >>> round(dark_count_prob(link(dark_count_rate_hz=1000, coincidence_window_ns=2)), 9)
        2e-06
    """
    return float(-np.expm1(-params["dark_count_rate_hz"] * params["coincidence_window_ns"] * 1e-9))


def arm_transmissivity(params):
    """Fraction of photons surviving the fiber from the source (in the middle) to one end."""
    return fock.fiber_transmissivity(params["distance_km"] / 2, params["loss_db_per_km"])


def _state_key(params):
    return (
        float(params["mean_photon_number"]),
        int(params["truncation"]),
        bool(params["heralded"]),
        float(params["distance_km"]),
        float(params["loss_db_per_km"]),
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


# one player's click pattern: (outcome-0 detector clicked, outcome-1 detector clicked)
_CLICK_PATTERNS = list(itertools.product((0, 1), repeat=2))


def click_probabilities(params, alice, bob):
    """p[cA0, cA1, cB0, cB1]: chance of each click pattern when Alice and Bob measure at these settings.

    ``alice`` and ``bob`` are ``(angle_deg, correction_deg or None)``. ``cA0`` is 1 when
    Alice's outcome-0 detector clicks, and so on.
    """
    t = int(params["truncation"])
    rho = arriving_state(params)
    u = np.kron(fock.analyzer_unitary(*alice, t), fock.analyzer_unitary(*bob, t))
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


def link_outcome_probabilities(params, alice, bob):
    """p[a, b] on this link when Alice and Bob measure at these settings, ``(angle_deg, correction_deg or None)``.

    The players turn their clicks into bits by the link's ``no_click`` and ``double_click``
    rules. If every round is discarded (``no_click="discard"`` and nobody clicks), there is
    no data and the result is NaN. ``bg.outcome_probabilities`` calls this for links.
    """
    clicks = click_probabilities(params, alice, bob)
    p = np.zeros((2, 2))
    kept = 0.0
    for ca, cb in itertools.product(_CLICK_PATTERNS, repeat=2):
        bits_a, bits_b = _bit_probabilities(*ca, params), _bit_probabilities(*cb, params)
        if bits_a is None or bits_b is None:
            continue
        kept += clicks[ca + cb]
        p += clicks[ca + cb] * np.outer(bits_a, bits_b)
    return p / kept if kept > 0 else np.full((2, 2), np.nan)


def coincidence_prob(params):
    """Chance, per pulse of the source, that both players see a click.

    A click from either of a player's two detectors counts, dark counts and double pairs
    included. It is exact, not sampled, and it doesn't depend on the measurement angles.
    Multiply by the source's pulse rate for coincidences per second. With
    ``no_click="discard"`` it is also the fraction of pulses that become rounds: ``N``
    rounds take about ``N / coincidence_prob`` pulses.

    Example:
        >>> round(coincidence_prob(link(distance_km=10)), 4)
        0.0081
    """
    # Whether a side clicks at all depends only on how many photons reach it; the angle
    # only picks which of its two detectors fires. So any angles give the same answer.
    clicks = click_probabilities(params, (0.0, None), (0.0, None))
    both = [clicks[ca + cb] for ca, cb in itertools.product(_CLICK_PATTERNS, repeat=2) if any(ca) and any(cb)]
    return float(sum(both))


def photon_numbers(params):
    """p[nA, nB]: chance of nA photons reaching Alice and nB reaching Bob (before detection)."""
    params = photonic(params)
    return fock.photon_number_distribution(arriving_state(params), int(params["truncation"]))


def describe_link(params):
    """Print every link parameter with its unit and meaning."""
    for name, value in params.items():
        print("  " + describe(name, value))


# Measuring ZZ, XX and YY on the pair: each player's (angle_deg, correction_deg). YY uses a
# correction wave plate that turns circular polarization onto H/V (the sign flip is the same
# for both players, so it cancels).
_PAULI_SETTINGS = {
    "zz": (0.0, None),
    "xx": (45.0, None),
    "yy": (0.0, (45.0, 0.0, 0.0)),
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
    # TODO (if the PI wants twists in Fock networks): network links have no twist parameter, so a
    # twisted Fock link can only be refused here. Four Bell weights can't hold a twist, but it could
    # travel alongside them: weights from the untwisted link, then bg.misalign on the delivered pair.
    # That is exact for one link (loss and click heralding don't depend on polarization; checked
    # against the full Fock twist), but only approximate across swaps or with memory decay, which
    # don't commute with the twist. Adding it means a twist on network links that run_network applies.
    for key in ("alice_misalignment_deg", "bob_misalignment_deg"):
        if np.any(np.asarray(full[key]) != 0):
            raise ValueError(
                f"fock_pair_weights needs an untwisted link, got {key}={full[key]}: four Bell-state "
                "weights can't describe a twisted pair. Leave the twist out here and add it to the "
                "delivered pairs with bg.misalign or bg.misaligned_source."
            )
    full["no_click"] = "discard"
    t = {}
    for name, setting in _PAULI_SETTINGS.items():
        p = link_outcome_probabilities(full, setting, setting)
        t[name] = float(p[0, 0] - p[0, 1] - p[1, 0] + p[1, 1])  # correlation: +1 same bits, -1 different
    weights = (
        np.array(
            [
                1 + t["xx"] - t["yy"] + t["zz"],  # Phi+
                1 - t["xx"] + t["yy"] + t["zz"],  # Phi-
                1 + t["xx"] + t["yy"] - t["zz"],  # Psi+
                1 - t["xx"] - t["yy"] - t["zz"],  # Psi-
            ]
        )
        / 4
    )
    weights = np.clip(weights, 0, None)
    return (weights / weights.sum()).tolist(), coincidence_prob(full)
