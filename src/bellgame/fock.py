"""Polarization photons in Fock (photon-number) modes.

A link has four optical modes, always in this order::

    0: A_H  (Alice, horizontal)    1: A_V  (Alice, vertical)
    2: B_H  (Bob, horizontal)      3: B_V  (Bob, vertical)

Each mode holds 0, 1, ..., ``truncation`` photons, so a mode has
``truncation + 1`` levels and the four-mode density matrix is
``(truncation + 1) ** 4`` square (81 x 81 for the default truncation of 2).

The pieces here reuse SeQUeNCe's Fock quantum manager
(``sequence.kernel.quantum_manager.QuantumManagerDensityFock``): its ladder
operators, its photon-loss channel, and the same TMSV amplitudes as
``SPDCSource._generate_tmsv_state``.

Example:
    >>> import bellgame.fock as fock
    >>> rho = fock.spdc_state(mean_photon_number=0.01, truncation=2)
    >>> rho.shape
    (81, 81)
"""

import numpy as np
from scipy.linalg import expm
from sequence.kernel.quantum_manager import QuantumManagerDensityFock

MODES = ["A_H", "A_V", "B_H", "B_V"]


def ladder(truncation):
    """(create, destroy) matrices for one mode, from SeQUeNCe's Fock manager.

    Example:
        >>> create, destroy = ladder(2)
        >>> np.round(create, 3)
        array([[0.   , 0.   , 0.   ],
               [1.   , 0.   , 0.   ],
               [0.   , 1.414, 0.   ]])
    """
    return QuantumManagerDensityFock(truncation=truncation).build_ladder()


def tmsv_amplitudes(mean_photon_number, truncation):
    """Amplitudes c_n of a two-mode squeezed vacuum sum_n c_n |n, n>.

    Same formula as SeQUeNCe's ``SPDCSource``: thermal amplitudes up to
    ``truncation - 1`` pairs, and the leftover probability goes in the last level.

    Example:
        >>> np.round(tmsv_amplitudes(0.1, 2) ** 2, 4)
        array([0.9091, 0.0826, 0.0083])
    """
    mu = mean_photon_number
    if mu < 0:
        raise ValueError(f"mean_photon_number must be >= 0, got {mu}")
    amps = [np.sqrt(mu / (mu + 1)) ** m / np.sqrt(mu + 1) for m in range(truncation)]
    amps.append(np.sqrt(max(0.0, 1 - sum(a**2 for a in amps))))
    return np.array(amps)


def spdc_state(mean_photon_number, truncation=2):
    """Four-mode density matrix of a polarization-entangled SPDC source.

    Two two-mode squeezed vacua, one pairing A_H with B_H and one pairing A_V
    with B_V. (A real type-II crystal pairs H with V; we include the half-wave
    plate on Bob's side that turns that into this form, so a single pair is the
    Bell state Phi+ = (|HH> + |VV>) / sqrt(2).)

    Example:
        >>> rho = spdc_state(0.01)
        >>> round(float(np.trace(rho).real), 6)
        1.0
    """
    c = tmsv_amplitudes(mean_photon_number, truncation)
    d = truncation + 1
    psi = np.zeros((d, d, d, d))
    for n in range(d):
        for m in range(d):
            psi[n, m, n, m] = c[n] * c[m]  # n pairs in H, m pairs in V
    psi = psi.flatten()
    return np.outer(psi, psi).astype(complex)


def apply_loss(rho, transmissivity, modes, truncation):
    """Send the listed modes through a lossy channel (fiber, or a lossy element).

    Each photon survives with probability ``transmissivity``. This uses
    SeQUeNCe's ``QuantumManagerDensityFock.add_loss``.

    Example:
        >>> rho = apply_loss(spdc_state(0.01), 0.5, modes=[0, 1], truncation=2)
    """
    if not 0 <= transmissivity <= 1:
        raise ValueError(f"transmissivity must be between 0 and 1, got {transmissivity}")
    qm = QuantumManagerDensityFock(truncation=truncation)
    keys = [qm.new() for _ in MODES]
    qm.set(keys, rho)
    for m in modes:
        qm.add_loss(keys[m], 1 - transmissivity)
    return np.array(qm.states[keys[0]].state, dtype=complex)


def fiber_transmissivity(distance_km, loss_db_per_km):
    """Fraction of photons that survive ``distance_km`` of fiber.

    Example:
        >>> round(fiber_transmissivity(50, 0.2), 3)
        0.1
    """
    if distance_km < 0:
        raise ValueError(f"distance_km must be >= 0, got {distance_km}")
    return 10 ** (-loss_db_per_km * distance_km / 10)


def _two_mode_generators(truncation):
    """Schwinger operators Jx, Jy, Jz on the (H, V) mode pair of one player."""
    create, destroy = ladder(truncation)
    eye = np.eye(truncation + 1)
    aH, aV = np.kron(destroy, eye), np.kron(eye, destroy)
    aHd, aVd = aH.conj().T, aV.conj().T
    jx = (aHd @ aV + aVd @ aH) / 2
    jy = (aHd @ aV - aVd @ aH) / 2j
    jz = (aHd @ aH - aVd @ aV) / 2
    return jx, jy, jz


def polarization_unitary(angles_deg, truncation):
    """Fock-space version of ``bg.polarization_rotation`` on one player's (H, V) modes.

    On one photon it is exactly the 2x2 qubit rotation. It is exact for up to
    ``truncation`` photons in the two modes together.

    Example:
        >>> U = polarization_unitary([0, 90, 0], truncation=1)
        >>> U.shape
        (4, 4)
    """
    a, b, c = np.radians(np.asarray(angles_deg, dtype=float))
    jx, jy, jz = _two_mode_generators(truncation)
    return expm(-2j * c * jz) @ expm(-2j * b * jy) @ expm(-2j * a * jx)


def analyzer_unitary(angle_deg, correction_deg, truncation):
    """What a player does before the polarizing beam splitter.

    First the correction wave plates, then a rotation that turns polarization
    ``angle_deg`` onto H. After this, the H port means outcome 0 and the V port
    means outcome 1.
    """
    return polarization_unitary([0, -angle_deg, 0], truncation) @ polarization_unitary(
        correction_deg, truncation
    )


def threshold_povm_diagonal(detector_efficiency, dark_count_prob, truncation):
    """Probability that a threshold detector does NOT click, for 0..truncation photons.

    Same model as SeQUeNCe's ``QSDetectorFockDirect`` POVMs (each photon is seen
    with probability ``detector_efficiency``), plus an independent dark count
    with probability ``dark_count_prob`` per detection window.

    Example:
        >>> np.round(threshold_povm_diagonal(0.5, 0.0, 2), 3)
        array([1.  , 0.5 , 0.25])
    """
    if not 0 <= detector_efficiency <= 1:
        raise ValueError(f"detector_efficiency must be between 0 and 1, got {detector_efficiency}")
    if not 0 <= dark_count_prob <= 1:
        raise ValueError(f"dark_count_prob must be between 0 and 1, got {dark_count_prob}")
    n = np.arange(truncation + 1)
    return (1 - dark_count_prob) * (1 - detector_efficiency) ** n


def photon_number_distribution(rho, truncation):
    """p[nA, nB]: probability of nA photons at Alice and nB at Bob (both polarizations).

    Example:
        >>> p = photon_number_distribution(spdc_state(0.1), 2)
        >>> round(float(p[0, 0]), 3)
        0.826
    """
    d = truncation + 1
    diag = np.real(np.diag(rho)).reshape(d, d, d, d)
    p = np.zeros((2 * d - 1, 2 * d - 1))
    for ah in range(d):
        for av in range(d):
            for bh in range(d):
                for bv in range(d):
                    p[ah + av, bh + bv] += diag[ah, av, bh, bv]
    return p
