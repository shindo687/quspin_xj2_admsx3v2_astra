"""Differentiable fixed-dimensional Floquet eigensystems.

The upstream :class:`quspin.tools.Floquet.Floquet` object remains responsible
for constructing a period propagator.  These functions operate on its dense
``UF`` matrix and expose the simple-eigenvalue eigensystem as an AD boundary.
Quasienergies use the principal eigenphase ``arg(lambda) in (-pi, pi]`` and
``epsilon = -arg(lambda) / T``, matching QuSpin's ``EF`` convention.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

import numpy as np

try:
    import chainrules as ad
except ModuleNotFoundError:  # pragma: no cover
    from . import _chainrules as ad


class FloquetSpectralGapError(ad.NonDifferentiablePoint):
    """Raised when a Floquet eigenvalue is degenerate or on the branch cut."""


class FloquetEigensystemResult(NamedTuple):
    """Tuple result with names matching QuSpin's ``EF`` and ``VF`` fields."""

    quasienergy: np.ndarray
    eigenvectors: np.ndarray

    @property
    def EF(self) -> np.ndarray:
        return self.quasienergy

    @property
    def VF(self) -> np.ndarray:
        return self.eigenvectors


def _array(value: object, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise TypeError(f"{name} must be a square dense matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _branch_array(branch: object, size: int) -> np.ndarray:
    if isinstance(branch, str):
        if branch != "principal":
            raise ValueError("branch must be 'principal' or an integer winding array")
        return np.zeros(size, dtype=int)
    b = np.asarray(branch)
    if b.ndim == 0:
        b = np.full(size, b.item())
    if b.shape != (size,) or not np.all(np.equal(b, np.round(b))):
        raise ValueError("branch must be 'principal' or an integer array of length N")
    return b.astype(int)


def _forward(UF: object, T: object, branch: object, gap_tol: float):
    U = _array(UF, "UF")
    if np.asarray(T).ndim != 0 or not np.isreal(T) or float(T) == 0:
        raise ValueError("T must be a nonzero real scalar")
    period = float(T)
    unitary_error = np.linalg.norm(U.conj().T @ U - np.eye(U.shape[0]), ord=np.inf)
    if unitary_error > max(100.0 * gap_tol, 1e-8):
        raise ValueError("UF must be unitary within the eigensystem AD tolerance")
    eigenvalues, vectors = np.linalg.eig(U)
    phases = np.angle(eigenvalues)
    if np.any(np.abs(np.abs(phases) - np.pi) <= gap_tol):
        raise FloquetSpectralGapError(
            "a Floquet eigenphase lies on the principal branch cut at +/-pi"
        )
    branches = _branch_array(branch, U.shape[0])
    energies = -(phases + 2.0 * np.pi * branches) / period
    order = np.argsort(energies, kind="stable")
    eigenvalues = eigenvalues[order]
    vectors = vectors[:, order]
    energies = energies[order]
    # QuSpin's UF is unitary, so eigvecs are orthogonal. Normalize explicitly
    # to make the rule stable for analytically supplied matrices.
    vectors = vectors / np.linalg.norm(vectors, axis=0, keepdims=True)
    gaps = np.abs(eigenvalues[:, None] - eigenvalues[None, :])
    gaps += np.eye(U.shape[0])
    min_gap = np.min(gaps)
    if min_gap <= gap_tol:
        raise FloquetSpectralGapError(
            f"Floquet eigenvalues are degenerate within gap_tol={gap_tol:g}"
        )
    return U, period, eigenvalues, vectors, energies


def floquet_eigensystem(
    UF: object,
    T: object = 1.0,
    branch: object = "principal",
    gap_tol: float = 1e-10,
) -> FloquetEigensystemResult:
    """Return ``(quasienergies, eigenvectors)`` for a dense period unitary.

    Eigenvectors are columns and are parallel-transport gauge-fixed in the
    derivative (``v.conj() @ dv == 0``).  ``branch`` is fixed during AD and can
    be an integer winding array to select a continuous quasienergy branch.
    """
    *_, vectors, energies = _forward(UF, T, branch, float(gap_tol))
    return FloquetEigensystemResult(energies, vectors)


def floquet_quasienergy(
    UF: object,
    T: object = 1.0,
    branch: object = "principal",
    gap_tol: float = 1e-10,
) -> np.ndarray:
    """Return ordered Floquet quasienergies using the documented branch."""
    return floquet_eigensystem(UF, T, branch, gap_tol).quasienergy


def floquet_projectors(
    UF: object,
    T: object = 1.0,
    branch: object = "principal",
    gap_tol: float = 1e-10,
) -> np.ndarray:
    """Return phase-invariant rank-one projectors with shape ``(N,N,N)``."""
    _, vectors = floquet_eigensystem(UF, T, branch, gap_tol)
    return np.einsum("ij,kj->jik", vectors, vectors.conj())


def _active(tangents: Mapping[str, object], name: str) -> object:
    return tangents.get(name, ad.ZERO)


def _unsupported(tangents: Mapping[str, object], supported: tuple[str, ...]) -> None:
    bad = set(tangents) - set(supported)
    if bad:
        raise ad.UnsupportedWrt(floquet_eigensystem, bad, supported=supported)


def _linearization(U, period, lam, V, E, dU, dT):
    A = V.conj().T @ dU @ V
    n = len(lam)
    dlam = np.diag(A)
    dphase = np.imag(dlam / lam)
    dE = -dphase / period - E / period * dT
    K = np.zeros((n, n), dtype=np.result_type(U, dU, np.complex128))
    for k in range(n):
        for j in range(n):
            if k != j:
                K[k, j] = A[k, j] / (lam[j] - lam[k])
    return dE, V @ K


@ad.rules.jvp_for(floquet_eigensystem)
def _eigensystem_jvp(tangents, UF, T=1.0, branch="principal", gap_tol=1e-10):
    _unsupported(tangents, ("UF", "T"))
    value = floquet_eigensystem(UF, T, branch, gap_tol)
    dU = _active(tangents, "UF")
    dT = _active(tangents, "T")
    if dU is ad.ZERO and dT is ad.ZERO:
        return value, ad.ZERO
    U, period, lam, V, E = _forward(UF, T, branch, float(gap_tol))
    dmat = np.zeros_like(U, dtype=np.result_type(U, np.complex128)) if dU is ad.ZERO else _array(dU, "dUF")
    if dU is not ad.ZERO and dmat.shape != U.shape:
        raise ValueError("dUF shape must match UF")
    dt = 0.0 if dT is ad.ZERO else float(np.asarray(dT))
    dE, dV = _linearization(U, period, lam, V, E, dmat, dt)
    return (E, V), (dE, dV)


@ad.rules.vjp_for(floquet_eigensystem)
def _eigensystem_vjp(wrt, UF, T=1.0, branch="principal", gap_tol=1e-10):
    _unsupported(dict.fromkeys(wrt), ("UF", "T"))
    value = floquet_eigensystem(UF, T, branch, gap_tol)
    U, period, lam, V, E = _forward(UF, T, branch, float(gap_tol))

    def pullback(cotangent):
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        if isinstance(cotangent, Mapping):
            gE, gV = cotangent.get("quasienergy", cotangent.get("EF")), cotangent.get("eigenvectors", cotangent.get("VF"))
        else:
            try:
                gE, gV = cotangent
            except (TypeError, ValueError) as exc:
                raise TypeError("Floquet eigensystem cotangent must be (dE, dV) or a mapping") from exc
        gU = np.zeros_like(U, dtype=np.result_type(U, np.complex128))
        gT = 0.0
        if gE is not None:
            ge = np.asarray(gE)
            if ge.shape != E.shape:
                raise ValueError("quasienergy cotangent must match EF shape")
            gU += V @ np.diag(-1j * ge * lam / period) @ V.conj().T
            gT += float(-np.sum(np.real(ge) * E / period))
        if gV is not None:
            gv = np.asarray(gV)
            if gv.shape != V.shape:
                raise ValueError("eigenvector cotangent must match VF shape")
            B = V.conj().T @ gv
            C = np.zeros_like(B, dtype=np.result_type(B, np.complex128))
            for k in range(len(lam)):
                for j in range(len(lam)):
                    if k != j:
                        C[k, j] = B[k, j] / np.conj(lam[j] - lam[k])
            gU += V @ C @ V.conj().T
        result = {}
        if "UF" in wrt:
            result["UF"] = gU
        if "T" in wrt:
            result["T"] = gT
        return result

    return value, pullback


@ad.rules.jvp_for(floquet_quasienergy)
def _quasienergy_jvp(tangents, UF, T=1.0, branch="principal", gap_tol=1e-10):
    _unsupported(tangents, ("UF", "T"))
    value = floquet_quasienergy(UF, T, branch, gap_tol)
    dU, dT = _active(tangents, "UF"), _active(tangents, "T")
    if dU is ad.ZERO and dT is ad.ZERO:
        return value, ad.ZERO
    U, period, lam, V, E = _forward(UF, T, branch, float(gap_tol))
    dmat = np.zeros_like(U, dtype=np.result_type(U, np.complex128)) if dU is ad.ZERO else _array(dU, "dUF")
    dt = 0.0 if dT is ad.ZERO else float(np.asarray(dT))
    dE, _ = _linearization(U, period, lam, V, E, dmat, dt)
    return value, dE


@ad.rules.vjp_for(floquet_quasienergy)
def _quasienergy_vjp(wrt, UF, T=1.0, branch="principal", gap_tol=1e-10):
    _unsupported(dict.fromkeys(wrt), ("UF", "T"))
    E = floquet_quasienergy(UF, T, branch, gap_tol)
    _, period, lam, V, _ = _forward(UF, T, branch, float(gap_tol))

    def pullback(cotangent):
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        ge = np.asarray(cotangent)
        if ge.shape != E.shape:
            raise ValueError("quasienergy cotangent must match EF shape")
        result = {}
        if "UF" in wrt:
            result["UF"] = V @ np.diag(-1j * ge * lam / period) @ V.conj().T
        if "T" in wrt:
            result["T"] = float(-np.sum(np.real(ge) * E / period))
        return result

    return E, pullback


@ad.rules.jvp_for(floquet_projectors)
def _projectors_jvp(tangents, UF, T=1.0, branch="principal", gap_tol=1e-10):
    _unsupported(tangents, ("UF", "T"))
    value = floquet_projectors(UF, T, branch, gap_tol)
    dU, dT = _active(tangents, "UF"), _active(tangents, "T")
    if dU is ad.ZERO and dT is ad.ZERO:
        return value, ad.ZERO
    U, period, lam, V, E = _forward(UF, T, branch, float(gap_tol))
    dmat = np.zeros_like(U, dtype=np.result_type(U, np.complex128)) if dU is ad.ZERO else _array(dU, "dUF")
    dt = 0.0 if dT is ad.ZERO else float(np.asarray(dT))
    _, dV = _linearization(U, period, lam, V, E, dmat, dt)
    tangent = np.einsum("ij,kj->jik", dV, V.conj()) + np.einsum("ij,kj->jik", V, dV.conj())
    return value, tangent


@ad.rules.vjp_for(floquet_projectors)
def _projectors_vjp(wrt, UF, T=1.0, branch="principal", gap_tol=1e-10):
    _unsupported(dict.fromkeys(wrt), ("UF", "T"))
    value = floquet_projectors(UF, T, branch, gap_tol)
    U, period, lam, V, E = _forward(UF, T, branch, float(gap_tol))

    def pullback(cotangent):
        if cotangent is ad.ZERO:
            return dict.fromkeys(wrt, ad.ZERO)
        gp = np.asarray(cotangent)
        if gp.shape != value.shape:
            raise ValueError("projector cotangent must match projector shape")
        gv = np.empty_like(V, dtype=np.result_type(V, gp, np.complex128))
        for j in range(V.shape[1]):
            gj = gp[j]
            gv[:, j] = (gj + gj.conj().T) @ V[:, j]
        # Reuse the eigensystem adjoint; projector cotangents carry no phase
        # information because the diagonal gauge component is discarded.
        raw = _eigensystem_vjp(wrt, UF, T, branch, gap_tol)[1]
        return raw((np.zeros_like(E), gv))

    return value, pullback


__all__ = [
    "FloquetSpectralGapError",
    "FloquetEigensystemResult",
    "floquet_eigensystem",
    "floquet_projectors",
    "floquet_quasienergy",
]
