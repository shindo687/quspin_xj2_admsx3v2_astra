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


class FloquetOperator:
    """Fixed-grid Floquet boundary with named physical parameter Jacobians.

    ``operator`` returns the dense one-period unitary for the supplied physical
    parameters.  ``parameter_derivatives`` contains analytic derivatives of
    that unitary at the reference point.  This small adapter is useful for a
    driven lattice where the upstream ``Floquet`` object has already been
    constructed, while keeping the upstream ``EF``/``VF`` values as the
    forward reference.  A derivative may be a matrix or a callable accepting
    the parameter values and returning a matrix.
    """

    PARAMETERS = ("drive_phase", "synthetic_gauge", "momentum")

    def __init__(
        self,
        operator: Callable[..., object] | object,
        T: float,
        *,
        drive_phase: float = 0.0,
        synthetic_gauge: float = 0.0,
        momentum: float = 0.0,
        parameter_derivatives: Mapping[str, object] | None = None,
        upstream: object | None = None,
    ) -> None:
        self.operator = operator
        self.T = T
        self.parameters = {
            "drive_phase": drive_phase,
            "synthetic_gauge": synthetic_gauge,
            "momentum": momentum,
        }
        self.parameter_derivatives = dict(parameter_derivatives or {})
        self.upstream = upstream

    @property
    def UF(self) -> np.ndarray:
        if callable(self.operator):
            return np.asarray(self.operator(**self.parameters))
        return np.asarray(self.operator)

    def derivative(self, name: str) -> np.ndarray:
        value = self.parameter_derivatives.get(name, ad.ZERO)
        if value is ad.ZERO:
            return np.zeros_like(self.UF, dtype=np.result_type(self.UF, np.complex128))
        if callable(value):
            value = value(**self.parameters)
        result = np.asarray(value)
        if result.shape != self.UF.shape:
            raise ValueError(f"dUF/d{name} shape must match UF")
        return result

    @property
    def EF(self) -> np.ndarray:
        """Native quasienergies when available, otherwise sidecar values."""
        if self.upstream is not None and hasattr(self.upstream, "EF"):
            return np.asarray(self.upstream.EF)
        return floquet_quasienergy(self)

    @property
    def VF(self) -> np.ndarray:
        """Native Floquet vectors when available, otherwise sidecar values."""
        if self.upstream is not None and hasattr(self.upstream, "VF"):
            return np.asarray(self.upstream.VF)
        return floquet_eigensystem(self).eigenvectors


def floquet_adapter(
    floquet: object,
    *,
    parameter_derivatives: Mapping[str, object] | None = None,
    operator: Callable[..., object] | object | None = None,
    drive_phase: float = 0.0,
    synthetic_gauge: float = 0.0,
    momentum: float = 0.0,
) -> FloquetOperator:
    """Adapt an upstream ``quspin.tools.Floquet.Floquet`` object.

    The upstream object supplies ``UF`` and ``T`` and remains the source of
    native ``EF`` and ``VF``.  The optional ``operator`` and derivative map
    describe the fixed-grid physical parameter dependence.  For a static
    upstream object, passing no map is valid and gives zero physical
    derivatives; a driven model should provide its analytic Jacobians.
    """
    if isinstance(floquet, FloquetOperator):
        return floquet
    if parameter_derivatives is None:
        parameter_derivatives = getattr(
            floquet, "parameter_derivatives",
            getattr(floquet, "_ad_parameter_derivatives", None),
        )
    if operator is None:
        operator = getattr(floquet, "ad_period_operator", None)
    if operator is None:
        if not hasattr(floquet, "UF") or not hasattr(floquet, "T"):
            raise TypeError("floquet must expose UF and T")
        operator = getattr(floquet, "UF")
    return FloquetOperator(
        operator,
        float(getattr(floquet, "T")),
        drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge,
        momentum=momentum,
        parameter_derivatives=parameter_derivatives,
        upstream=floquet,
    )


def _array(value: object, name: str) -> np.ndarray:
    result = np.asarray(value)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise TypeError(f"{name} must be a square dense matrix")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain finite values")
    return result


def _operator_and_metadata(
    UF: object,
    T: object,
    parameter_derivatives: Mapping[str, object] | None,
):
    """Resolve dense UF/T and analytic physical-parameter Jacobians."""
    if isinstance(UF, FloquetOperator):
        adapter = UF
        derivatives = dict(adapter.parameter_derivatives)
        return adapter.UF, adapter.T, derivatives
    if hasattr(UF, "UF") and hasattr(UF, "T") and not isinstance(UF, np.ndarray):
        adapter = floquet_adapter(UF, parameter_derivatives=parameter_derivatives)
        return adapter.UF, adapter.T, dict(adapter.parameter_derivatives)
    return UF, T, dict(parameter_derivatives or {})


def _physical_derivative(
    name: str,
    UF: object,
    parameters: Mapping[str, object],
    parameter_derivatives: Mapping[str, object],
) -> np.ndarray:
    value = parameter_derivatives.get(name, ad.ZERO)
    if value is ad.ZERO and isinstance(UF, FloquetOperator):
        return UF.derivative(name)
    if callable(value):
        value = value(**parameters)
    if value is ad.ZERO:
        dense = UF.UF if isinstance(UF, FloquetOperator) else UF
        return np.zeros_like(_array(dense, "UF"), dtype=np.result_type(dense, np.complex128))
    result = np.asarray(value)
    dense = UF.UF if isinstance(UF, FloquetOperator) else UF
    shape = _array(dense, "UF").shape
    if result.shape != shape:
        raise ValueError(f"dUF/d{name} shape must match UF")
    return result


def _physical_parameters(
    UF: object,
    drive_phase: object,
    synthetic_gauge: object,
    momentum: object,
    parameter_derivatives: Mapping[str, object] | None,
) -> tuple[dict[str, object], dict[str, object]]:
    # A FloquetOperator owns the reference point at which its callable
    # operator and Jacobians are defined.  The AD rule entry points expose
    # these names with zero-valued Python defaults, so using those arguments
    # directly would evaluate a callable Jacobian at zero while evaluating UF
    # at the adapter's stored (possibly nonzero) parameters.  Keep the two
    # evaluations at the same reference point by resolving parameters from
    # the adapter whenever one is supplied.
    if isinstance(UF, FloquetOperator):
        parameters = dict(UF.parameters)
    else:
        parameters = {
            "drive_phase": drive_phase,
            "synthetic_gauge": synthetic_gauge,
            "momentum": momentum,
        }
    source = dict(parameter_derivatives or {})
    if isinstance(UF, FloquetOperator):
        source = {**UF.parameter_derivatives, **source}
    return parameters, source


def _total_dU(
    UF: object,
    tangents: Mapping[str, object],
    parameter_derivatives: Mapping[str, object],
    parameters: Mapping[str, object],
    base: object = ad.ZERO,
) -> object:
    dU = base
    for name in FloquetOperator.PARAMETERS:
        tangent = tangents.get(name, ad.ZERO)
        if tangent is ad.ZERO:
            continue
        contribution = _physical_derivative(name, UF, parameters, parameter_derivatives)
        contribution = contribution * tangent
        dU = contribution if dU is ad.ZERO else dU + contribution
    return dU


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
    *,
    drive_phase: float = 0.0,
    synthetic_gauge: float = 0.0,
    momentum: float = 0.0,
    parameter_derivatives: Mapping[str, object] | None = None,
) -> FloquetEigensystemResult:
    """Return ``(quasienergies, eigenvectors)`` for a dense period unitary.

    Eigenvectors are columns and are parallel-transport gauge-fixed in the
    derivative (``v.conj() @ dv == 0``).  ``branch`` is fixed during AD and can
    be an integer winding array to select a continuous quasienergy branch.
    """
    UF, T, _ = _operator_and_metadata(UF, T, parameter_derivatives)
    *_, vectors, energies = _forward(UF, T, branch, float(gap_tol))
    return FloquetEigensystemResult(energies, vectors)


def floquet_quasienergy(
    UF: object,
    T: object = 1.0,
    branch: object = "principal",
    gap_tol: float = 1e-10,
    *,
    drive_phase: float = 0.0,
    synthetic_gauge: float = 0.0,
    momentum: float = 0.0,
    parameter_derivatives: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Return ordered Floquet quasienergies using the documented branch."""
    return floquet_eigensystem(
        UF, T, branch, gap_tol,
        drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge,
        momentum=momentum,
        parameter_derivatives=parameter_derivatives,
    ).quasienergy


def floquet_eigensystem_from_object(
    floquet: object,
    branch: object = "principal",
    gap_tol: float = 1e-10,
    *,
    drive_phase: float = 0.0,
    synthetic_gauge: float = 0.0,
    momentum: float = 0.0,
    parameter_derivatives: Mapping[str, object] | None = None,
) -> FloquetEigensystemResult:
    """Return sidecar ``EF``/``VF`` parity for an upstream Floquet object."""
    adapter = floquet_adapter(
        floquet,
        parameter_derivatives=parameter_derivatives,
        drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge,
        momentum=momentum,
    )
    return floquet_eigensystem(adapter, adapter.T, branch, gap_tol)


def floquet_quasienergy_from_object(
    floquet: object,
    branch: object = "principal",
    gap_tol: float = 1e-10,
    *,
    drive_phase: float = 0.0,
    synthetic_gauge: float = 0.0,
    momentum: float = 0.0,
    parameter_derivatives: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Return native-object Floquet quasienergies through the AD boundary."""
    return floquet_eigensystem_from_object(
        floquet, branch, gap_tol,
        drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge,
        momentum=momentum,
        parameter_derivatives=parameter_derivatives,
    ).quasienergy


def floquet_projectors(
    UF: object,
    T: object = 1.0,
    branch: object = "principal",
    gap_tol: float = 1e-10,
    *,
    drive_phase: float = 0.0,
    synthetic_gauge: float = 0.0,
    momentum: float = 0.0,
    parameter_derivatives: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Return phase-invariant rank-one projectors with shape ``(N,N,N)``."""
    _, vectors = floquet_eigensystem(
        UF, T, branch, gap_tol,
        drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge,
        momentum=momentum,
        parameter_derivatives=parameter_derivatives,
    )
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
def _eigensystem_jvp(
    tangents, UF, T=1.0, branch="principal", gap_tol=1e-10,
    *, drive_phase=0.0, synthetic_gauge=0.0, momentum=0.0,
    parameter_derivatives=None,
):
    supported = ("UF", "T", *FloquetOperator.PARAMETERS)
    _unsupported(tangents, supported)
    value = floquet_eigensystem(
        UF, T, branch, gap_tol, drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge, momentum=momentum,
        parameter_derivatives=parameter_derivatives,
    )
    dU = _active(tangents, "UF")
    dT = _active(tangents, "T")
    params, derivatives = _physical_parameters(
        UF, drive_phase, synthetic_gauge, momentum, parameter_derivatives
    )
    dU = _total_dU(UF, tangents, derivatives, params, dU)
    if dU is ad.ZERO and dT is ad.ZERO:
        return value, ad.ZERO
    dense_U, resolved_T, _ = _operator_and_metadata(UF, T, parameter_derivatives)
    U, period, lam, V, E = _forward(dense_U, resolved_T, branch, float(gap_tol))
    dmat = np.zeros_like(U, dtype=np.result_type(U, np.complex128)) if dU is ad.ZERO else _array(dU, "dUF")
    if dU is not ad.ZERO and dmat.shape != U.shape:
        raise ValueError("dUF shape must match UF")
    dt = 0.0 if dT is ad.ZERO else float(np.asarray(dT))
    dE, dV = _linearization(U, period, lam, V, E, dmat, dt)
    return (E, V), (dE, dV)


@ad.rules.vjp_for(floquet_eigensystem)
def _eigensystem_vjp(
    wrt, UF, T=1.0, branch="principal", gap_tol=1e-10,
    *, drive_phase=0.0, synthetic_gauge=0.0, momentum=0.0,
    parameter_derivatives=None,
):
    supported = ("UF", "T", *FloquetOperator.PARAMETERS)
    _unsupported(dict.fromkeys(wrt), supported)
    value = floquet_eigensystem(
        UF, T, branch, gap_tol, drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge, momentum=momentum,
        parameter_derivatives=parameter_derivatives,
    )
    dense_U, resolved_T, _ = _operator_and_metadata(UF, T, parameter_derivatives)
    U, period, lam, V, E = _forward(dense_U, resolved_T, branch, float(gap_tol))
    params, derivatives = _physical_parameters(
        UF, drive_phase, synthetic_gauge, momentum, parameter_derivatives
    )

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
        for name in FloquetOperator.PARAMETERS:
            if name in wrt:
                result[name] = float(np.real(np.vdot(gU, _physical_derivative(
                    name, UF, params, derivatives
                ))))
        return result

    return value, pullback


@ad.rules.jvp_for(floquet_quasienergy)
def _quasienergy_jvp(
    tangents, UF, T=1.0, branch="principal", gap_tol=1e-10,
    *, drive_phase=0.0, synthetic_gauge=0.0, momentum=0.0,
    parameter_derivatives=None,
):
    supported = ("UF", "T", *FloquetOperator.PARAMETERS)
    _unsupported(tangents, supported)
    value = floquet_quasienergy(
        UF, T, branch, gap_tol, drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge, momentum=momentum,
        parameter_derivatives=parameter_derivatives,
    )
    dU, dT = _active(tangents, "UF"), _active(tangents, "T")
    params, derivatives = _physical_parameters(
        UF, drive_phase, synthetic_gauge, momentum, parameter_derivatives
    )
    dU = _total_dU(UF, tangents, derivatives, params, dU)
    if dU is ad.ZERO and dT is ad.ZERO:
        return value, ad.ZERO
    dense_U, resolved_T, _ = _operator_and_metadata(UF, T, parameter_derivatives)
    U, period, lam, V, E = _forward(dense_U, resolved_T, branch, float(gap_tol))
    dmat = np.zeros_like(U, dtype=np.result_type(U, np.complex128)) if dU is ad.ZERO else _array(dU, "dUF")
    dt = 0.0 if dT is ad.ZERO else float(np.asarray(dT))
    dE, _ = _linearization(U, period, lam, V, E, dmat, dt)
    return value, dE


@ad.rules.vjp_for(floquet_quasienergy)
def _quasienergy_vjp(
    wrt, UF, T=1.0, branch="principal", gap_tol=1e-10,
    *, drive_phase=0.0, synthetic_gauge=0.0, momentum=0.0,
    parameter_derivatives=None,
):
    supported = ("UF", "T", *FloquetOperator.PARAMETERS)
    _unsupported(dict.fromkeys(wrt), supported)
    E = floquet_quasienergy(
        UF, T, branch, gap_tol, drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge, momentum=momentum,
        parameter_derivatives=parameter_derivatives,
    )
    dense_U, resolved_T, _ = _operator_and_metadata(UF, T, parameter_derivatives)
    _, period, lam, V, _ = _forward(dense_U, resolved_T, branch, float(gap_tol))
    params, derivatives = _physical_parameters(
        UF, drive_phase, synthetic_gauge, momentum, parameter_derivatives
    )

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
        if any(name in wrt for name in FloquetOperator.PARAMETERS):
            gU = V @ np.diag(-1j * ge * lam / period) @ V.conj().T
            for name in FloquetOperator.PARAMETERS:
                if name in wrt:
                    result[name] = float(np.real(np.vdot(
                        gU, _physical_derivative(name, UF, params, derivatives)
                    )))
        return result

    return E, pullback


@ad.rules.jvp_for(floquet_projectors)
def _projectors_jvp(
    tangents, UF, T=1.0, branch="principal", gap_tol=1e-10,
    *, drive_phase=0.0, synthetic_gauge=0.0, momentum=0.0,
    parameter_derivatives=None,
):
    supported = ("UF", "T", *FloquetOperator.PARAMETERS)
    _unsupported(tangents, supported)
    value = floquet_projectors(
        UF, T, branch, gap_tol, drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge, momentum=momentum,
        parameter_derivatives=parameter_derivatives,
    )
    dU, dT = _active(tangents, "UF"), _active(tangents, "T")
    params, derivatives = _physical_parameters(
        UF, drive_phase, synthetic_gauge, momentum, parameter_derivatives
    )
    dU = _total_dU(UF, tangents, derivatives, params, dU)
    if dU is ad.ZERO and dT is ad.ZERO:
        return value, ad.ZERO
    dense_U, resolved_T, _ = _operator_and_metadata(UF, T, parameter_derivatives)
    U, period, lam, V, E = _forward(dense_U, resolved_T, branch, float(gap_tol))
    dmat = np.zeros_like(U, dtype=np.result_type(U, np.complex128)) if dU is ad.ZERO else _array(dU, "dUF")
    dt = 0.0 if dT is ad.ZERO else float(np.asarray(dT))
    _, dV = _linearization(U, period, lam, V, E, dmat, dt)
    tangent = np.einsum("ij,kj->jik", dV, V.conj()) + np.einsum("ij,kj->jik", V, dV.conj())
    return value, tangent


@ad.rules.vjp_for(floquet_projectors)
def _projectors_vjp(
    wrt, UF, T=1.0, branch="principal", gap_tol=1e-10,
    *, drive_phase=0.0, synthetic_gauge=0.0, momentum=0.0,
    parameter_derivatives=None,
):
    supported = ("UF", "T", *FloquetOperator.PARAMETERS)
    _unsupported(dict.fromkeys(wrt), supported)
    value = floquet_projectors(
        UF, T, branch, gap_tol, drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge, momentum=momentum,
        parameter_derivatives=parameter_derivatives,
    )
    dense_U, resolved_T, _ = _operator_and_metadata(UF, T, parameter_derivatives)
    U, period, lam, V, E = _forward(dense_U, resolved_T, branch, float(gap_tol))

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
        raw = _eigensystem_vjp(
            wrt, UF, T, branch, gap_tol,
            drive_phase=drive_phase,
            synthetic_gauge=synthetic_gauge,
            momentum=momentum,
            parameter_derivatives=parameter_derivatives,
        )[1]
        return raw((np.zeros_like(E), gv))

    return value, pullback


@ad.rules.jvp_for(floquet_quasienergy_from_object)
def _object_quasienergy_jvp(
    tangents, floquet, branch="principal", gap_tol=1e-10,
    *, drive_phase=0.0, synthetic_gauge=0.0, momentum=0.0,
    parameter_derivatives=None,
):
    _unsupported(tangents, FloquetOperator.PARAMETERS)
    adapter = floquet_adapter(
        floquet,
        parameter_derivatives=parameter_derivatives,
        drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge,
        momentum=momentum,
    )
    return _quasienergy_jvp(
        tangents, adapter, adapter.T, branch, gap_tol,
        drive_phase=drive_phase, synthetic_gauge=synthetic_gauge,
        momentum=momentum, parameter_derivatives=parameter_derivatives,
    )


@ad.rules.vjp_for(floquet_quasienergy_from_object)
def _object_quasienergy_vjp(
    wrt, floquet, branch="principal", gap_tol=1e-10,
    *, drive_phase=0.0, synthetic_gauge=0.0, momentum=0.0,
    parameter_derivatives=None,
):
    _unsupported(dict.fromkeys(wrt), FloquetOperator.PARAMETERS)
    adapter = floquet_adapter(
        floquet,
        parameter_derivatives=parameter_derivatives,
        drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge,
        momentum=momentum,
    )
    return _quasienergy_vjp(
        wrt, adapter, adapter.T, branch, gap_tol,
        drive_phase=drive_phase, synthetic_gauge=synthetic_gauge,
        momentum=momentum, parameter_derivatives=parameter_derivatives,
    )


@ad.rules.jvp_for(floquet_eigensystem_from_object)
def _object_eigensystem_jvp(
    tangents, floquet, branch="principal", gap_tol=1e-10,
    *, drive_phase=0.0, synthetic_gauge=0.0, momentum=0.0,
    parameter_derivatives=None,
):
    _unsupported(tangents, FloquetOperator.PARAMETERS)
    adapter = floquet_adapter(
        floquet,
        parameter_derivatives=parameter_derivatives,
        drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge,
        momentum=momentum,
    )
    return _eigensystem_jvp(
        tangents, adapter, adapter.T, branch, gap_tol,
        drive_phase=drive_phase, synthetic_gauge=synthetic_gauge,
        momentum=momentum, parameter_derivatives=parameter_derivatives,
    )


@ad.rules.vjp_for(floquet_eigensystem_from_object)
def _object_eigensystem_vjp(
    wrt, floquet, branch="principal", gap_tol=1e-10,
    *, drive_phase=0.0, synthetic_gauge=0.0, momentum=0.0,
    parameter_derivatives=None,
):
    _unsupported(dict.fromkeys(wrt), FloquetOperator.PARAMETERS)
    adapter = floquet_adapter(
        floquet,
        parameter_derivatives=parameter_derivatives,
        drive_phase=drive_phase,
        synthetic_gauge=synthetic_gauge,
        momentum=momentum,
    )
    return _eigensystem_vjp(
        wrt, adapter, adapter.T, branch, gap_tol,
        drive_phase=drive_phase, synthetic_gauge=synthetic_gauge,
        momentum=momentum, parameter_derivatives=parameter_derivatives,
    )


__all__ = [
    "FloquetOperator",
    "FloquetSpectralGapError",
    "FloquetEigensystemResult",
    "floquet_adapter",
    "floquet_eigensystem",
    "floquet_eigensystem_from_object",
    "floquet_projectors",
    "floquet_quasienergy",
    "floquet_quasienergy_from_object",
]
