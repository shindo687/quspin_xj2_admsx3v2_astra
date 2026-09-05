"""Automatic-differentiation rules for selected QuSpin callables.

This package is a sidecar: QuSpin remains the source of every primal value,
while this module only supplies explicit ChainRules JVP/VJP rules.  Importing
``quspin_ad`` registers rules for the sidecar callables exported here.  When a
QuSpin installation is available, the corresponding upstream callables are
also registered (``register_upstream_rules`` is idempotent).
"""

from .rules import (
    ED_state_vs_time,
    KL_div,
    anti_commutator,
    coherent_state,
    commutator,
    lin_comb_Q_T,
    project_op,
    register_upstream_rules,
)
from .rules import ad as _ad
from .floquet import (
    FloquetOperator,
    FloquetEigensystemResult,
    FloquetSpectralGapError,
    floquet_adapter,
    floquet_eigensystem,
    floquet_eigensystem_from_object,
    floquet_projectors,
    floquet_quasienergy,
    floquet_quasienergy_from_object,
)

ZERO = _ad.ZERO
grad = _ad.grad
jvp = _ad.jvp
value_and_grad = _ad.value_and_grad
vjp = _ad.vjp

__all__ = [
    "ZERO",
    "ED_state_vs_time",
    "FloquetSpectralGapError",
    "FloquetOperator",
    "FloquetEigensystemResult",
    "KL_div",
    "anti_commutator",
    "coherent_state",
    "commutator",
    "grad",
    "floquet_eigensystem",
    "floquet_eigensystem_from_object",
    "floquet_adapter",
    "floquet_projectors",
    "floquet_quasienergy",
    "floquet_quasienergy_from_object",
    "jvp",
    "lin_comb_Q_T",
    "project_op",
    "register_upstream_rules",
    "value_and_grad",
    "vjp",
]
