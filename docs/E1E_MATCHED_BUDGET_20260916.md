# `T^T D T` against the hierarchical construction, at a comparable budget

Date 2026-09-16.  Seat 0328, `d = 12792`.  Script `superelement/factor_fit/e1e.py`;
raw `docs/data/E1E.json`.

## Why

E1-D compared `T^T D T` at 720k parameters against a hierarchical construction at 12.23e6 — a 17x
budget gap, so "hierarchical is 254x better on the gate" was not one question.  This gives
`T^T D T` a comparable budget.

**The budget did not land where it was aimed.**  The target was 11.9e6 lifting coefficients from
six checkerboard levels capped at 2.8e6 each; the natural pair counts saturate below the cap at the
first five radii, so the run got **7,119,181 lifting + 326,000 block = 7,445,181 parameters
(9.10% of dense)**.  That is compared against the hierarchical rows that bracket it, not against
the 12.23e6 one.

Everything else as in E1-D: checkerboard bipartition (100% coverage at every level), `T` initialised
at `I`, `D` eliminated in closed form, Adam `lr 1e-2` cosine-annealed, then a full-spectrum audit.

## Measured

The descent worked, by floor standards — the largest floor improvement any descent here has produced:

```
    floor at T = I      3.355179e-01
    floor after 600     1.353013e-02        x24.8, flat from step ~500
```

The audit:

```
    eps_op   1.5248e+01        mu in [2.2885e-02, 1.6248e+01]
    1864 modes outside +-3%    404 outside +-10%
```

Against the hierarchical factor construction on the same seat:

| construction | params | % of dense | `eps_op` | fitted? |
|---|---:|---:|---:|---|
| hierarchical, tol 1e-1 | 6,860,058 | 8.38% | **0.4164** | not at all |
| **`T^T D T`, descended** | **7,445,181** | **9.10%** | **15.248** | 600 Adam steps |
| hierarchical, tol 3e-2 | 8,581,236 | 10.49% | **0.1524** | not at all |

**With more parameters than the 8.38% hierarchical row, the descended `T^T D T` is 37x worse on the
gate**, and it does not reach the worst hierarchical point measured at any budget.

## The floor misled again, by a factor of about 3

`T = I` with an optimal block-diagonal `D` has floor `3.355e-01` and, from `BANDED_AUDIT`'s
radius-0 row (the floor is invariant under block-diagonal rescaling of `T`, so that row is the same
operator class), `eps_op = 142.8`.  So:

```
    floor  3.355e-01 -> 1.353e-02      improved 24.8x
    gate   1.428e+02 -> 1.525e+01      improved  9.4x
```

The descent really did make the operator better; it improved the thing it was minimising 2.6x more
than the thing that decides acceptance, and it finishes 152x above +-10%.

## Scope

One descent, one initialisation, one schedule, one layer family, one seat.  `F_class = inf_T F(T)`
is bounded above by whatever a descent finds, never below, so this **does not bound the
`T^T D T` class** — see `docs/CLAIM_SCOPE_20260916.md`.  What it does retire is the hypothesis that
E1-D's result was an artefact of the budget gap: at a comparable budget, with the coverage bug
fixed and a descent that moved the floor 25x, the architecture is still nowhere near a construction
that needs no fitting at all.
