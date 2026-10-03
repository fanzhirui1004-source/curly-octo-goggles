# The gate cost across four geometries

Date 2026-09-16.  Script `superelement/factor_fit/multiseat.py`; raw `docs/data/MULTISEAT.json`.

Hierarchical low-rank approximation of the factor `R*`, the construction that first passed +-3% on
seat 0328.  Each seat asserts the same invariants before any number is read: the partition covers
every entry exactly once, and filling every block straight from `R*` reproduces it (`eps_op` 4.3e-14
to 8.1e-14).  Each row is an achieved operator with a full-spectrum audit.

| seat | `d` | +-10% at | | +-3% at | | params / `d` |
|---|---:|---:|---:|---:|---:|---:|
| 0328 | 12,792 | 10,221,964 | 12.49% | 12,229,217 | 14.95% | 956 |
| 0253 | 12,822 | 10,259,290 | 12.48% | 12,346,525 | 15.02% | 963 |
| 0403 | 14,562 | 12,502,732 | 11.79% | 14,760,175 | 13.92% | 1014 |
| 0347 | 15,702 | 16,796,307 | 13.62% | 19,260,401 | 15.62% | 1227 |

**+-3% costs 13.9%-15.6% of the dense entry count on all four.**  That is the result: the figure is
a property of this operator family on these geometries, not of one lucky seat.

What it is **not**: a scaling law.  `d` spans only 1.23x here, the fraction has no monotone trend
across it, and `params/d` rises from 956 to 1227 — so neither "constant fraction" nor "linear in
`d`" is established, and an earlier note in this session claiming the fraction falls with `d` was
written from the first three seats and is withdrawn; 0347 does not support it.

Scope: four achieved constructions, one per seat, each proving that operator passes. Upper bounds —
a better representation can only need fewer — and silent on whether a network can predict them.
