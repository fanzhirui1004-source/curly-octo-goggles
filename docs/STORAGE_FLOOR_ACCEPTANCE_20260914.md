# Accepting a direction that cannot be measured

## What the gate does today

`no_gp_same_qr_reference.py::classify_acceptance` compares, on the four screen
directions, the root energy against the original-G body energy:

```python
ratio = root / energy
bad   = (ratio < .9) | (ratio > 1.1)
```

Anything outside the band is attributed to the factor, to the compilation, or
left as `MULTISTAGE_ATTRIBUTION_UNRESOLVED`.

(For the record: this is not in `batch_verification.py`, which only tracks batch
cadence and which sample in a batch is the fixed QA sample.)

## Why that is wrong for the softest directions

The stored factor R is FP64. For a direction z, the energy it can represent is
bounded below by its own rounding:

```
E = ||R z||^2                                   the stored energy
N = sum_r ( 0.5 * eps * sum_j |R_rj z_j| )^2    what FP64 storage cannot resolve
```

`N` is the row-wise half-ulp of the magnitude actually accumulated in that row,
squared and summed - the energy of the rounding, computed on the same rows that
produce `E`. `E/N` is how far the value stands above its own floor, and the
achievable relative energy error is about `1/(E/N)`.

Measured:

| seat | direction | E/N | ratio the gate saw |
|---|---|---|---|
| 52 | SOFT_POWER_0 | 3.04 | 0.275 |
| 52 | SOFT_POWER_1 | 3.58 | 0.157 |
| 52 | RANDOM_0 | 1.13e31 | 1.0000 |
| 52 | RANDOM_1 | 1.10e31 | 1.0000 |
| 13 | SOFT_POWER_0 | 32.9 | in band, 4.2% off |
| 13 | SOFT_POWER_1 | 32.6 | in band, 9.2% off |
| 13 | RANDOM_0/1 | 1.11e31 | 1.0000 |

Regenerating the screen directions instead of reading the recorded ones
reproduces this: for seat 13 the two random directions come back bit-identical
(same RNG draws) and the two soft ones give `E/N` 33.1 and 33.9 against the
recorded 32.9 and 32.6 - the power iteration converges to the same soft subspace
through a different summation order. The backfill is therefore comparable across
the 31 labels that never recorded their directions.

seat 52's two soft directions are within a factor of ~3 of their own floor. Their
stored energies, 1.88e-42 and 2.22e-42, sit on a floor of 6.20e-43. A deviation
there is not evidence about the factor; it is the factor's own rounding read back.
The ±10% band cannot be evaluated on such a direction at all.

This is not the same claim as "the error is small enough to ignore". It is
"the quantity is not measurable, so the test does not apply". The two must stay
distinguishable, or the gate quietly stops being able to detect a real factor
error in a soft direction.

## The floor predicts the observed deviation

Across a decade of `E/N`, the deviation the gate measured is about `3/(E/N)`:

| seat | direction | E/N | \|ratio-1\| | product |
|---|---|---|---|---|
| 52 | SOFT_POWER_0 | 3.04 | 0.725 | 2.20 |
| 52 | SOFT_POWER_1 | 3.58 | 0.843 | 3.02 |
| 13 | SOFT_POWER_0 | 32.9 | 0.042 | 1.38 |
| 13 | SOFT_POWER_1 | 32.6 | 0.092 | 3.00 |

The product stays in 1.4-3.0 while `E/N` moves by 10x and the deviation by 20x.
So the floor is not just a plausibility argument - it quantitatively accounts for
what the acceptance gate saw. Take `C = 3` as the upper envelope.

Other measured labels sit far above this regime and are unaffected: seat 34
`E/N = 69`, seat 139 `3.6e8`, seat 85 `3.5e16`, and every RANDOM direction
measured so far is ~1.1e31.

## The rule

Two changes, neither of which weakens the gate on a direction that carries
information.

```python
C = 3.0            # measured floor-to-deviation constant, refit as data lands
UNJUDGEABLE = 30   # below this, C/(E/N) alone spans the +/-10% band

sn        = np.asarray(screen['signal_to_storage_noise'], dtype=float)
floor_tol = C/sn                       # what rounding alone can produce
tol       = 0.10 + floor_tol           # the band, widened only where needed
judgeable = np.isfinite(sn) & (sn >= UNJUDGEABLE)
bad       = judgeable & (np.abs(ratio-1) > tol)
```

`UNJUDGEABLE = 30` is exactly where `C/(E/N)` reaches the band width: below it,
an out-of-band ratio is consistent with rounding alone and says nothing about the
factor. Above it, the widened tolerance is small - seat 34 at `E/N = 69` widens
the band from 10% to 14.3%, seat 139 from 10% to 10.0000001%, seat 85 not at all.
The gate is unchanged for every direction that has more than about two digits.

The passing status splits, so an unjudgeable direction is never reported as a
pass:

- all four judgeable and in band -> `SELECTED_ACCEPTANCE_CHECKS_PASS`
- some direction below the floor -> `SELECTED_ACCEPTANCE_CHECKS_PASS_BELOW_STORAGE_FLOOR`,
  carrying `signal_to_storage_noise`, `judgeable_directions`, the widened
  tolerances, and the excluded directions by name.

Under this rule seat 52 is delivered with documented scope: its two random
directions agree to 5 and 4 significant figures, and its two soft directions are
recorded as below the representability floor with their measured `E/N`, rather
than being counted as a factor inconsistency.

## Producer side

`probes()` in `no_gp_r3_root_screen.py` already streams R's rows once in
`action_extended`. The floor needs one more accumulation over the same rows:

```python
def storage_floor(root, z):
    """Energy that FP64 storage of R cannot resolve, per supplied direction."""
    a = np.zeros(z.shape); A = np.abs(z); off = 0
    for i in range(root.d):
        row = np.abs(np.asarray(root.packed[off:off+root.d-i], dtype=np.float64))
        a[i] = row @ A[i:]; off += root.d-i
    return ((0.5*np.finfo(np.float64).eps*a)**2).sum(0)
```

emitted into `screen/RESULT.json` alongside `predicted_energy`, as
`storage_noise_floor` and `signal_to_storage_noise`. Cost is one pass over R,
the same as one `action`, against a screen stage that already takes ~18 s on
seat 52.

Both files are frozen by `--source-sha`, so this lands as a new versioned
package, not an edit in place, and it does not run until dispatch resumes.

## Why this matters beyond the gate

The 25-order margin between the storage floor and physically meaningful
thin-plate modes protects *absolute* quantities. It does not protect a
*relative* loss. The whitened log-det divergence

```
D = sum_i ( mu_i - log mu_i - 1 )
```

weights every direction equally in relative terms. A direction whose teacher
value is its own rounding contributes a large penalty that no network can ever
reduce, so capacity is spent fitting noise and the convergence curve stops
reporting real progress. Recording `signal_to_storage_noise` per label per
direction is what lets the training loss mask or down-weight those directions.

That is the operational consequence of accepting the floor as error: accept it,
and you must know where it is.

## Backfill

`superelement/storage_floor.py` computes this for any delivered packet. Only 3
of the 34 delivered labels (13, 85, 139) recorded their production screen
directions; for the rest the tool regenerates them with the registered
convention (seed 2026091407, 12 power iterations, two soft starts then two
random) so the two cases are comparable. R is streamed, never materialised.
