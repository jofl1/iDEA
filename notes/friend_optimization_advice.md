# Expert advice — iDEA interacting solver optimization

Source: physicist+programmer friend, after reading the prompt and these
six files: `iDEA/methods/interacting.py`, `iDEA/methods/non_interacting.py`,
`iDEA/system.py`, `iDEA/state.py`, `iDEA/interactions.py`,
`tests/test_manybody.py`.

Saved here so the full reasoning survives outside conversation context.

---

## The first thing I would do

I would **not start with Davidson/LOBPCG**. I would first replace the full
labelled-particle Kronecker CSR Hamiltonian with an **exact fixed-spin
determinant-basis, matrix-free `LinearOperator`**.

The key reason: the attached implementation is still solving on the full
tensor product grid, then antisymmetrizing afterward. `hamiltonian()` builds
the many-body operator from Kronecker products over `s.count` electrons,
then adds the interaction as a diagonal over that full tensor shape.
`solve()` then computes several eigenvectors and only afterward constructs
spin factors and calls `antisymmetrize()`. The antisymmetrizer loops over
electron permutations, filters zero antisymmetrized states, normalizes,
then filters duplicates.

So your "combinatorial dimension" is the representation you want, not the
one this file is using today.

For fixed `N_up, N_down`, the exact real-space FCI basis dimension should be

$$
D = \binom{N}{N_\uparrow}\binom{N}{N_\downarrow},
$$

not $N^{N_e}$. That is not an approximation; it is just the Pauli-allowed
fixed-$S_z$ sector.

Approximate impact on your examples:

| system      | current labelled tensor | determinant basis | dimension cut |
| ----------- | ----------------------: | ----------------: | ------------: |
| 1↑1↓, N=80  |                   6,400 |             6,400 |          none |
| 2↑1↓, N=120 |               1,728,000 |           856,800 |           ~2× |
| 2↑2↓, N=60  |              12,960,000 |         3,132,900 |         ~4.1× |

It also removes the need to overcompute eigenpairs just to survive
antisymmetrization. For example, `_estimate_level()` returns
`(abs(up-down)+1)^2 * count * (k+1)`, which is especially costly for
same-spin cases; in the determinant basis, the physical ground state is
just the lowest eigenpair in the physical sector.

## How I would structure that operator

Use a product of spin-channel occupation bases:

$$
\mathcal{B} = \{I_\uparrow\} \times \{I_\downarrow\}, \quad
I_\sigma = \text{sorted occupied grid indices}.
$$

Then represent the Hamiltonian as

$$
H = \sum_{\sigma}\sum_{ij} h_{ij}\, c^\dagger_{i\sigma}c_{j\sigma}
+ \text{diagonal pair interaction}.
$$

Implementation direction:

1. Precompute all sorted combinations for the up channel and down channel.
2. Precompute a rank map from combination tuple to row/column index.
3. Precompute one-body "hops" induced by the finite-difference stencil:
   for each occupied site, try stencil-neighbor destinations; skip moves
   outside the grid or onto an already occupied same-spin site; attach the
   fermionic sign.
4. Precompute the diagonal contribution:
   $$
   \sum_i v_\text{ext}(i)\,n_i + \sum_{\text{electron pairs }a<b} v_\text{int}(i_a,i_b).
   $$
5. Store the Lanczos vector as a 2D array shaped `(D_up, D_down)`.
6. Apply up-channel transitions to rows and down-channel transitions to
   columns, plus the diagonal multiply.

For `2↑2↓, N=60`, this is particularly attractive because
`D_up = D_down = C(60,2) = 1770`; the vector is a 1770×1770 array, and the
one-body transition lists are small compared with a global CSR matrix.

This is also the cleanest way to support GPU later: the vector is dense,
the diagonal is dense, and the kinetic part is a structured sparse update.
You avoid shipping a giant CSR index structure around.

## Matrix-free before preconditioned eigensolver

After moving to the determinant basis, I would first run **plain
ARPACK/SciPy `eigsh` on a `LinearOperator`** as the control experiment.
SciPy's `eigsh` accepts sparse matrices and `LinearOperator`s, so you can
separate the representation win from the eigensolver win.

Only after that would I switch eigensolvers. Otherwise it will be hard to
know whether PRIMME/LOBPCG helped, or whether you simply stopped solving
unphysical states.

A low-risk stepping stone, before determinant basis, is a full-tensor
matrix-free matvec:

$$
H\psi = \sum_{e=1}^{N_e} h^{(e)}\psi + U(x_1,\dots,x_{N_e})\psi.
$$

That alone avoids CSR construction, Kronecker assembly, and sparse index
traffic. But it keeps the $N^{N_e}$ dimension, so I would treat it as a
prototype, not the final design.

## Library choice

My first external solver would be **PRIMME**.

PRIMME's Python `eigsh` interface accepts matrices, sparse matrices, and
`LinearOperator`s; it also exposes initial guesses, preconditioners, method
selection, and optional stats/history. It is specifically aimed at
computing a few eigenpairs of large Hermitian problems and supports
preconditioning.

My ranking:

1. **SciPy `eigsh(LinearOperator)` first**: establishes the benefit of
   determinant + matrix-free.
2. **PRIMME second**: best CPU-side production candidate for "few smallest
   eigenpairs + preconditioning + warm starts".
3. **SLEPc/slepc4py** if you want MPI/distributed scaling or a solver lab.
   SLEPc is designed for large sparse eigenproblems, and its EPS module
   includes Krylov-Schur, Jacobi-Davidson, generalized Davidson, RQCG,
   and LOBPCG.
4. **SciPy/CuPy LOBPCG** as an experiment, not the first production
   choice. SciPy's LOBPCG accepts `LinearOperator`/callables and a
   preconditioner, and CuPy also exposes LOBPCG.
5. **ChASE: probably no**. ChASE is for dense Hermitian/symmetric
   eigenproblems and sequences of dense eigenproblems; that is the
   opposite of the sparse/matrix-free FCI structure here.

One correction to your prompt: the attached file's `GPU=True` path does
appear to call `cupyx.scipy.sparse.linalg.eigsh`, then transfers eigenpairs
back to host with `.get()`. CuPy's current docs also list `eigsh` and say
it accepts a CuPy ndarray, sparse matrix, or CuPy `LinearOperator`. That
still does not make "GPU eigsh" the first win; it is still Lanczos on the
wrong representation if you keep the current basis.

## Preconditioner direction

Start simple: a **Davidson/Jacobi diagonal preconditioner** in the
determinant basis,

$$
M^{-1}r \approx \frac{r}{\operatorname{diag}(H)-\theta},
$$

with clipping near zero denominators. It is cheap, exact-shape, CPU/GPU-
friendly, and easy to plug into PRIMME or LOBPCG.

Then test a stronger non-interacting preconditioner:

$$
H_0 = \sum_{\sigma,ij} h_{ij}\, c^\dagger_{i\sigma}c_{j\sigma}.
$$

In a Slater determinant basis built from one-particle eigenorbitals, $H_0$
is diagonal, so $H_0 - \theta I$ is an excellent conceptual preconditioner.
The complication is that your natural interaction is diagonal in the grid
basis, not the orbital basis, so a full basis transform may cost more than
it saves. I would prototype the diagonal grid-basis preconditioner first,
then compare.

Warm starts are worth adding but are not the main win. For single systems,
use the non-interacting Slater determinant as `v0`. For parameter sweeps,
disorder ensembles, grid convergence, or TDDFT continuation, use the
previous converged state/subspace. PRIMME is better than SciPy ARPACK here
because it exposes richer initial-guess and stats controls.

## Symmetry exploitation

The priority order should be:

1. **Same-spin antisymmetry / fixed-spin determinant basis**. This is the
   big one and is exact for every system.
2. **Spatial parity**, gated by detection. If `v_ext[i] == v_ext[N-1-i]`
   and `v_int[i,j] == v_int[N-1-i,N-1-j]`, then parity commutes with $H$.
   In determinant basis, parity maps each occupied tuple to the reflected
   tuple. Build even/odd orbit representatives and solve one or both
   blocks.
3. **Total spin ($S^2$)** only later. Spin-adapted configuration state
   functions can reduce the fixed-$S_z$ determinant space, but
   implementation complexity is much higher, and the current API
   constructs a spin product from `s.electrons`. For 3–4 electrons, parity
   plus determinant basis will likely give better ROI.

For `k=0` and a symmetric potential, the ground state is often even, but I
would not hard-code that if exactness is the goal. For robust
`solve(system, k=...)`, solve enough eigenpairs in both parity blocks and
merge them by energy.

## Subtle structure you can exploit

The interaction is diagonal in the grid basis. The current implementation
builds it through an `exp → einsum → log` trick to sum pair interactions
over the full tensor. In the matrix-free operator, do not preserve that
route. Precompute a real diagonal vector directly by summing pair
potentials. It is simpler, avoids unnecessary `exp/log`, and avoids dense
$N^{N_e}$ temporary construction in the labelled basis.

The one-body part is a Kronecker sum. In the full tensor representation,
that means stencil application along each electron axis. In the
determinant representation, it means sparse single-particle hops inside
each spin determinant channel. Both are better than materializing the
global CSR.

The predefined softened interaction is a function of `abs(x[i]-x[j])`, so
on uniform grids it is symmetric and Toeplitz-like. That helps with
generating/compressing the interaction diagonal, but it does **not** turn
the many-body interaction matvec into an FFT convolution; in the many-body
Hamiltonian it is just a diagonal multiply.

## Red flags I would avoid

Do not spend much time tuning `_estimate_level()`. It is compensating for
solving in the wrong space. Once you solve directly in the physical
determinant sector, `level` becomes `k+1` or a small block buffer.

Do not make dense GPU diagonalization a general path. Even the 2-electron
`N=80` case has a 6400×6400 dense matrix; dense eigensolvers are only
attractive if you need a large fraction of the spectrum, not one or a few
eigenpairs.

Do not start with total-spin ($S^2$) machinery. It is elegant, but
fixed-spin determinants get most of the exactness and dimension reduction
with far less risk.

Do not expect unpreconditioned LOBPCG to be a free win. LOBPCG shines with
a good block size and a good preconditioner; otherwise Lanczos/ARPACK can
be hard to beat for one smallest eigenpair.

Do not benchmark only wall time. Add a counting `LinearOperator` and
record setup time, matvec count, matvec time, orthogonalization time if
the solver exposes it, memory peak, residual norm, energy, and density
error. The tests use very tight energy/density tolerances, so validate
numerical equivalence before judging speed.

## Expected payoff

For your current 4-electron example, just moving from labelled tensor
space to fixed-spin determinants cuts dimension by about 4×. Solving only
the physical eigenpairs instead of computing surplus labelled-space
eigenvectors gives another meaningful reduction. Matrix-free application
removes CSR construction and reduces memory bandwidth. Parity can give
another ~2× on symmetric systems. PRIMME plus a diagonal/non-interacting
preconditioner is the next multiplicative lever.

My practical expectation: **determinant-basis matrix-free + parity** is
the path to a reliable 5–10×. **Add PRIMME with a useful preconditioner
and warm starts**, and 20×+ becomes plausible on repeated workloads. A
100× result is more likely from the combination of determinant basis,
parity, avoiding `fulls` materialization for intermediate eigenvectors,
GPU matrix-free kernels, and continuation/warm-started ensembles than
from swapping `eigsh` for one different solver alone.

---

## Recommended execution order (distilled)

1. **Determinant basis matrix-free `LinearOperator`** in fixed-$S_z$
   sector — dimension cut, removes antisymmetrize-after-solve, removes
   `_estimate_level` overprovisioning.
2. **SciPy `eigsh(LinearOperator)`** as control — isolates representation
   win from solver win.
3. **Spatial parity blocks** when detected — ~2× on symmetric `V_ext`.
4. **PRIMME** with **diagonal Jacobi preconditioner** in determinant
   basis.
5. **Warm-started initial vector** from non-interacting Slater
   determinant; for sweeps, from previous solution.
6. **GPU matrix-free kernels** (only after CPU determinant-basis works) —
   dense state vector, structured sparse hops.
7. **Counting LinearOperator** infrastructure for proper benchmarks
   (matvec count, residual, energy, density error) — set up before solver
   shopping.

Do **not** start with Davidson/LOBPCG, do **not** tune `_estimate_level`,
do **not** reach for ChASE.
