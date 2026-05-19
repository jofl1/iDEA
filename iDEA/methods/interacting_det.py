"""Determinant-basis matrix-free FCI solver for iDEA.

Phase B replacement for ``iDEA.methods.interacting``. Solves the interacting
Schrödinger equation directly in the fixed-spin Slater-determinant basis
via a matrix-free ``scipy.sparse.linalg.LinearOperator`` + ``eigsh``.

The basis is the Cartesian product of sorted-occupation tuples for the up
and down spin channels — dimension ``C(N, N_up) * C(N, N_down)`` per the
Pauli-allowed sector. This eliminates the labelled $N^{N_e}$ tensor
representation and the post-solve antisymmetrize pass used in
``iDEA.methods.interacting``.

See ``notes/friend_optimization_advice.md`` for the rationale. The
expected speedup is the dimension cut (4-70x for 3-4 electron cases) plus
removal of the antisymmetrize-after-solve overhead.

Public API (Phase B scope: energy and density only):

    state = iDEA.methods.interacting_det.solve(s, k=0)
    n = iDEA.methods.interacting_det.density(s, state)
"""

import itertools
import os

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import iDEA.methods.non_interacting
import iDEA.state
import iDEA.system

if os.environ.get("IDEA_DET_DISABLE_PRIMME", "").strip():
    # Audit-mode hook: treat PRIMME as absent even if installed, so the
    # scipy fallback path can be exercised end-to-end without uninstalling
    # the optional dependency. See A6 in
    # notes/second_friend_followup.md.
    primme = None
    _HAS_PRIMME = False
else:
    try:
        import primme

        _HAS_PRIMME = True
    except ImportError:
        _HAS_PRIMME = False


name = "interacting_det"
_EIGSH_BACKENDS = ("scipy", "primme")


def _get_primme_min_dim() -> int:
    """Dimension below which we fall back to scipy regardless of PRIMME.

    PRIMME's Jacobi-Davidson defaults have higher per-iteration overhead
    than ARPACK Lanczos for tiny problems. Default 2000, override via
    environment variable ``IDEA_DET_PRIMME_MIN_DIM``.
    """
    return int(os.environ.get("IDEA_DET_PRIMME_MIN_DIM", "2000"))


def _validated_backend_name(backend: str, source: str) -> str:
    resolved = str(backend).strip().lower()
    if resolved not in _EIGSH_BACKENDS:
        expected = ", ".join(repr(name) for name in _EIGSH_BACKENDS)
        raise ValueError(
            f"Unsupported eigensolver backend {backend!r} from {source}; "
            f"expected one of: {expected}."
        )
    if resolved == "primme" and not _HAS_PRIMME:
        raise ImportError(
            "PRIMME eigensolver backend was requested, but the 'primme' "
            "package is not installed. Install the optional fast dependency "
            "or choose backend='scipy'."
        )
    return resolved


def _resolve_backend(D: int, explicit: str = None) -> str:
    """Pick eigensolver backend. Precedence: explicit > env > auto."""
    if explicit is not None:
        return _validated_backend_name(explicit, "explicit backend argument")
    env = os.environ.get("IDEA_DET_EIGSH_BACKEND")
    if env:
        return _validated_backend_name(env, "IDEA_DET_EIGSH_BACKEND")
    if _HAS_PRIMME and D >= _get_primme_min_dim():
        return "primme"
    return "scipy"


def _solve_with_preconditioner(
    op,
    k: int = 1,
    tol: float = 0.0,
    diag_for_prec: np.ndarray = None,
    v0: np.ndarray = None,
    backend: str = None,
):
    """Smallest-algebraic eigenpair solve via PRIMME or scipy.

    Both ``_solve_in_parity_block`` and the full-basis fallback in
    ``solve`` route through this helper so backend selection,
    preconditioner construction (Phase E2), and warm-start v0
    plumbing (Phase E3) live in one place.

    | Args:
    |     op: scipy ``LinearOperator`` (sparse matrix also works).
    |     k: number of eigenpairs to return.
    |     tol: solver tolerance. ``0`` means scipy machine precision;
    |         remapped to ``1e-15`` for PRIMME (whose ``tol=0`` means
    |         the library's looser default).
    |     diag_for_prec: full-basis diagonal of ``op`` for the Jacobi
    |         preconditioner. Used by Phase E2 (PRIMME only); ignored
    |         in Phase E1 even if supplied.
    |     v0: initial guess for the eigensolve (Phase E3).
    |     backend: explicit override; ``"primme"`` or ``"scipy"``. If
    |         ``None``, resolves via env var ``IDEA_DET_EIGSH_BACKEND``
    |         then auto-selects (PRIMME if installed and ``op.shape[0]
    |         >= IDEA_DET_PRIMME_MIN_DIM``, else scipy).

    | Returns:
    |     ``(energies, eigvecs)`` with energies ascending and
    |     ``eigvecs.shape == (D, k)``, matching scipy's column-major
    |     convention.
    """
    D = op.shape[0]
    resolved = _resolve_backend(D, backend)

    if resolved == "primme":
        # PRIMME's tol=0 falls back to its loose default (sqrt of machine
        # epsilon, ~1.5e-8). Our existing scipy call sites use tol=0 to
        # mean machine precision; map to a tighter value so PRIMME
        # doesn't silently relax the convergence criterion. Upstream
        # tests bracket the acceptable range:
        #   - test_manybody.py::TestShort: density tol 1.2e-11
        #   - test_manybody.py::TestLong:  density tol 2.0e-13
        # An eigenpair converged at tol=t has residual ~|max_eig|*t,
        # which feeds back into density precision roughly linearly.
        # 1e-15 is the empirically validated default used for the
        # transparent fast path; looser values can fail the long density
        # tests' analytical-comparison tolerances.
        primme_tol = tol if tol > 0 else 1e-15
        v0_arg = v0.reshape(-1, 1) if v0 is not None else None
        # Phase E2 will construct a Jacobi preconditioner from
        # diag_for_prec here. Phase E1 leaves it unused.
        prec_op = None
        energies, eigvecs = primme.eigsh(
            op, k=k, which="SA", tol=primme_tol, v0=v0_arg, OPinv=prec_op,
            raise_for_unconverged=True,
        )
    else:
        # scipy.sparse.linalg.eigsh accepts v0 as a starting vector; pass
        # it through so warm-starts work in the scipy-backend regime
        # too (e.g. parity-block dims below the PRIMME threshold).
        # Pass v0 only when non-None — explicit v0=None and omitting v0
        # take slightly different ARPACK RNG trajectories, which can
        # shift convergence on ill-conditioned cases (e.g. potentials
        # dominated by a constant offset).
        eigsh_kwargs = {"k": k, "which": "SA", "tol": tol}
        if v0 is not None:
            eigsh_kwargs["v0"] = v0
        energies, eigvecs = spsla.eigsh(op, **eigsh_kwargs)

    order = np.argsort(energies)
    energies = energies[order]
    eigvecs = eigvecs[:, order]

    # Vector orientation alignment. For continuation across related solves
    # (parameter sweeps, time evolution, repeated KS iterations) a sign
    # flip in the returned eigenvector adds spurious noise to raw .full
    # comparisons. When v0 is supplied — i.e. the caller has a prior or
    # cold-start vector in hand — align the lowest eigenvector's sign so
    # ``vdot(v0, eigvec_0)`` is non-negative.
    if v0 is not None and eigvecs.shape[1] >= 1:
        overlap = complex(np.vdot(v0.ravel(), eigvecs[:, 0]))
        if overlap.real < 0:
            eigvecs[:, 0] = -eigvecs[:, 0]

    return energies, eigvecs


def _slater_amplitudes(phi_occ: np.ndarray, combs):
    """Slater-determinant minors of ``phi_occ`` at grid sites in ``combs``.

    For an M-orbital channel with M = N_sigma electrons, the amplitude
    for grid-combination tuple I is

        amp_I = det( phi_occ[ list(I), : ] ),

    i.e. the determinant of the M×M submatrix of the orbital coefficient
    matrix at the grid sites listed in I. Returned as a length-D array.

    For the vacuum channel (M = N_sigma = 0) the single basis state has
    amplitude 1. Implementation builds the ``(D, M, M)`` stacked
    submatrix tensor and dispatches to ``np.linalg.det``'s batched
    routine — Python-loop ``np.linalg.det`` per comb is too slow for
    larger cases.
    """
    if not combs or not combs[0]:
        return np.ones(len(combs), dtype=float)
    indices = np.array(combs, dtype=np.intp)  # (D, M)
    sub_matrices = phi_occ[indices, :]  # (D, M, M)
    return np.linalg.det(sub_matrices)


def _noninteracting_slater_det(
    s: "iDEA.system.System", up_combs, down_combs
) -> np.ndarray:
    """Non-interacting ground-state Slater determinant in grid-det basis.

    Solves the single-particle Hamiltonian ``K + diag(v_ext)`` (no
    Hartree, no exchange), picks the lowest ``N_up`` orbitals for the
    up channel and the lowest ``N_down`` for down, and expands the
    product Slater determinant as Slater minors per channel.

    Returns a ``(D_up, D_down)`` amplitude array, Euclidean-normalised.
    Used as the cold-start ``v0`` for the PRIMME eigensolve.
    """
    K = iDEA.methods.non_interacting.kinetic_energy_operator(s)
    H_sp = K + np.diag(s.v_ext)
    _energies, phi = np.linalg.eigh(H_sp)
    up_amps = _slater_amplitudes(phi[:, : s.up_count], up_combs)
    down_amps = _slater_amplitudes(phi[:, : s.down_count], down_combs)
    amps = up_amps[:, None] * down_amps[None, :]
    nrm = float(np.linalg.norm(amps))
    if nrm > 0:
        amps = amps / nrm
    return amps


def _build_basis(n_grid: int, n_per_spin: int):
    """Sorted-tuple combinations and rank map for one spin channel.

    For ``n_per_spin == 0`` returns the trivial "vacuum" basis: a single
    empty tuple. This lets the matvec uniformly treat both spin channels.
    """
    if n_per_spin == 0:
        return [()], {(): 0}
    combs = list(itertools.combinations(range(n_grid), n_per_spin))
    rank = {c: i for i, c in enumerate(combs)}
    return combs, rank


def _fermionic_sign(comb, i_old, i_new):
    """Sign for hopping an electron from ``i_old`` to ``i_new`` in ``comb``.

    The sign is ``(-1)**(number of occupied sites strictly between i_old
    and i_new)`` — i.e. the parity of permutations needed to restore
    sorted order after removing ``i_old`` and inserting ``i_new``.
    """
    lo = i_old if i_old < i_new else i_new
    hi = i_new if i_old < i_new else i_old
    count = 0
    for k in comb:
        if lo < k < hi:
            count += 1
    return -1.0 if count & 1 else 1.0


def _build_one_body_hops(combs, rank, K, n_grid):
    """Sparse ``(D, D)`` matrix of one-body hops for one spin channel.

    For each determinant ``comb`` and each off-diagonal kinetic entry
    ``K[j, i]`` (with ``i in comb``, ``j not in comb``, ``j != i``),
    accumulate the matrix element ``K[j, i] * sign`` into row ``new_idx``
    column ``src_idx``.
    """
    D = len(combs)
    if D == 1 and len(combs[0]) == 0:
        return sps.csr_matrix((1, 1))

    rows = []
    cols = []
    data = []

    for src_idx, comb in enumerate(combs):
        comb_set = set(comb)
        for i_old in comb:
            row_K = K[:, i_old]
            for j_new in range(n_grid):
                if j_new == i_old or j_new in comb_set:
                    continue
                coeff = row_K[j_new]
                if coeff == 0.0:
                    continue
                new_comb_set = (comb_set - {i_old}) | {j_new}
                new_comb = tuple(sorted(new_comb_set))
                dst_idx = rank.get(new_comb)
                if dst_idx is None:
                    continue
                sign = _fermionic_sign(comb, i_old, j_new)
                rows.append(dst_idx)
                cols.append(src_idx)
                data.append(coeff * sign)

    return sps.csr_matrix((data, (rows, cols)), shape=(D, D))


def _build_diagonal(up_combs, down_combs, v_ext, v_int, h_diag):
    """Diagonal matrix elements ``<u,d|H|u,d>`` as a ``(D_up, D_down)`` array.

    Decomposes into:
      one-body: sum over electrons of ``h_diag[i] + v_ext[i]``
      same-spin pair interaction: sum over (a,b in same comb, a<b) of ``v_int[a,b]``
      cross-spin interaction: sum over (i in up, j in down) of ``v_int[i,j]``
        (all i, j including i==j — same site opposite spin contributes v_int[i,i])
    """
    D_up = len(up_combs)
    D_down = len(down_combs)

    up_one_body = np.zeros(D_up)
    up_pair_int = np.zeros(D_up)
    for u_idx, comb in enumerate(up_combs):
        if not comb:
            continue
        sites = np.array(comb)
        up_one_body[u_idx] = float(np.sum(h_diag[sites] + v_ext[sites]))
        for a, b in itertools.combinations(comb, 2):
            up_pair_int[u_idx] += v_int[a, b]

    down_one_body = np.zeros(D_down)
    down_pair_int = np.zeros(D_down)
    for d_idx, comb in enumerate(down_combs):
        if not comb:
            continue
        sites = np.array(comb)
        down_one_body[d_idx] = float(np.sum(h_diag[sites] + v_ext[sites]))
        for a, b in itertools.combinations(comb, 2):
            down_pair_int[d_idx] += v_int[a, b]

    diag = (
        up_one_body[:, None]
        + down_one_body[None, :]
        + up_pair_int[:, None]
        + down_pair_int[None, :]
    )

    # Cross-spin interaction. For each (up_comb, down_comb) pair, sum
    # v_int[i, j] over all i in up_comb and j in down_comb. Vectorise per
    # up_idx so the inner sum is one numpy call.
    if up_combs and up_combs[0] and down_combs and down_combs[0]:
        for u_idx, up_comb in enumerate(up_combs):
            up_sites = np.array(up_comb)
            v_int_slice = v_int[np.ix_(up_sites, np.arange(v_int.shape[1]))]
            for d_idx, down_comb in enumerate(down_combs):
                down_sites = np.array(down_comb)
                diag[u_idx, d_idx] += float(np.sum(v_int_slice[:, down_sites]))

    return diag


def _build_indicator(combs, n_grid):
    """Sparse ``(D, N)`` indicator: row ``d_idx`` has 1s at columns ``combs[d_idx]``."""
    D = len(combs)
    if D == 1 and len(combs[0]) == 0:
        return sps.csr_matrix((1, n_grid))
    rows = []
    cols = []
    for d_idx, comb in enumerate(combs):
        for i in comb:
            rows.append(d_idx)
            cols.append(i)
    data = np.ones(len(rows), dtype=float)
    return sps.csr_matrix((data, (rows, cols)), shape=(D, n_grid))


class _InvariantComponents:
    """Sweep-invariant precomputed structures shared by all solves on a
    fixed (x, dx, electrons, stencil) system.

    These structures depend ONLY on the grid geometry, the electron
    counts, and the finite-difference stencil — NOT on v_ext or v_int.
    A SolveContext that iterates over a sweep (varying v_ext, fixed
    everything else) can build this once and reuse it for every solve.

    Phase S3.1a factoring step: the previous monolithic
    ``_build_solver_components`` is now ``_build_invariant_components``
    plus ``_build_op_for_potential``. The composer that produces the
    full ``_SolverComponents`` lives below.
    """

    __slots__ = (
        "up_combs",
        "down_combs",
        "up_rank",
        "down_rank",
        "up_hops",
        "down_hops",
        "up_indicator",
        "down_indicator",
        "D",
        "D_up",
        "D_down",
        "h_diag",
        "n_grid",
        "n_up",
        "n_down",
        # Lazy parity caches. _build_parity_orbits reads only fields on
        # this class, so we can populate it lazily without rebuilding
        # per solve. Both ± projectors are cached independently because
        # solve() may want one or both per call.
        "_parity_orbits_cache",
        "_parity_projector_pos_cache",
        "_parity_projector_neg_cache",
    )

    def parity_orbits(self):
        """Return ``(s_ref, partner, self_mask, reps_pos, reps_neg)``.

        Built once and cached on the invariants object; the result
        depends only on the basis geometry and electron counts, so it
        is reusable across any number of solves on this template.
        """
        cached = getattr(self, "_parity_orbits_cache", None)
        if cached is None:
            cached = _build_parity_orbits(
                self, self.n_grid, self.n_up, self.n_down
            )
            self._parity_orbits_cache = cached
        return cached

    def parity_projector(self, p: int):
        """Return the sparse projector ``Q_p`` for total parity ``p``.

        Built lazily on first access per sign. ``p`` must be ``+1`` or
        ``-1``.
        """
        if p == +1:
            cached = getattr(self, "_parity_projector_pos_cache", None)
            if cached is None:
                s_ref, partner, self_mask, reps_pos, _reps_neg = (
                    self.parity_orbits()
                )
                cached = _build_parity_projection(
                    reps_pos, partner, self_mask, s_ref, +1, self.D
                )
                self._parity_projector_pos_cache = cached
            return cached
        if p == -1:
            cached = getattr(self, "_parity_projector_neg_cache", None)
            if cached is None:
                s_ref, partner, self_mask, _reps_pos, reps_neg = (
                    self.parity_orbits()
                )
                cached = _build_parity_projection(
                    reps_neg, partner, self_mask, s_ref, -1, self.D
                )
                self._parity_projector_neg_cache = cached
            return cached
        raise ValueError(
            f"parity sign must be +1 or -1, got {p!r}"
        )


class _SolverComponents:
    """Precomputed structures shared between solve() and validation tests.

    Holds the per-spin determinant basis, the hop matrices, the diagonal,
    the assembled LinearOperator, and the indicator matrices used by the
    density helper. Tests reach for the LinearOperator via
    ``_build_solver_components(s).op`` to run Hermiticity and residual
    checks without re-implementing matvec.
    """

    __slots__ = (
        "op",
        "matvec",
        "D",
        "D_up",
        "D_down",
        "up_combs",
        "down_combs",
        "up_rank",
        "down_rank",
        "up_indicator",
        "down_indicator",
        "diag",
        "up_hops",
        "down_hops",
        # Back-reference to the invariants this _SolverComponents was
        # composed from. Lets solve() reach the lazy parity accessors
        # without rebuilding the orbits each call.
        "_inv",
    )


def _build_invariant_components(s: iDEA.system.System) -> _InvariantComponents:
    """Build the sweep-invariant components of the solver.

    Depends only on ``(s.x, s.dx, s.electrons, s.stencil)``. Reused
    across every solve in a sweep by ``SolveContext``.
    """
    n_grid = s.x.shape[0]
    n_up = s.up_count
    n_down = s.down_count

    up_combs, up_rank = _build_basis(n_grid, n_up)
    down_combs, down_rank = _build_basis(n_grid, n_down)
    D_up = len(up_combs)
    D_down = len(down_combs)
    D = D_up * D_down

    K = iDEA.methods.non_interacting.kinetic_energy_operator(s)
    h_diag = np.diag(K).copy()

    up_hops = _build_one_body_hops(up_combs, up_rank, K, n_grid)
    down_hops = _build_one_body_hops(down_combs, down_rank, K, n_grid)

    inv = _InvariantComponents()
    inv.up_combs = up_combs
    inv.down_combs = down_combs
    inv.up_rank = up_rank
    inv.down_rank = down_rank
    inv.up_hops = up_hops
    inv.down_hops = down_hops
    inv.up_indicator = _build_indicator(up_combs, n_grid)
    inv.down_indicator = _build_indicator(down_combs, n_grid)
    inv.D = D
    inv.D_up = D_up
    inv.D_down = D_down
    inv.h_diag = h_diag
    inv.n_grid = n_grid
    inv.n_up = n_up
    inv.n_down = n_down
    inv._parity_orbits_cache = None
    inv._parity_projector_pos_cache = None
    inv._parity_projector_neg_cache = None
    return inv


def _build_op_for_potential(inv: _InvariantComponents, v_ext, v_int):
    """Build the per-potential pieces (diagonal, matvec, LinearOperator).

    Reuses the invariant ``inv.up_hops``, ``inv.down_hops``,
    ``inv.h_diag`` and rebuilds only what changes when ``v_ext`` or
    ``v_int`` change. Returns ``(diag, matvec, op)`` so the caller can
    cache the diagonal, attach it to a ``_SolverComponents``, or
    feed it into a Jacobi preconditioner.
    """
    D = inv.D
    D_up = inv.D_up
    D_down = inv.D_down
    up_hops = inv.up_hops
    down_hops = inv.down_hops

    diag = _build_diagonal(inv.up_combs, inv.down_combs, v_ext, v_int, inv.h_diag)

    up_has_hops = up_hops.nnz > 0
    down_has_hops = down_hops.nnz > 0

    def matvec(psi_flat):
        psi = psi_flat.reshape(D_up, D_down)
        out = diag * psi
        if up_has_hops:
            out = out + up_hops @ psi
        if down_has_hops:
            out = out + (down_hops @ psi.T).T
        return out.ravel()

    op = spsla.LinearOperator(shape=(D, D), matvec=matvec, dtype=float)
    return diag, matvec, op


def _build_solver_components(s: iDEA.system.System) -> _SolverComponents:
    inv = _build_invariant_components(s)
    return _compose_solver_components(inv, s.v_ext, s.v_int)


def _compose_solver_components(
    inv: _InvariantComponents, v_ext, v_int
) -> _SolverComponents:
    """Build a ``_SolverComponents`` from cached invariants + potentials.

    Used by both ``_build_solver_components`` (single-shot) and
    ``SolveContext.solve`` (sweep, reusing ``inv`` across calls).
    """
    diag, matvec, op = _build_op_for_potential(inv, v_ext, v_int)

    comps = _SolverComponents()
    comps.op = op
    comps.matvec = matvec
    comps.D = inv.D
    comps.D_up = inv.D_up
    comps.D_down = inv.D_down
    comps.up_combs = inv.up_combs
    comps.down_combs = inv.down_combs
    comps.up_rank = inv.up_rank
    comps.down_rank = inv.down_rank
    comps.up_indicator = inv.up_indicator
    comps.down_indicator = inv.down_indicator
    comps.diag = diag
    comps.up_hops = inv.up_hops
    comps.down_hops = inv.down_hops
    comps._inv = inv
    return comps


def _v_ext_is_parity_symmetric(s: iDEA.system.System, atol: float = 1e-10) -> bool:
    """Absolute-difference reflection-symmetry check on ``v_ext``.

    Uses an explicit absolute tolerance (``rtol=0``) rather than
    ``np.allclose``'s default relative tolerance. A shifted potential
    such as ``v_ext + 1e6`` plus an asymmetric kink would otherwise
    pass the symmetry test with NumPy's default ``rtol=1e-5``, and the
    parity fast path would solve only one block of a Hamiltonian where
    parity is not actually conserved → wrong ground state.
    """
    return bool(np.allclose(s.v_ext, s.v_ext[::-1], atol=atol, rtol=0.0))


def _v_int_is_parity_symmetric(s: iDEA.system.System, atol: float = 1e-10) -> bool:
    """Absolute-difference reflection-symmetry check on ``v_int``.

    See ``_v_ext_is_parity_symmetric`` for the rtol rationale.
    """
    return bool(np.allclose(s.v_int, s.v_int[::-1, ::-1], atol=atol, rtol=0.0))


def _reflect_comb(comb, n_grid):
    return tuple(sorted(n_grid - 1 - i for i in comb))


def _reflection_sign(n_up: int, n_down: int) -> int:
    """epsilon_{N_up} * epsilon_{N_down} where epsilon_m = (-1)^(m(m-1)/2).

    Comes from reordering m reversed creation operators back into sorted
    order: removing each excess transposition contributes one fermionic
    sign. The friend's correction — easy to miss and the source of the
    'uu has odd ground state' result.
    """
    eps_up = -1 if (n_up * (n_up - 1) // 2) % 2 else 1
    eps_down = -1 if (n_down * (n_down - 1) // 2) % 2 else 1
    return eps_up * eps_down


def _predict_ground_parity(s: iDEA.system.System) -> int:
    """Predicted ground-state total parity per the friend's formula."""
    return _reflection_sign(s.up_count, s.down_count)


def _build_parity_orbits(comps: _SolverComponents, n_grid: int, n_up: int, n_down: int):
    """Build the joint-state parity orbit map and per-block representative lists.

    Returns:
        s_ref: int, sign of joint reflection = eps_{N_up} * eps_{N_down}.
        partner: int64 array of length D, partner[i] = joint index of the
            parity reflection of joint index i.
        self_mask: bool array, True where partner[i] == i (self-orbit).
        reps_pos, reps_neg: int64 arrays, orbit representatives in the
            +1 and -1 total-parity blocks respectively.

    For pair-orbits, the lower joint index of {i, partner[i]} is used as
    the canonical representative; it appears in both blocks (one even
    combination, one odd). For self-orbits, the orbit lives only in the
    block p = s_ref.
    """
    D_up = comps.D_up
    D_down = comps.D_down
    D = comps.D
    s_ref = _reflection_sign(n_up, n_down)

    up_ref = np.empty(D_up, dtype=np.int64)
    for u_idx, comb in enumerate(comps.up_combs):
        if comb:
            up_ref[u_idx] = comps.up_rank[_reflect_comb(comb, n_grid)]
        else:
            up_ref[u_idx] = 0

    down_ref = np.empty(D_down, dtype=np.int64)
    for d_idx, comb in enumerate(comps.down_combs):
        if comb:
            down_ref[d_idx] = comps.down_rank[_reflect_comb(comb, n_grid)]
        else:
            down_ref[d_idx] = 0

    partner = np.empty(D, dtype=np.int64)
    for u_idx in range(D_up):
        for d_idx in range(D_down):
            joint = u_idx * D_down + d_idx
            partner[joint] = up_ref[u_idx] * D_down + down_ref[d_idx]

    self_mask = partner == np.arange(D)

    reps_pos = []
    reps_neg = []
    seen = np.zeros(D, dtype=bool)
    for i in range(D):
        if seen[i]:
            continue
        j = int(partner[i])
        if i == j:
            if s_ref == 1:
                reps_pos.append(i)
            else:
                reps_neg.append(i)
            seen[i] = True
        else:
            rep = min(i, j)
            reps_pos.append(rep)
            reps_neg.append(rep)
            seen[i] = True
            seen[j] = True

    return (
        s_ref,
        partner,
        self_mask,
        np.array(reps_pos, dtype=np.int64),
        np.array(reps_neg, dtype=np.int64),
    )


def _build_parity_projection(reps, partner, self_mask, s_ref, p, D):
    """Sparse projector Q_p mapping a block-coordinate vector to full basis.

    For self-orbit reps i (only present when p == s_ref): Q_p[i, k] = 1.
    For pair-orbit reps i with partner j != i:
        Q_p[i, k] = 1/sqrt(2), Q_p[j, k] = p*s_ref/sqrt(2).
    """
    rows = []
    cols = []
    data = []
    sqrt2_inv = 1.0 / np.sqrt(2.0)
    for k, i in enumerate(reps):
        if self_mask[i]:
            rows.append(int(i))
            cols.append(k)
            data.append(1.0)
        else:
            j = int(partner[i])
            rows.append(int(i))
            cols.append(k)
            data.append(sqrt2_inv)
            rows.append(j)
            cols.append(k)
            data.append(p * s_ref * sqrt2_inv)
    return sps.csr_matrix((data, (rows, cols)), shape=(D, len(reps)), dtype=float)


def _project_to_block(v_full, Q_p, cutoff: float = 1e-12):
    """Project a full-basis vector into a parity-block, Euclidean-normalised.

    Returns ``None`` if ``v_full is None`` (no v0 to project), or if the
    projection norm is below ``cutoff`` (the input has the opposite
    parity from the block). The eigensolver then falls back to its
    default starting vector.

    ``cutoff`` defaults to ``1e-12`` for cold-start v0 (Slater
    determinant), which has unit norm only after the channel-amplitude
    outer product so projection-norm noise is comfortably below the
    cutoff. SolveContext raises this to ``1e-8`` for warm-starts: a
    full-basis warm-start eigvec has unit norm, and a misprojection
    onto the wrong parity block yields a residual ~O(eps · D) that can
    slip past ``1e-12`` for large bases.
    """
    if v_full is None:
        return None
    v_block = Q_p.T @ v_full
    nrm = float(np.linalg.norm(v_block))
    if nrm <= cutoff:
        return None
    return v_block / nrm


def _solve_in_parity_block(
    comps, Q_p, k=0, tol=0.0, v0=None, safety_extra_k: int = 0
):
    """Run eigsh on Q_p^T H Q_p (the friend's 'implicit projector' form).

    Returns (energies[: n], eigvecs in full basis[:, : n]) with
    ``n = k + 1 + safety_extra_k``. ``safety_extra_k > 0`` asks the
    eigensolver for additional eigenpairs beyond the caller's
    requested ``k``; used by ``SolveContext`` warm-starts to guard
    against converging to a higher-energy state preferentially when v0
    has greater overlap with it near a level crossing. Stock
    ``solve()`` leaves it at ``0``.
    """
    D_block = Q_p.shape[1]
    if D_block == 0:
        return np.array([], dtype=float), np.zeros((comps.D, 0))

    n_eig = k + 1 + safety_extra_k

    def matvec_block(y):
        return Q_p.T @ comps.matvec(Q_p @ y)

    if n_eig >= D_block - 1:
        dense = np.empty((D_block, D_block), dtype=float)
        for col in range(D_block):
            e = np.zeros(D_block)
            e[col] = 1.0
            dense[:, col] = matvec_block(e)
        energies, eigvecs = np.linalg.eigh(dense)
        eigvecs_full = Q_p @ eigvecs[:, :n_eig]
        return energies[:n_eig], eigvecs_full

    op_block = spsla.LinearOperator(
        shape=(D_block, D_block), matvec=matvec_block, dtype=float
    )
    energies, eigvecs = _solve_with_preconditioner(
        op_block, k=n_eig, tol=tol, v0=v0
    )
    eigvecs_full = Q_p @ eigvecs
    return energies, eigvecs_full


def solve(
    s: iDEA.system.System,
    k: int = 0,
    tol: float = 0.0,
    use_parity: bool = True,
    verify_parity: bool = False,
) -> iDEA.state.ManyBodyState:
    """Solve the interacting Schrödinger equation in determinant basis.

    | Args:
    |     s: iDEA.system.System, System object.
    |     k: int, Energy level to return (0 = ground state).
    |     tol: float, eigsh tolerance (0 = machine precision).
    |     use_parity: bool, opt out of parity block-diagonalisation
    |         (forces the full-basis solve; default True).
    |     verify_parity: bool, when True solves BOTH parity blocks and
    |         merges by energy (asserts the ground parity matches the
    |         friend's formula for ``k=0``). Default False uses the
    |         Phase C+ fast path: solve only the predicted ground-parity
    |         block. ``k > 0`` always forces the both-blocks path because
    |         we don't predict excited-state parities.

    | Returns:
    |     state: iDEA.state.ManyBodyState with .energy, .det_amplitudes,
    |     and .det_up_combs / .det_down_combs / .det_up_indicator /
    |     .det_down_indicator. When parity was used, .parity is set to
    |     +1 or -1 (the parity of the returned eigenstate). .full /
    |     .space / .spin are intentionally left as ArrayPlaceholder in
    |     Phase B/C; use iDEA.methods.interacting_det.density() for the
    |     density observable.
    """
    comps = _build_solver_components(s)
    D = comps.D
    D_up = comps.D_up
    D_down = comps.D_down

    can_use_parity = (
        use_parity
        and _v_ext_is_parity_symmetric(s)
        and _v_int_is_parity_symmetric(s)
    )

    # Cold-start v0: non-interacting Slater determinant in the
    # grid-determinant basis. Used as the PRIMME initial guess so the
    # solver doesn't have to find the ground-state subspace from a
    # random vector. Skip the construction when scipy will handle every
    # branch (PRIMME absent or operator dim below the threshold) since
    # scipy.eigsh doesn't accept v0 in this code path and the
    # Slater-determinant minors are wasted work.
    expected_solve_dim = (D // 2) if can_use_parity else D
    will_use_primme = _resolve_backend(expected_solve_dim) == "primme"
    if will_use_primme:
        v0_full = _noninteracting_slater_det(
            s, comps.up_combs, comps.down_combs
        ).ravel()
    else:
        v0_full = None

    if can_use_parity:
        # Fast path: solve only the predicted-parity block for k=0 unless
        # the caller asks to verify. k>0 always falls through to both
        # blocks because we have no prediction for excited-state parities.
        if (not verify_parity) and k == 0:
            predicted = _predict_ground_parity(s)
            Q_p = comps._inv.parity_projector(predicted)
            v0_block = _project_to_block(v0_full, Q_p)
            energies_p, eigvecs_p = _solve_in_parity_block(
                comps, Q_p, k=0, tol=tol, v0=v0_block,
            )
            eigval = float(energies_p[0])
            amplitudes = eigvecs_p[:, 0].reshape(D_up, D_down)
            selected_parity = predicted
        else:
            Q_pos = comps._inv.parity_projector(+1)
            Q_neg = comps._inv.parity_projector(-1)
            v0_pos = _project_to_block(v0_full, Q_pos)
            v0_neg = _project_to_block(v0_full, Q_neg)
            energies_pos, eigvecs_pos = _solve_in_parity_block(
                comps, Q_pos, k=k, tol=tol, v0=v0_pos,
            )
            energies_neg, eigvecs_neg = _solve_in_parity_block(
                comps, Q_neg, k=k, tol=tol, v0=v0_neg,
            )
            all_energies = np.concatenate([energies_pos, energies_neg])
            all_eigvecs = np.concatenate([eigvecs_pos, eigvecs_neg], axis=1)
            parities = np.concatenate(
                [np.full(len(energies_pos), +1), np.full(len(energies_neg), -1)]
            )
            order = np.argsort(all_energies)
            eigval = float(all_energies[order[k]])
            amplitudes = all_eigvecs[:, order[k]].reshape(D_up, D_down)
            selected_parity = int(parities[order[k]])

            if verify_parity and k == 0:
                predicted = _predict_ground_parity(s)
                assert selected_parity == predicted, (
                    f"ground-state parity mismatch: solver returned "
                    f"{selected_parity}, prediction "
                    f"(eps_Nup * eps_Ndown) is {predicted}"
                )
    else:
        # Full-basis fallback for asymmetric systems or use_parity=False.
        n_eig = k + 1
        if n_eig >= D - 1:
            dense = np.empty((D, D), dtype=float)
            for col in range(D):
                e = np.zeros(D)
                e[col] = 1.0
                dense[:, col] = comps.matvec(e)
            energies, eigvecs = np.linalg.eigh(dense)
            eigvecs = eigvecs[:, :n_eig]
            energies = energies[:n_eig]
        else:
            energies, eigvecs = _solve_with_preconditioner(
                comps.op, k=n_eig, tol=tol, v0=v0_full,
            )
        eigval = float(energies[k])
        amplitudes = eigvecs[:, k].reshape(D_up, D_down)
        selected_parity = 0  # parity unknown / not applicable

    state = iDEA.state.ManyBodyState(energy=eigval)
    state.det_amplitudes = amplitudes
    state.det_up_combs = comps.up_combs
    state.det_down_combs = comps.down_combs
    state.det_up_indicator = comps.up_indicator
    state.det_down_indicator = comps.down_indicator
    state.parity = selected_parity
    return state


def density(s: iDEA.system.System, state: iDEA.state.ManyBodyState) -> np.ndarray:
    """Total density from determinant amplitudes.

    .. math::
       n(x_i) = \\frac{1}{\\Delta x}
                \\left[\\sum_{I_\\uparrow \\ni i}
                       \\sum_{I_\\downarrow} |c_{I_\\uparrow, I_\\downarrow}|^2
                     + \\sum_{I_\\downarrow \\ni i}
                       \\sum_{I_\\uparrow} |c_{I_\\uparrow, I_\\downarrow}|^2
                \\right]

    The indicator matrices are built once at solve time and cached on the
    state object.
    """
    amps = state.det_amplitudes
    abs2 = np.abs(amps) ** 2
    up_indicator = state.det_up_indicator
    down_indicator = state.det_down_indicator
    n_up = (up_indicator.T @ abs2.sum(axis=1)) / s.dx
    n_down = (down_indicator.T @ abs2.sum(axis=0)) / s.dx
    return np.asarray(n_up + n_down).ravel()


def build_labelled_state(
    s: iDEA.system.System, det_state: iDEA.state.ManyBodyState
) -> iDEA.state.ManyBodyState:
    """Reconstruct .space, .spin, .full from determinant amplitudes.

    Plants each amplitude at its canonical labelled position then reuses
    ``iDEA.methods.interacting.antisymmetrize`` to fill in all permutations,
    normalize, and dedupe. Returns a new ManyBodyState with all the
    attributes the original labelled solver provides — so existing
    ``iDEA.observables`` code that indexes ``state.full`` continues to work
    transparently.

    The per-channel determinant amplitudes from ``det_state`` are
    preserved on the returned state as ``det_amplitudes``,
    ``det_up_combs``, ``det_down_combs``, ``det_up_indicator``,
    ``det_down_indicator``, and ``parity`` for callers who want the
    cheaper determinant-form observables.

    | Args:
    |     s: iDEA.system.System, the system the det_state was solved on.
    |     det_state: iDEA.state.ManyBodyState with .det_amplitudes etc.
    |         populated by ``interacting_det.solve``.

    | Returns:
    |     state: iDEA.state.ManyBodyState with .energy, .space, .spin,
    |     .full all populated, plus the original det attributes.
    """
    import string
    from math import factorial
    import iDEA.methods.interacting as _labelled

    n_grid = s.x.shape[0]
    n_e = s.count
    electrons = s.electrons

    up_axes = [i for i, c in enumerate(electrons) if c == "u"]
    down_axes = [i for i, c in enumerate(electrons) if c == "d"]

    # Plant each det amplitude at the canonical labelled position. Each
    # (I_up, I_down) maps to exactly one labelled position; antisymmetrize
    # handles the permutations.
    spaces_single = np.zeros((n_grid,) * n_e, dtype=det_state.det_amplitudes.dtype)
    for u_idx, up_comb in enumerate(det_state.det_up_combs):
        for d_idx, down_comb in enumerate(det_state.det_down_combs):
            c = det_state.det_amplitudes[u_idx, d_idx]
            if c == 0:
                continue
            coords = [0] * n_e
            for axis, idx in zip(up_axes, up_comb):
                coords[axis] = idx
            for axis, idx in zip(down_axes, down_comb):
                coords[axis] = idx
            spaces_single[tuple(coords)] = c
    spaces = spaces_single[..., np.newaxis]

    # Spins tensor: same construction the existing labelled solver uses.
    symbols = string.ascii_lowercase
    spin_basis = [
        np.array([1.0, 0.0]) if c == "u" else np.array([0.0, 1.0])
        for c in electrons
    ]
    spin_letters = symbols[:n_e]
    spin_subscripts = ",".join(spin_letters) + "->" + "".join(spin_letters)
    spin = np.einsum(spin_subscripts, *spin_basis)
    spins = np.zeros((2,) * n_e + (1,), dtype=float)
    spins[..., 0] = spin

    energies = np.array([det_state.energy])

    fulls, spaces_out, spins_out, energies_out = _labelled.antisymmetrize(
        s, spaces, spins, energies
    )

    # antisymmetrize() normalises .full but does NOT renormalise or
    # re-antisymmetrise spaces (it returns the input, filtered/cropped).
    # Legacy convention is .space = sqrt(M) * .full[canonical spin slice]
    # where M = N_e! / (N_up! * N_down!) is the number of distinct spin
    # arrangements with these counts. For uu (M=1) the canonical slice
    # IS .space; for ud (M=2) it is .space / sqrt(2); etc.
    spin_index = {"u": 0, "d": 1}
    canonical_indexer = []
    for c in s.electrons:
        canonical_indexer.append(slice(None))
        canonical_indexer.append(spin_index[c])
    canonical_indexer = tuple(canonical_indexer)

    spin_mult = factorial(s.count) / (
        factorial(s.up_count) * factorial(s.down_count)
    )
    space = np.sqrt(spin_mult) * fulls[canonical_indexer + (0,)]

    state = iDEA.state.ManyBodyState(
        space=space,
        spin=spins_out[..., 0],
        full=fulls[..., 0],
        energy=float(energies_out[0]),
    )
    state.det_amplitudes = det_state.det_amplitudes
    state.det_up_combs = det_state.det_up_combs
    state.det_down_combs = det_state.det_down_combs
    state.det_up_indicator = det_state.det_up_indicator
    state.det_down_indicator = det_state.det_down_indicator
    state.parity = getattr(det_state, "parity", 0)
    return state


class SolveContext:
    """Explicit warm-start context for sweeps over related Systems.

    Caches the sweep-invariant solver setup work — basis, hop matrices,
    indicators, kinetic diagonal, and (lazily) the parity orbits and
    projectors — and warm-starts each subsequent eigensolve from the
    previous solve's full-basis eigenvector. Compatible sweep elements
    share ``x``, ``dx``, ``stencil``, ``electrons`` and ``v_int`` with
    the template; only ``v_ext`` varies.

    Single-threaded — create one context per worker if parallelising.
    Not a hash-keyed cache: the snapshot is taken at construction time,
    and every ``solve(s)`` call validates the incoming system against
    that snapshot.

    Example:

        ctx = iDEA.methods.interacting_det.SolveContext(s_template)
        for s in disordered_systems:
            state = ctx.solve(s)

    Each call returns a ``ManyBodyState`` matching
    ``interacting_det.solve``'s return type. Use
    ``ctx.reset()`` to drop the warm-start eigvec.

    Only ``k == 0`` benefits from warm-start; calls with ``k > 0`` fall
    back to cold-start v0 (Slater determinant) for that solve.

    Future Phase E2 (Jacobi precond) hook: when ``diag_for_prec`` is
    wired up to ``_solve_with_preconditioner``, this class should pass
    the freshly-built ``diag`` rather than rebuilding it.
    """

    def __init__(self, s_template: "iDEA.system.System"):
        # Defensive copies on mutable arrays so a caller mutating the
        # template post-construction cannot poison the snapshot
        # silently. Immutables (dx, stencil, electrons) stored
        # directly.
        self._template_x = np.array(s_template.x, copy=True)
        self._template_dx = float(s_template.dx)
        self._template_stencil = int(s_template.stencil)
        self._template_electrons = str(s_template.electrons)
        self._template_v_int = np.array(s_template.v_int, copy=True)
        self._inv = _build_invariant_components(s_template)
        # Predicted ground-state parity is invariant under v_ext
        # (depends only on N_up, N_down via _predict_ground_parity).
        self._predicted_parity = _predict_ground_parity(s_template)
        # Warm-start eigvec, always stored in FULL-BASIS coords. The
        # parity-block path recovers it via ``Q_p @ eigvec_block``
        # before caching. Storing block coords would break the next
        # solve if symmetry changes mid-sweep.
        self._last_eigvec_full = None

    def reset(self):
        """Drop the cached warm-start eigvec. Next solve cold-starts."""
        self._last_eigvec_full = None

    def _validate(self, s: "iDEA.system.System"):
        if not np.array_equal(s.x, self._template_x):
            raise ValueError("SolveContext template mismatch: x")
        if float(s.dx) != self._template_dx:
            raise ValueError("SolveContext template mismatch: dx")
        if int(s.stencil) != self._template_stencil:
            raise ValueError("SolveContext template mismatch: stencil")
        if str(s.electrons) != self._template_electrons:
            raise ValueError("SolveContext template mismatch: electrons")
        if not np.array_equal(s.v_int, self._template_v_int):
            raise ValueError("SolveContext template mismatch: v_int")

    def solve(
        self,
        s: "iDEA.system.System",
        k: int = 0,
        tol: float = 0.0,
        use_parity: bool = True,
        verify_parity: bool = False,
    ) -> "iDEA.state.ManyBodyState":
        """Solve for ``s`` reusing cached invariants and the last eigvec.

        Mirrors ``interacting_det.solve``'s dispatch. The first call
        (or any call with ``k > 0``, or any call after ``reset()``)
        cold-starts the eigensolver. Subsequent calls with ``k == 0``
        warm-start from the previously-cached full-basis eigvec.
        """
        self._validate(s)
        inv = self._inv
        comps = _compose_solver_components(inv, s.v_ext, s.v_int)
        D = inv.D
        D_up = inv.D_up
        D_down = inv.D_down

        can_use_parity = (
            use_parity
            and _v_ext_is_parity_symmetric(s)
            and _v_int_is_parity_symmetric(s)
        )

        # Warm-start is only useful for k=0 (we cache only the lowest
        # eigvec). For k > 0 fall through to stock cold-start logic so
        # the eigensolver isn't biased by a poorly-matching v0.
        warm_starting = (k == 0) and (self._last_eigvec_full is not None)

        if warm_starting:
            v0_full = self._last_eigvec_full
        else:
            # Mirror stock solve()'s cold-start v0 gate exactly so
            # single-call equivalence holds bit-for-bit.
            expected_solve_dim = (D // 2) if can_use_parity else D
            will_use_primme = (
                _resolve_backend(expected_solve_dim) == "primme"
            )
            if will_use_primme:
                v0_full = _noninteracting_slater_det(
                    s, comps.up_combs, comps.down_combs
                ).ravel()
            else:
                v0_full = None

        # See _project_to_block for the cutoff rationale.
        projection_cutoff = 1e-8 if warm_starting else 1e-12
        # No safety_extra_k on warm-start. We initially asked for k+1
        # eigenpairs as a guard against ARPACK/PRIMME mis-converging to
        # a higher state with which v0 has greater overlap. In practice
        # the extra eigenpair caused PRIMME to "converge" a second
        # eigenpair to a near-zero ghost eigenvalue, which leaked back
        # through the lowest-energy pick and returned wrong energies on
        # uud_40-scale problems. ARPACK's which="SA" semantics already
        # post-filter to the smallest-algebraic eigenpair, so the guard
        # was redundant for scipy and actively harmful for PRIMME.
        # Robust answers everywhere outweigh the (theoretical) gain
        # near degeneracies.
        safety_extra_k = 0

        if can_use_parity:
            if (not verify_parity) and k == 0:
                predicted = _predict_ground_parity(s)
                Q_p = inv.parity_projector(predicted)
                v0_block = _project_to_block(
                    v0_full, Q_p, cutoff=projection_cutoff
                )
                energies_p, eigvecs_p = _solve_in_parity_block(
                    comps, Q_p, k=0, tol=tol, v0=v0_block,
                    safety_extra_k=safety_extra_k,
                )
                eigval = float(energies_p[0])
                amplitudes = eigvecs_p[:, 0].reshape(D_up, D_down)
                selected_parity = predicted
                eigvec_full = eigvecs_p[:, 0]
            else:
                Q_pos = inv.parity_projector(+1)
                Q_neg = inv.parity_projector(-1)
                v0_pos = _project_to_block(
                    v0_full, Q_pos, cutoff=projection_cutoff
                )
                v0_neg = _project_to_block(
                    v0_full, Q_neg, cutoff=projection_cutoff
                )
                energies_pos, eigvecs_pos = _solve_in_parity_block(
                    comps, Q_pos, k=k, tol=tol, v0=v0_pos,
                    safety_extra_k=safety_extra_k,
                )
                energies_neg, eigvecs_neg = _solve_in_parity_block(
                    comps, Q_neg, k=k, tol=tol, v0=v0_neg,
                    safety_extra_k=safety_extra_k,
                )
                all_energies = np.concatenate(
                    [energies_pos, energies_neg]
                )
                all_eigvecs = np.concatenate(
                    [eigvecs_pos, eigvecs_neg], axis=1
                )
                parities = np.concatenate(
                    [
                        np.full(len(energies_pos), +1),
                        np.full(len(energies_neg), -1),
                    ]
                )
                order = np.argsort(all_energies)
                eigval = float(all_energies[order[k]])
                amplitudes = all_eigvecs[:, order[k]].reshape(
                    D_up, D_down
                )
                selected_parity = int(parities[order[k]])
                eigvec_full = all_eigvecs[:, order[k]]

                if verify_parity and k == 0:
                    predicted = _predict_ground_parity(s)
                    assert selected_parity == predicted, (
                        f"ground-state parity mismatch: solver returned "
                        f"{selected_parity}, prediction "
                        f"(eps_Nup * eps_Ndown) is {predicted}"
                    )
        else:
            # Full-basis fallback for asymmetric systems or
            # use_parity=False.
            n_eig = k + 1 + safety_extra_k
            if n_eig >= D - 1:
                dense = np.empty((D, D), dtype=float)
                for col in range(D):
                    e = np.zeros(D)
                    e[col] = 1.0
                    dense[:, col] = comps.matvec(e)
                energies, eigvecs = np.linalg.eigh(dense)
                eigvecs = eigvecs[:, :n_eig]
                energies = energies[:n_eig]
            else:
                energies, eigvecs = _solve_with_preconditioner(
                    comps.op, k=n_eig, tol=tol, v0=v0_full,
                )
            eigval = float(energies[k])
            amplitudes = eigvecs[:, k].reshape(D_up, D_down)
            selected_parity = 0  # parity unknown / not applicable
            eigvec_full = eigvecs[:, k]

        # Persist the full-basis eigvec for the next solve. Copy so a
        # caller mutating the returned state's det_amplitudes can't
        # poison the cache.
        self._last_eigvec_full = np.array(eigvec_full, copy=True)

        state = iDEA.state.ManyBodyState(energy=eigval)
        state.det_amplitudes = amplitudes
        state.det_up_combs = comps.up_combs
        state.det_down_combs = comps.down_combs
        state.det_up_indicator = comps.up_indicator
        state.det_down_indicator = comps.down_indicator
        state.parity = selected_parity
        return state
