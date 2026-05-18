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

import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as spsla

import iDEA.methods.non_interacting
import iDEA.state
import iDEA.system


name = "interacting_det"


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
    )


def _build_solver_components(s: iDEA.system.System) -> _SolverComponents:
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
    diag = _build_diagonal(up_combs, down_combs, s.v_ext, s.v_int, h_diag)

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

    comps = _SolverComponents()
    comps.op = op
    comps.matvec = matvec
    comps.D = D
    comps.D_up = D_up
    comps.D_down = D_down
    comps.up_combs = up_combs
    comps.down_combs = down_combs
    comps.up_rank = up_rank
    comps.down_rank = down_rank
    comps.up_indicator = _build_indicator(up_combs, n_grid)
    comps.down_indicator = _build_indicator(down_combs, n_grid)
    comps.diag = diag
    comps.up_hops = up_hops
    comps.down_hops = down_hops
    return comps


def solve(
    s: iDEA.system.System,
    k: int = 0,
    tol: float = 0.0,
) -> iDEA.state.ManyBodyState:
    """Solve the interacting Schrödinger equation in determinant basis.

    | Args:
    |     s: iDEA.system.System, System object.
    |     k: int, Energy level to return (0 = ground state).
    |     tol: float, eigsh tolerance (0 = machine precision).

    | Returns:
    |     state: iDEA.state.ManyBodyState with .energy, .det_amplitudes,
    |     .det_up_combs, .det_down_combs, .det_up_indicator, .det_down_indicator.
    |     .full / .space / .spin are intentionally left as ArrayPlaceholder
    |     in Phase B; use iDEA.methods.interacting_det.density() for the
    |     density observable.
    """
    comps = _build_solver_components(s)
    D = comps.D
    D_up = comps.D_up
    D_down = comps.D_down

    n_eig = k + 1
    if n_eig >= D - 1:
        # eigsh requires k < D - 1; for tiny problems just densify.
        dense = np.empty((D, D), dtype=float)
        for col in range(D):
            e = np.zeros(D)
            e[col] = 1.0
            dense[:, col] = comps.matvec(e)
        energies, eigvecs = np.linalg.eigh(dense)
        eigvecs = eigvecs[:, : k + 1]
        energies = energies[: k + 1]
    else:
        energies, eigvecs = spsla.eigsh(comps.op, k=n_eig, which="SA", tol=tol)
        order = np.argsort(energies)
        energies = energies[order]
        eigvecs = eigvecs[:, order]

    eigval = float(energies[k])
    amplitudes = eigvecs[:, k].reshape(D_up, D_down)

    state = iDEA.state.ManyBodyState(energy=eigval)
    state.det_amplitudes = amplitudes
    state.det_up_combs = comps.up_combs
    state.det_down_combs = comps.down_combs
    state.det_up_indicator = comps.up_indicator
    state.det_down_indicator = comps.down_indicator
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
