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

try:
    import primme

    _HAS_PRIMME = True
except ImportError:
    _HAS_PRIMME = False


name = "interacting_det"


def _get_primme_min_dim() -> int:
    """Dimension below which we fall back to scipy regardless of PRIMME.

    PRIMME's Jacobi-Davidson defaults have higher per-iteration overhead
    than ARPACK Lanczos for tiny problems. Default 2000, override via
    environment variable ``IDEA_DET_PRIMME_MIN_DIM``.
    """
    return int(os.environ.get("IDEA_DET_PRIMME_MIN_DIM", "2000"))


def _resolve_backend(D: int, explicit: str = None) -> str:
    """Pick eigensolver backend. Precedence: explicit > env > auto."""
    if explicit is not None:
        return explicit
    env = os.environ.get("IDEA_DET_EIGSH_BACKEND")
    if env:
        return env
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
    |         remapped to ``1e-12`` for PRIMME (whose ``tol=0`` means
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
        # doesn't silently relax the convergence criterion. Some
        # upstream tests (test_manybody.py::TestShort) compare against
        # analytical wavefunctions at the 1e-7 to 1e-11 level; eps is
        # too tight for PRIMME's unpreconditioned Lanczos on larger
        # problems (fails to converge within maxiter on uudd_30).
        # 1e-13 is the empirically validated sweet spot.
        primme_tol = tol if tol > 0 else 1e-13
        v0_arg = v0.reshape(-1, 1) if v0 is not None else None
        # Phase E2 will construct a Jacobi preconditioner from
        # diag_for_prec here. Phase E1 leaves it unused.
        prec_op = None
        energies, eigvecs = primme.eigsh(
            op, k=k, which="SA", tol=primme_tol, v0=v0_arg, OPinv=prec_op,
            raise_for_unconverged=True,
        )
    else:
        energies, eigvecs = spsla.eigsh(op, k=k, which="SA", tol=tol)

    order = np.argsort(energies)
    return energies[order], eigvecs[:, order]


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


def _v_ext_is_parity_symmetric(s: iDEA.system.System, atol: float = 1e-10) -> bool:
    return bool(np.allclose(s.v_ext, s.v_ext[::-1], atol=atol))


def _v_int_is_parity_symmetric(s: iDEA.system.System, atol: float = 1e-10) -> bool:
    return bool(np.allclose(s.v_int, s.v_int[::-1, ::-1], atol=atol))


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


def _solve_in_parity_block(comps, Q_p, k=0, tol=0.0):
    """Run eigsh on Q_p^T H Q_p (the friend's 'implicit projector' form).

    Returns (energies[: k+1], eigvecs in full basis[:, : k+1]).
    """
    D_block = Q_p.shape[1]
    if D_block == 0:
        return np.array([], dtype=float), np.zeros((comps.D, 0))

    n_eig = k + 1

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
    energies, eigvecs = _solve_with_preconditioner(op_block, k=n_eig, tol=tol)
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

    if can_use_parity:
        n_grid = s.x.shape[0]
        n_up = s.up_count
        n_down = s.down_count
        s_ref, partner, self_mask, reps_pos, reps_neg = _build_parity_orbits(
            comps, n_grid, n_up, n_down
        )

        # Fast path: solve only the predicted-parity block for k=0 unless
        # the caller asks to verify. k>0 always falls through to both
        # blocks because we have no prediction for excited-state parities.
        if (not verify_parity) and k == 0:
            predicted = _predict_ground_parity(s)
            reps = reps_pos if predicted == +1 else reps_neg
            Q_p = _build_parity_projection(reps, partner, self_mask, s_ref, predicted, D)
            energies_p, eigvecs_p = _solve_in_parity_block(comps, Q_p, k=0, tol=tol)
            eigval = float(energies_p[0])
            amplitudes = eigvecs_p[:, 0].reshape(D_up, D_down)
            selected_parity = predicted
        else:
            Q_pos = _build_parity_projection(reps_pos, partner, self_mask, s_ref, +1, D)
            Q_neg = _build_parity_projection(reps_neg, partner, self_mask, s_ref, -1, D)
            energies_pos, eigvecs_pos = _solve_in_parity_block(comps, Q_pos, k=k, tol=tol)
            energies_neg, eigvecs_neg = _solve_in_parity_block(comps, Q_neg, k=k, tol=tol)
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
                comps.op, k=n_eig, tol=tol
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
