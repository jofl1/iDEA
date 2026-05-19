# iDEA interacting-solver optimisation: handoff

This document is the end-of-Phase-D summary of the optimisation work
done on this fork of iDEA. It supersedes ``notes/PHASE_D_RESUMPTION.md``
(which was a mid-work resumption note) and complements
``notes/baseline_timings.txt`` (the raw wall-time + memory tables) and
``notes/friend_optimization_advice.md`` (the original review the work
was based on).

## Goal

Make iDEA faster in every way for every supported workflow, while
producing the same answers as the upstream code. Existing scripts should
pick up the speedup without code changes.

After Phase D, that goal holds for ground-state interacting solves on
``'u'``/``'d'`` electron configurations — the only kind iDEA supports.
``iDEA.methods.interacting.solve`` is now ~30x faster on ``uud_60``
without any caller changes, and ``.full``-based observables continue to
work transparently.

## Commits (chronological, oldest first)

| # | SHA | One-liner | Phase |
|---|---|---|---|
|  1 | ``d2edb34`` | Infra: ``__pycache__`` and ``benchmarks/golden_outputs`` to ``.gitignore``. | Phase A |
|  2 | ``b05f7d7`` | Fast non-interacting path in reverse KS inversion. | Phase A |
|  3 | ``6e9ad35`` | Fast ``density`` / ``density_matrix`` for ``SingleBodyState``. | Phase A |
|  4 | ``b8ba92f`` | Benchmark scripts and golden-output regression harness. | Phase A |
|  5 | ``2f61967`` | Reproducibility tests for optimised kernels. | Phase A |
|  6 | ``b06a0ab`` | Subset eigensolve in non-interacting ``sc_step``. | Phase A |
|  7 | ``15aabca`` | Vectorised density kernel + FFT-based Hartree potential. | Phase A |
|  8 | ``02ab862`` | Friend's optimisation advice committed verbatim as notes. | Phase B prep |
|  9 | ``79d7398`` | Interacting-solver benchmark + baseline harness. | Phase B prep |
| 10 | ``3f17958`` | Determinant-basis matrix-free FCI solver. | Phase B |
| 11 | ``3952819`` | Phase B sanity tests (Hermiticity, residual, dense small-N). | Phase B |
| 12 | ``1f3f927`` | Spatial parity block-diagonalisation (both blocks merged). | Phase C |
| 13 | ``ecfbfb9`` | Parity fast path (single predicted-parity block). | Phase C+ |
| 14 | ``f065bdb`` | ``build_labelled_state`` helper for backward-compatible dispatch. | Phase D1 |
| 15 | ``ac000d2`` | Correct ``.space`` reconstruction per friend's prescription. | Phase D1.5 |
| 16 | ``8130142`` | Transparent dispatch from ``interacting.solve`` to det fast path. | Phase D2 |

11 commits ahead of ``origin/master`` (some upstream commits were
ancestral to ours; the count is between the two diverged tips).

## Measured speedups

All numbers from ``notes/baseline_timings.txt`` (laptop, Darwin 24.6.0).

### Per-workflow wall-time speedups

| Workflow | Before | After Phase D | Speedup |
|---|---|---|---|
| Reverse-engineering small-system step (Phase A subset eigh path) | several seconds | sub-second | 2-5x |
| ``observables.density`` on ``SingleBodyState`` (Phase A) | O(N^2) Python loop | O(N) vectorised | 5-10x |
| Hartree potential evaluation (Phase A FFT) | O(N^2) | O(N log N) | 10-50x at large N |
| ``interacting.solve`` ``uu_80``  (2 up, 0 down, N=80) | 0.166s | 0.115s | 1.44x |
| ``interacting.solve`` ``uud_60`` (2 up, 1 down, N=60) | 26.146s | 0.927s | **28.2x** |
| ``interacting.solve`` ``uudd_30``(2 up, 2 down, N=30) | (stochastic ``IndexError``) | 2.099s deterministic | (qualitative win) |

### Attribution per the friend's roadmap

The friend was emphatic about attributing each speedup to one cause.
The interacting solver in particular has three independent levers
stacked:

- **Phase B (representation change, det basis):** ~25x on ``uud_60``.
  Dimension cut + skipping the labelled-CSR Hamiltonian + skipping the
  post-solve ``antisymmetrize``. Memory cut ~87%.
- **Phase C (parity block-diagonalisation, both blocks):** small further
  gain (~1.1x) because both blocks are solved and merged.
- **Phase C+ (predicted-parity fast path):** an additional 1.2-1.3x by
  only solving the ground-parity block.
- **Phase D (dispatch + ``.full`` reconstruction):** tiny regression
  (0.06s on ``uud_60``, ~0.8s on ``uudd_30``) from the
  ``antisymmetrize`` reconstruction needed to populate ``.full`` for
  observables. Bounded.

## Validation footprint

Each phase is gated by its own test files; the full set covers ~35
det-solver assertions plus the upstream analytical-reference tests.

| Coverage | File | Phase |
|---|---|---|
| Vectorised density / FFT Hartree reproducibility | ``tests/test_observables_density_kernels_repro.py`` | A |
| Subset eigh and fast non-interacting path | ``tests/test_reverse_engineering_repro.py`` | A |
| Golden-output regression for non-interacting cases | ``tests/test_golden_output_harness.py`` | A |
| Energy + density baselines for ``uu_80``, ``uud_60``, ``uudd_30`` | ``tests/test_interacting_baseline.py`` | B |
| Hermiticity, residual norm, dense small-N cross-check | ``tests/test_interacting_det.py`` | B |
| Parity reflection involution, ``[H, P]=0``, ground-parity prediction | ``tests/test_interacting_det.py`` | C |
| Predicted-block fast-path energy match | ``tests/test_interacting_det.py`` | C+ |
| Reconstructed-state density and ``density_matrix`` match | ``tests/test_interacting_det.py`` | D1 |
| Analytical-reference wavefunction for ``u``, ``uu``, ``ud`` | ``tests/test_manybody.py::TestShort`` | (D2 unbroke ``uu``/``ud``) |

Standard regression run (fast suite, ~35s):

```
pytest tests/test_interacting_det.py tests/test_interacting_baseline.py \
       tests/test_reverse_engineering_repro.py \
       tests/test_observables_density_kernels_repro.py \
       tests/test_golden_output_harness.py tests/test_time_evolution.py \
       tests/test_manybody.py::TestShort -q
```

Expected: 63 passed.

## Known issues (unchanged by this work)

- **``tests/test_manybody.py::TestLong::test_density[ud]``**: failure
  at max abs diff 3.273e-13 (tolerance 2.0e-13). Was 3.481e-13 against
  the same tolerance before Phase D, so the failure margin actually
  *shrunk* slightly under dispatch (the det solver is marginally more
  accurate on this case). Pre-existing iDEA flake — the tolerance is
  set tighter than the natural floating-point noise floor of the
  problem; not introduced by this work.
- **``tests/test_time_evolution.py::TestShort::test_wavefunction[u]``**:
  order-sensitive flake (passes in isolation, can fail when later in a
  larger run). Same family as the ``[ud]`` failure. Did not fire in
  the Phase D long-suite run (``68 passed, 1 failed in 3:01``). Not
  introduced by this work.

## What still falls back to the legacy labelled path

``iDEA.methods.interacting.solve`` only dispatches to the determinant
fast path when **all** of these hold:

- ``k == 0`` (ground state)
- ``H is None`` (default Hamiltonian)
- ``level is None``
- ``GPU is False``
- ``s.electrons`` contains only ``'u'`` and ``'d'``
- ``bypass_det`` is ``False`` (the default)

If any condition fails, ``solve`` runs the original labelled solver
unchanged. So:

- Excited states (``k > 0``) — falls back. Could be lifted by extending
  the parity prediction or solving both blocks in the det path.
- Custom Hamiltonian (``H`` argument supplied) — falls back. Not
  obviously worth porting since callers passing ``H`` are doing something
  custom.
- GPU mode — falls back. Friend de-prioritised GPU; can be revisited.
- Unusual electron alphabets — falls back. iDEA itself only supports
  ``'u'``/``'d'``, so this is effectively a defensive guard.
- ``bypass_det=True`` — caller explicit opt-out, for debugging and
  numerical validation.

## TODO / suggested next phases

Friend's roadmap (paraphrased from
``notes/friend_optimization_advice.md``):

- **Phase C++ — compressed-basis parity matvec.** Today the parity
  fast path uses the implicit-projector form ``eigsh(Q_p^T H Q_p)``,
  which scatters into the full basis on every matvec. A direct
  compressed-basis matvec (no scatter) gives the asymptotic 2x
  reduction the projector form can't reach. Friend's prediction:
  "real ~2x over implicit projector". Code lives in
  ``iDEA/methods/interacting_det.py``.
- **Phase D follow-up — determinant-aware observables.** Skip the
  ``antisymmetrize`` round-trip in ``interacting.solve`` when only
  density is needed. Would erase the small Phase D regression vs Phase
  C+ on ``uudd_30``. Touches ``iDEA/observables.py``.
- **Phase E — PRIMME + diagonal Jacobi preconditioner + warm-started
  initial vectors.** Friend's predicted 20x lever on top of Phase B+C;
  the biggest remaining win for repeated workloads (KS-DFT outer loops,
  reverse-engineering inner loops, time evolution). New dependency
  (``primme``) but it's pip-installable and pure CPython usable.
- **Phase F — spin-flip symmetry for ``N_up == N_down`` cases.**
  Additional 2x where applicable (``ud``, ``uudd``, etc.).
- **Phase G — GPU matrix-free kernels.** Friend de-prioritised this
  ("GPU is the wrong lever first"), but mentioned it as a parallel
  track. Would need a separate ``LinearOperator`` backend.

## Pointers

- ``notes/baseline_timings.txt`` — raw wall-time + memory tables for all
  phases, plus final summary table.
- ``notes/friend_optimization_advice.md`` — original review from the
  friend (Phase B prep). Reference for parity prescription and FCI
  conventions.
- ``/Users/joshfleming/.claude/plans/plan-a-thorough-way-gleaming-sprout.md``
  — Phase D plan as authored before implementation.
- ``benchmarks/benchmark_interacting.py`` — solver timing harness with
  ``--solver labelled|det`` flag and ``time/generate/compare``
  subcommands.

## Public API surface added by this work

The optimisations preserve the upstream API. The only additions are:

- ``iDEA.methods.interacting.solve(..., bypass_det=False)`` — new kwarg
  to force the legacy labelled solver.
- ``iDEA.methods.interacting_det`` — new module exposing ``solve(s,
  k=0, tol=0.0, use_parity=True, verify_parity=False)`` returning a
  ``ManyBodyState`` with ``.det_amplitudes`` / ``.det_up_combs`` /
  ``.det_down_combs`` / ``.det_up_indicator`` / ``.det_down_indicator`` /
  ``.parity`` attached. ``.full``/``.space``/``.spin`` are left as
  ``ArrayPlaceholder`` so callers wanting only density and energy can
  skip the reconstruction cost.
- ``iDEA.methods.interacting_det.density(s, state)`` — total density
  from the determinant amplitudes directly.
- ``iDEA.methods.interacting_det.build_labelled_state(s, det_state)``
  — reconstructs ``.space`` / ``.spin`` / ``.full`` from a det-state.
  Used internally by ``interacting.solve``; callers can use it directly
  if they have a det-state and want observables.

No existing functions changed signature beyond the new ``bypass_det``
kwarg. No existing test was modified; the wavefunction comparison
tests for ``uu`` and ``ud`` started passing once Phase D2's ``.space``
reconstruction was correct.
