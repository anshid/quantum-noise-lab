# Interview Questions & Answers

Q&A accumulated while building `quantum-noise-lab`, organized by project part. Each entry
records a real question that came up during development (often from a design decision or a
failing test) plus the full answer, so it can be reviewed and re-derived from memory before an
interview rather than just re-read.

## Part 1 — Qiskit Fundamentals

### Q1. A test asserted exact statevector amplitudes for the `psi_minus` Bell state and failed — the circuit produced `(0, -1, 1, 0)/√2` instead of the expected `(0, 1, -1, 0)/√2`. Is that a bug in the circuit or a bug in the test?

**Answer.** Neither is "wrong" on its own — the test was asserting something that isn't physically
meaningful: exact equality of a specific vector *representative*, rather than equality of the
physical *state*.

A quantum state is not really a vector $|\psi\rangle \in \mathcal{H}$; it is an equivalence class
of vectors under $|\psi\rangle \sim e^{i\theta}|\psi\rangle$ — a ray in Hilbert space, i.e. an
element of the projective Hilbert space $\mathbb{P}(\mathcal{H})$. This is forced, not a
convention: every measurable quantity is either a Born-rule probability
$|\langle\phi|\psi\rangle|^2$ or an expectation value $\langle\psi|A|\psi\rangle$ for Hermitian
$A$, and substituting $|\psi'\rangle = e^{i\theta}|\psi\rangle$ into either leaves it unchanged —
the phase cancels exactly. Equivalently, in density-matrix form:
$\rho = |\psi'\rangle\langle\psi'| = e^{i\theta}|\psi\rangle\langle\psi|e^{-i\theta} =
|\psi\rangle\langle\psi| = \rho$. The density matrix — the physically meaningful object — never
sees a global phase at all.

Tracing the `psi_minus` circuit (`H(0)`, `Z(0)`, `CX(0,1)`, `X(1)`) through Qiskit's own indexing
convention (qubit 0 = least-significant bit) step by step:

- `H(0)` on $|00\rangle$: $(|00\rangle + |01\rangle)/\sqrt2 = (1,1,0,0)/\sqrt2$
- `Z(0)`: flips sign where $q_0{=}1$ → $(1,-1,0,0)/\sqrt2$
- `CX(0,1)`: flips $q_1$ where $q_0{=}1$, moving the $-1$ amplitude from index 1 ("01") to index 3
  ("11") → $(1,0,0,-1)/\sqrt2$ (this is $\Phi^-$, and matches the `phi_minus` test that already
  passed)
- `X(1)`: flips $q_1$ everywhere, index 0 → index 2, index 3 → index 1 →
  $(0,-1,1,0)/\sqrt2$

The intended textbook target was $(0,1,-1,0)/\sqrt2$ (i.e. $\Psi^- = (|01\rangle-|10\rangle)/\sqrt2$).
The actual output is exactly $-1$ times that — a global phase of $e^{i\pi}$, and nothing more.
Same ray, same $\rho$, same entanglement, same measurement statistics.

**Fix:** compare via `Statevector.equiv()` instead of `np.allclose` on raw amplitudes.
`.equiv()` computes the overlap $|\langle\phi|\psi\rangle|$ and checks it's $\approx 1$ — invariant
to exactly the phase freedom that isn't physical — rather than asserting equality of one arbitrary
phase representative.

**Contrast — why this doesn't mean "phase never matters":** the phase *between terms* inside a
single superposition (a *relative* phase) is fully observable. $|0\rangle+|1\rangle$ and
$|0\rangle-|1\rangle$ give identical Z-basis measurement statistics (50/50) but are distinguished
perfectly by applying `H` first: $H(|0\rangle+|1\rangle)/\sqrt2 = |0\rangle$ while
$H(|0\rangle-|1\rangle)/\sqrt2 = |1\rangle$. Global phase multiplies the *whole* vector by one
constant; relative phase changes the interference pattern *between* components of the same
vector, and interference is exactly what a change of measurement basis exposes.

*Relevant code:* `src/circuits.py::bell_state`, `tests/test_circuits.py::test_bell_state_amplitudes`.

### Q2. What's the precise mathematical reason relative phase is observable while global phase is not?

**Answer.** A global phase is the transformation $|\psi\rangle \mapsto e^{i\theta}|\psi\rangle$
for a single scalar $\theta$ applied uniformly — i.e. it's the operator $e^{i\theta}I$, a scalar
multiple of the identity. Because it's proportional to the identity, it commutes with *every*
operator: $[e^{i\theta}I, A] = 0$ for all $A$, and it acts identically regardless of which basis
you choose to measure in. It can never survive into any expression of the form
$\langle\phi|A|\psi\rangle$ because the phase factors out on the bra side and the ket side and
cancels (or, for a bare inner product, comes out as an overall unobservable phase on a complex
number whose *magnitude* is all that ever enters a probability).

A relative-phase gate, by contrast, is a *non-scalar* diagonal unitary in some fixed basis — e.g.
$\mathrm{diag}(1, e^{i\varphi})$ in the computational basis, sending
$\alpha|0\rangle+\beta|1\rangle \mapsto \alpha|0\rangle + \beta e^{i\varphi}|1\rangle$. Because it
is *not* a multiple of the identity, it does not commute with every operator — in particular it
doesn't commute with $H$ or $X$ — so there exist observables $A$ for which
$\langle\psi|U^\dagger A U|\psi\rangle \ne \langle\psi|A|\psi\rangle$. Concretely: rotate into the
Hadamard basis and the relative phase becomes a difference between orthogonal, perfectly
distinguishable outcomes (as in Q1's example, $|0\rangle\pm|1\rangle \to |0\rangle$ or $|1\rangle$
under $H$).

This generalizes via Schur's lemma: on an irreducible representation, the *only* operators that
commute with every unitary (equivalently, with every observable) are scalar multiples of the
identity — i.e. exactly the global-phase transformations. Any unitary that isn't a global scalar
is therefore guaranteed to fail to commute with *some* observable, which is precisely what makes
it detectable by choosing the right measurement basis. So "global phase is invisible" isn't a
special quirk of quantum mechanics tacked on by convention — it's the maximal invisible symmetry
allowed by the structure of $\langle\psi|A|\psi\rangle$, and everything outside that one-parameter
center of the unitary group is generically observable.
