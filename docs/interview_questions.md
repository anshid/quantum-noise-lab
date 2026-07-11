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

### Q3. What exactly is the Born rule, and how does it relate to measure theory / POVMs?

**Answer.** For $|\psi\rangle=\sum_i c_i|i\rangle$ in an orthonormal basis, measuring in that basis
gives outcome $i$ with probability $P(i)=|c_i|^2=\langle\psi|\Pi_i|\psi\rangle$, where
$\Pi_i=|i\rangle\langle i|$ is the (idempotent, Hermitian) projector onto that basis vector. The
map $i\mapsto\langle\psi|\Pi_i|\psi\rangle$ is a genuine probability measure on the (finite)
outcome space: it's nonnegative, and $\sum_i P(i)=\langle\psi|\big(\sum_i\Pi_i\big)|\psi\rangle=
\langle\psi|\psi\rangle=1$ because $\{\Pi_i\}$ resolves the identity.

The projector family $\{\Pi_i\}$ is a special case — a projection-valued measure — of the more
general notion of a **POVM** (positive operator-valued measure): a set $\{E_i\}$ with each
$E_i\succeq 0$ (positive semidefinite, not necessarily a projector) and $\sum_i E_i = I$, giving
$P(i)=\langle\psi|E_i|\psi\rangle$. Projective measurements are the special case $E_i=\Pi_i$ with
$\Pi_i\Pi_j=\delta_{ij}\Pi_i$. POVMs are the right formalism once measurements are imperfect or
noisy (a realistic detector doesn't implement a clean orthogonal projection) — relevant once noise
modeling starts in Part 2.

**Finite-shot statistics:** a device only ever returns $N$ i.i.d. draws from the multinomial
distribution with probabilities $P(i)$. The empirical frequency $\hat P(i)=\text{count}_i/N$ has
standard error $\sqrt{P(i)(1-P(i))/N}=O(1/\sqrt N)$ — confirmed empirically in the notebook by
sampling a uniform 2-qubit superposition at $16,\dots,16384$ shots and watching the $L_1$ error
against the exact $0.25$-per-outcome distribution shrink at very close to the predicted
$1/\sqrt N$ rate. This is also exactly why the pytest suite checks Bell-state measurement
proportions with a loose $\pm10\%$ tolerance rather than exact equality.

*Relevant code:* `src/simulation.py::sample_counts`, `theoretical_probabilities`;
`notebooks/01_qiskit_fundamentals.ipynb` §2.

### Q4. What signature distinguishes an entangled state from a merely classically-correlated one, and why is the reduced density matrix the right tool to see it?

**Answer.** For a bipartite pure state, the **Schmidt decomposition**
$|\psi\rangle=\sum_k\lambda_k|a_k\rangle|b_k\rangle$ (orthonormal $\{|a_k\rangle\}$,
$\{|b_k\rangle\}$, $\lambda_k\ge0$, $\sum_k\lambda_k^2=1$) always exists, and the state is
separable iff its Schmidt rank (number of nonzero $\lambda_k$) is 1. For the Bell state
$|\Phi^+\rangle=(|00\rangle+|11\rangle)/\sqrt2$, $\lambda_1=\lambda_2=1/\sqrt2$ — Schmidt rank 2,
the maximum two qubits allow.

The operational fingerprint is the **reduced density matrix** $\rho_A=\mathrm{Tr}_B(|\psi\rangle
\langle\psi|)$. For $|\Phi^+\rangle$, computing this via `qiskit.quantum_info.partial_trace` gives
exactly $\rho_A=I/2$: the maximally mixed single-qubit state — confirmed numerically in the
notebook. This is the sharp classical/quantum contrast: classically, a joint distribution over
$(A,B)$ with zero entropy (a known, deterministic outcome) forces zero-entropy marginals too —
$H(A,B)=0\Rightarrow H(A)=H(B)=0$. Here the *global* state is pure ($S(\rho_{AB})=0$, since
$\rho_{AB}$ is a rank-1 projector) while each *local* reduced state has **maximal** entropy. A
globally pure state with maximally mixed marginals has no classical analogue — that gap is what
"entanglement" means operationally, not just "the qubits are correlated" (classical correlation
alone doesn't produce it; see Q6 for the sharpened version of this point with GHZ).

Also worth flagging: `partial_trace` is a completely-positive, trace-preserving (CPTP) map — the
same category of object that defines a noise channel. Tracing out a subsystem and suffering
physical decoherence are the same kind of mathematical operation; decoherence *is*, physically,
entanglement with an unmeasured/inaccessible environment followed by tracing it out. This is the
direct bridge into Part 2's Kraus-operator noise models.

*Relevant code:* `src/circuits.py::bell_state`; `notebooks/01_qiskit_fundamentals.ipynb` §3.

### Q5. Qiskit prints a measurement outcome as e.g. `"01"` — which qubit is which bit?

**Answer.** Qiskit orders qubits with **qubit 0 as the least-significant bit**. So a bitstring is
read right-to-left against qubit index: `"01"` means $q_1{=}0, q_0{=}1$, not "qubit 0 is 0, qubit
1 is 1" as a naive left-to-right reading would suggest. This is a frequent, easy-to-miss source of
off-by-reversal bugs, especially when cross-referencing hand-derived amplitude vectors (which are
usually written with qubit 0 first, left to right, in most textbooks) against Qiskit's
`Statevector.data` ordering (index $k$'s binary expansion has qubit 0 as the least-significant
bit of $k$) or against printed counts dictionaries. When indexing programmatically (e.g.
`partial_trace(sv, qubit_indices)`), always pass qubit *indices*, not string positions, to avoid
depending on this convention at all.

*Relevant code:* `notebooks/01_qiskit_fundamentals.ipynb` §3 (flagged inline after the Bell-state
reduced-density-matrix computation).

### Q6. Why is a GHZ state's entanglement considered more "fragile" than a Bell pair's?

**Answer.** Take $|\mathrm{GHZ}_3\rangle=(|000\rangle+|111\rangle)/\sqrt2$ on qubits $(0,1,2)$ and
compare two different partial traces (both verified numerically in the notebook):

- Trace out **two** qubits, keep one: $\rho_0=I/2$ — maximally mixed, same as the Bell-state case.
  A lone qubit can never exhibit "correlation" (that requires $\ge2$ parties), so this only shows
  local mixedness again, not anything GHZ-specific.
- Trace out **one** qubit, keep two: $\rho_{01}=\tfrac12(|00\rangle\langle00|+|11\rangle\langle11|)$
  — confirmed exactly by `partial_trace(ghz_sv, [2])` in the notebook, giving a diagonal matrix
  with $0.5$ at the `00` and `11` entries and zero elsewhere. This is a **classical mixture**: the
  off-diagonal coherence term $|00\rangle\langle11|$ vanishes because it would require
  $\langle0|1\rangle$ on the traced-out qubit, which is $0$. The two remaining qubits still always
  agree (perfect correlation survives) but that correlation is now entirely classical —
  indistinguishable from "flip a coin once, copy the result to two parties."

So losing access to **any single qubit** of a GHZ state instantly destroys all quantum coherence
among the rest, leaving only classical correlation behind. A Bell pair has no intermediate case to
compare against — there's nothing left to partially lose access to without destroying the pair
outright. This asymmetry in how different entangled states degrade under qubit loss is a direct
preview of why maintaining large entangled states on noisy hardware gets harder as they grow, and
it's why GHZ-state fidelity is a standard benchmark for how much noise a real device has (relevant
again once Part 2/3 quantify this under actual noise channels rather than an idealized "erase a
qubit" operation).

*Relevant code:* `src/circuits.py::ghz_state`; `notebooks/01_qiskit_fundamentals.ipynb` §4.
