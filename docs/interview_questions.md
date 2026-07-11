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

## Part 2 — Quantum Noise Models

### Q7. Why can't unitary evolution alone model physical noise, and where do Kraus operators actually come from?

**Answer.** A unitary is reversible ($U^\dagger U = I$) and preserves the purity of any state it
acts on — $\rho \to U\rho U^\dagger$ has the same eigenvalues as $\rho$. Real decoherence is
*not* reversible in this sense (you can't undo a spontaneous-emission event by re-applying a
gate), so it cannot be a unitary on the qubit's own Hilbert space alone.

The right model: let the qubit ($\rho$) start uncorrelated with an environment $E$ in a fixed
state $|0\rangle_E$, evolve the *joint* system unitarily (the whole universe is still closed), then
discard $E$ — exactly the `partial_trace` operation from Part 1's Q4/Q6, now tracing out an
environment instead of an entanglement partner:

$$\rho' = \mathrm{Tr}_E\big[U(\rho\otimes|0\rangle\langle0|_E)U^\dagger\big]
        = \sum_i K_i\rho K_i^\dagger, \qquad K_i := \langle e_i|_E\,U\,|0\rangle_E$$

for any orthonormal basis $\{|e_i\rangle\}$ of $E$. The $K_i$ act only on the qubit — the
environment index has been summed away — and this operator-sum form is exactly as general as
"unitary on system+environment, then trace out the environment": that equivalence is the
**Stinespring dilation theorem**. Trace preservation ($\mathrm{Tr}\rho'=1$ for all $\rho$) holds
iff $\sum_i K_i^\dagger K_i = I$ (the completeness relation); complete positivity is automatic from
the operator-sum form itself.

*Relevant code:* `src/noise_models.py` module docstring; `notebooks/02_noise_models.ipynb` §2.

### Q8. Kraus operators for a channel aren't unique — so how do you check two different-looking implementations of "the same" channel are actually equal?

**Answer.** If $\{K_i\}$ is a valid Kraus decomposition of a channel, so is $\{\sum_j U_{ij}K_j\}$
for *any* unitary matrix $U$ mixing the operators — this is an isometry freedom with no physical
content, not a different channel. So two Kraus lists that look completely different elementwise
can represent the identical physical map, and conversely comparing raw operators
(`np.allclose(K_list_1, K_list_2)`) is simply the wrong check.

The correct, basis-independent object is the channel's transfer-matrix representation — Qiskit's
`SuperOp` (equivalently `Choi`) — which is uniquely determined by the channel regardless of which
Kraus decomposition produced it. `tests/test_noise_models.py` verifies every hand-derived channel
against Qiskit's built-in `noise.errors` via `np.allclose(SuperOp(Kraus(ours)).data,
SuperOp(Kraus(theirs)).data)`, never by comparing the Kraus operators directly. The same
non-uniqueness is *why* phase damping and phase flip (Q10) can be "the same channel" despite
having visibly different-looking Kraus operators.

*Relevant code:* `tests/test_noise_models.py` (all four `test_*_matches_qiskit_builtin` tests).

### Q9. What's the actual difference between a "unital" and "non-unital" channel, and which of the five channels here is the odd one out?

**Answer.** A channel is **unital** if it maps the maximally mixed state to itself:
$\mathcal{E}(I/2) = I/2$. Bit flip, phase flip, depolarizing, and phase damping are all unital —
each is a linear map on the Bloch vector with no additive shift ($r \to M r$ for some matrix $M$),
so $r=0$ (the center of the sphere, i.e. $I/2$) is always a fixed point.

**Amplitude damping is not**: its Bloch-vector map is $r \to Mr + c$ with $c=(0,0,\gamma)\neq 0$
(derived and confirmed numerically in the notebook), so $r=0 \to (0,0,\gamma) \ne 0$. Physically
this makes sense — amplitude damping models spontaneous relaxation *toward the ground state*
$|0\rangle$, a directional physical process with a preferred destination, not merely "randomize and
lose information." As $\gamma\to1$, *every* input state (regardless of its own Bloch vector) is
pulled all the way to $r=(0,0,1)$, i.e. $|0\rangle$ exactly — a **pure** state, so purity actually
climbs back to 1 at $\gamma=1$, the only channel among the five whose purity-vs-parameter curve is
non-monotonic (confirmed in the comparison plot: amplitude damping's curve dips and then returns to
1).

*Relevant code:* `src/noise_models.py::amplitude_damping_kraus`;
`notebooks/02_noise_models.ipynb` §6, §8 (comparison plot).

### Q10. Isn't phase damping just phase flip with a different name?

**Answer.** More than "similar" — they're literally the same channel, under a specific
reparametrization, which is confirmed by direct `SuperOp` equality in the notebook: phase
damping with parameter $\lambda$ equals phase flip with $p = \tfrac{1-\sqrt{1-\lambda}}{2}$, for
every $\lambda$ tested. This can look surprising because their conventional Kraus operators look
different (phase flip: $\{\sqrt{1-p}\,I,\ \sqrt p\,Z\}$; phase damping:
$\{\mathrm{diag}(1,\sqrt{1-\lambda}),\ \mathrm{diag}(0,\sqrt\lambda)\}$) — but per Q8, different
Kraus operators don't imply different channels.

What actually distinguishes them in practice is not the math but the **physical story attached to
the parameter**: phase flip's $p$ is the natural parameter for a discrete, one-shot Pauli-error
event (e.g. one noisy gate application); phase damping's $\lambda = 1-e^{-t/T_2}$ is the natural
parameter for a *continuously acting* physical dephasing process during idle time $t$ next to a
$T_2$ bath. Same mathematical object, two different physical contexts that motivate naming and
parametrizing it differently — a good illustration of why "what channel is this" and "why do we
call it that" are separate questions.

*Relevant code:* `notebooks/02_noise_models.ipynb` §7 (direct `SuperOp` equality check with
`p_equiv = (1 - np.sqrt(1 - lam)) / 2`).

### Q11. Why does Qiskit's depolarizing channel include an identity term, and does that change what the parameter "$p$" means compared to bit/phase flip?

**Answer.** Yes — it's a real convention difference, not just notation. Qiskit's single-qubit
depolarizing channel is $\mathcal E(\rho)=(1-\tfrac{3p}{4})\rho+\tfrac p4(X\rho X+Y\rho Y+Z\rho Z)$,
a uniform mixture over *all four* single-qubit Paulis including identity. Using the identity
$X\rho X+Y\rho Y+Z\rho Z=2I-\rho$ (valid for normalized $\rho$), this simplifies exactly to

$$\mathcal E(\rho) = (1-p)\,\rho + p\,\frac{I}{2}$$

— "with probability $p$, replace the state by the maximally mixed state; otherwise leave it
alone." That's a clean, physically transparent form, and it's why depolarizing's Bloch-vector map
is *isotropic*: $r\to(1-p)r$, shrinking all three components by the same factor (confirmed in the
notebook — the one channel whose three Bloch-decay curves sit exactly on top of each other).

But it means depolarizing's $p$ is **not** directly comparable to bit-flip's or phase-flip's $p$
at face value: bit-flip's $p$ is "the probability of specifically an $X$ error," while
depolarizing's $p$ already has the "do nothing" identity outcome folded into its own
normalization. Reporting "10% depolarizing noise" and "10% bit-flip noise" are not the same
*amount* of damage to the state — comparing error rates across channel *types* requires picking a
common summary statistic (e.g. purity or fidelity at matched parameter, as in the notebook's
comparison plot) rather than assuming the raw parameters are on the same scale.

*Relevant code:* `src/noise_models.py::depolarizing_kraus`; `notebooks/02_noise_models.ipynb` §5.

## Part 3 — Noise Comparison Experiments

### Q12. If you apply the same noise channel independently, `k` times in a row, is that the same channel with a bigger parameter, or something qualitatively different?

**Answer.** For three of the five channels in this project, it's provably *the same channel*
with a specific, derivable composed parameter — not a coincidence, and verified via `SuperOp`
equality (per Q8, never by comparing Kraus operators) in `notebooks/03_noise_experiments.ipynb`:

- **Bit flip**, composed $k$ times: since $X^2=I$, the net effect is an $X$ error iff an *odd*
  number of the $k$ independent flips occurred. For i.i.d. Bernoulli($p$) trials, the standard
  "random walk on $\mathbb Z_2$" identity gives $p_k = \tfrac12-\tfrac12(1-2p)^k$.
- **Depolarizing**, composed $k$ times: $\mathcal E(\rho)=(1-p)\rho+p\,I/2$ is an affine
  contraction toward $I/2$ by factor $(1-p)$ each application, so composing $k$ times contracts by
  $(1-p)^k$, giving $p_k=1-(1-p)^k$.
- **Amplitude damping**, composed $k$ times: $\gamma_k = 1-(1-\gamma)^k$ — and this is exactly the
  *discrete* form of continuous exponential $T_1$ decay. If each layer models idle time $\delta t$
  with $\gamma=1-e^{-\delta t/T_1}$, then $\gamma_k=1-e^{-k\delta t/T_1}$ — the textbook $T_1$ decay
  law falls straight out of composing independent layers, rather than being a separate assumption.

Phase flip and phase damping don't get a separate treatment here only because (Q10) they're
already the same channel family as each other — the bit-flip-style composition law applies to
phase flip identically by the $X\leftrightarrow Z$ symmetry.

This matters practically: it means "depth" in a noisy-circuit sweep isn't merely "run the
simulation longer and see what happens" — for these channels, the effect of depth is *exactly*
predictable in closed form, and the sweep script's `depth` axis is a legitimate stand-in for
accumulated idle time / gate count, not an ad hoc knob.

*Relevant code:* `notebooks/03_noise_experiments.ipynb` §2 (`compose_k_times` verification cell).

### Q13. Fidelity and error rate are both "how much did noise hurt," so why report both instead of picking one?

**Answer.** They measure different things and can disagree sharply. `error_rate` (and its
complement, `success_probability`) only asks "did the measured computational-basis bitstring fall
inside the ideal support?" — a coarse, single-basis yes/no. `fidelity` is the full quantum-state
overlap $F(\rho,\sigma)$, sensitive to *any* deviation, not just ones visible in one fixed
measurement basis.

The disagreement isn't hypothetical — it showed up immediately in the committed sweep data: for a
Bell state at depth 4, phase flip and phase damping both show `error_rate = 0.0` (measuring
`{"00","11"}` with certainty, exactly as ideal) while their `fidelity` had already dropped to
$\approx 0.61$. This is exactly Part 2's Q10 lesson resurfacing with consequences: phase flip and
phase damping only ever damage *coherence* (off-diagonal density-matrix terms), never
computational-basis *populations* — so a Z-basis measurement is structurally blind to the damage
they do, even though the state has genuinely become a different, far-less-useful quantum state
(useless for anything requiring the destroyed coherence, e.g. a subsequent Hadamard-basis
operation). Reporting only `error_rate` here would be actively misleading; reporting only
`fidelity` loses the operationally-relevant "does my specific measurement protocol still work"
answer. Both, together, are the honest summary.

*Relevant code:* `notebooks/03_noise_experiments.ipynb` §5 (fidelity-vs-error-rate comparison
cell); `src/experiment_utils.py::success_probability`, `error_rate`.

### Q14. Why is the qubit-count axis capped at 6 in this experiment, when the shot-based sampling backend clearly handles more qubits fine in Parts 1–2?

**Answer.** Two different computations are running side by side in this sweep, with very
different scaling. The **exact fidelity** metric requires the full density matrix, a
$2^n\times2^n$ object, and applying a Kraus channel to it costs $O(4^n)$ — this is the actual
bottleneck, and it's what the qubit-count cap is protecting against, not the sampling side.
**Shot-based sampling** (`sample_counts` via `AerSimulator`/`SamplerV2`) doesn't need to build or
store a dense $2^n\times2^n$ matrix at all — Qiskit's simulator backend can simulate many more
qubits before becoming impractical. So the constraint here is specifically "the experiment wants
*exact* fidelity, not just empirical counts," and exact fidelity is the expensive thing — a
reminder that `fidelity` and `success_probability`/`error_rate` aren't just two flavors of the same
answer (Q13) — they also don't cost the same to compute, which is itself a real practical tradeoff
in noise characterization work (fast approximate estimates vs. slow exact ones).

*Relevant code:* `src/experiment_utils.py::apply_noise_layers`; `experiments/configs/noise_sweep.toml`
(`qubit_counts = [2, 3, 4, 5, 6]`); `notebooks/03_noise_experiments.ipynb` "Limitations" section.

### Q15. Part 1 said GHZ entanglement is "fragile" because losing one qubit destroys all coherence among the rest. What does that look like quantitatively under an actual noise channel, instead of an idealized "erase a qubit" operation?

**Answer.** Under one layer of independent per-qubit noise (any of the five channels), GHZ-state
fidelity falls off roughly **geometrically**, not linearly, in qubit count: the committed sweep
(depolarizing, per-qubit parameter 0.1) shows fidelity ratios between consecutive qubit counts
sitting at $\approx0.927$ across $n=2\to6$ — essentially constant, the signature of
$F(n)\approx c\cdot r^n$ rather than a linear decline. The mechanism is direct: each additional
qubit is one more *independent* opportunity for the channel to damage the state, so the *surviving
fraction* compounds multiplicatively with $n$, exactly like a chain of independent Bernoulli
survivals. This is the quantitative version of Part 1's qualitative claim, and it directly
motivates the working Part 5 research question: which channel's fidelity decays fastest, in this
geometric sense, as qubit count grows.

*Relevant code:* `notebooks/03_noise_experiments.ipynb` §4 (GHZ fidelity vs. qubit count, single
channel) and §5 (all five channels, from the committed sweep).

## Part 4 — Simple Error Mitigation

### Q16. Why is the majority-vote logical error rate exactly $3p^2-2p^3$, and why does it cross the "no mitigation" line at exactly $p=0.5$?

**Answer.** The 3-qubit repetition code encodes a logical bit by copying it onto 3 physical
qubits, each of which independently suffers a bit flip with probability $p$. Majority vote decodes
*incorrectly* exactly when 2 or 3 of the 3 physical qubits flip — a direct binomial tail:

$$
p_{\text{logical}} = P[\text{Binomial}(3,p)\ge 2] = \binom{3}{2}p^2(1-p) + \binom{3}{3}p^3 = 3p^2-2p^3
$$

The crossover at $p=0.5$ isn't a coincidence needing a separate proof — it falls straight out of
the formula, $3(0.25)-2(0.125)=0.5$, and it has a clean interpretation: at $p=0.5$ each physical
qubit is a fair coin, and majority vote over 3 independent fair coins is *also* a fair coin — no
information is gained by taking a vote among symmetric noise. Below $p=0.5$, redundancy helps
(needing 2 independent failures is harder than 1); above $p=0.5$, it actively hurts, since 3
independent votes for the *already more likely* wrong answer become more likely than one vote for
it. This is a direct application of order statistics / majority-of-$n$ reasoning from classical
probability, applied to a quantum decoding step.

*Relevant code:* `src/error_mitigation.py::logical_error_rate_theoretical`;
`tests/test_error_mitigation.py::test_logical_error_rate_theoretical_matches_binomial_tail`;
`notebooks/04_error_mitigation.ipynb` §2, §4.

### Q17. Why isn't the repetition code + majority vote considered full quantum error correction (QEC)?

**Answer.** Two independent reasons, often conflated but worth keeping separate:

1. **Majority-vote decoding destroys superpositions.** Decoding requires a direct
   computational-basis measurement of all 3 physical qubits. That measurement collapses any
   encoded superposition outright — this scheme only ever recovers a *classical* bit value, never
   a qubit that can continue being computed on. Real QEC instead measures **stabilizers** — parity
   checks like $Z_1Z_2$ and $Z_2Z_3$, via ancilla qubits — which reveal *where* an error occurred
   (the syndrome) without revealing (and therefore without collapsing) the encoded logical state,
   followed by a corrective gate conditioned on the syndrome.
2. **It only protects one error type** (see Q18) — a genuinely general-purpose code needs to
   correct both bit-flip and phase-flip errors, since an arbitrary single-qubit error is a linear
   combination of $I, X, Y, Z$.

Both gaps point at the same stretch-goal direction: stabilizer formalism, syndrome extraction, and
(eventually) surface codes.

*Relevant code:* `notebooks/04_error_mitigation.ipynb` §6.

### Q18. Why does the repetition code give exactly zero protection against phase-flip noise, and what does that imply?

**Answer.** This code encodes and decodes entirely in the $Z$ basis (CNOT-copy the input, measure
computationally, vote). For a **computational-basis input**, $Z|0\rangle=|0\rangle$ and
$Z|1\rangle=-|1\rangle$ (up to an unobservable global phase) — a phase flip literally does nothing
observable to a $Z$-eigenstate, at any $p$. So majority vote reports **zero logical error for every
value of $p$**, which looks like perfect protection but is a measurement blind spot, not a
correction: a $Z$-basis measurement of a $Z$-eigenstate can never see a $Z$ error.

The real degradation only becomes visible with a **superposition input**: encoding $|+\rangle$
produces $(|000\rangle+|111\rangle)/\sqrt2$ (structurally a GHZ state), and checking the true state
fidelity after independent phase-flip noise on each physical qubit shows real, monotonic loss
(dropping to fidelity exactly $0$ at $p=1$, since flipping all three qubits' phases sends
$|000\rangle+|111\rangle \to |000\rangle-|111\rangle$, an orthogonal state) — invisible to the
majority-vote metric entirely. This is the same "fidelity vs. error rate can disagree" lesson from
Part 3 (Q13), taken to its logical extreme: here the disagreement isn't partial, it's total. It
also motivates why a genuinely robust code (Shor's 9-qubit code, and stabilizer/CSS codes more
generally) nests this bit-flip code inside its Hadamard-conjugate phase-flip code, protecting both
bases at once.

*Relevant code:* `src/error_mitigation.py::repetition_encode`;
`tests/test_error_mitigation.py::test_phase_flip_noise_is_invisible_to_majority_vote_on_computational_input`
and `::test_phase_flip_noise_still_degrades_true_state_fidelity_of_a_superposition_input`;
`notebooks/04_error_mitigation.ipynb` §5.
