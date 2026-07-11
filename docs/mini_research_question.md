# Mini Research Question: Which Noise Channel Degrades GHZ Fidelity Fastest?

## Hypothesis

Among the five noise channels studied in Parts 2–3 (bit-flip, phase-flip, depolarizing, amplitude
damping, phase damping), **bit-flip noise degrades an $n$-qubit GHZ state's fidelity fastest as
qubit count grows**, and this is not merely a quantitative difference in decay speed — it reflects
a qualitative mechanism: channels whose Kraus operators are diagonal in the computational basis
(phase-flip, phase damping) can never reduce GHZ fidelity below $1/2$, no matter how large $n$
grows, while channels that are not diagonal (bit-flip, depolarizing, amplitude damping) genuinely
drive fidelity to $0$, with bit-flip doing so fastest.

## Methodology

1. **Reused Part 3's committed sweep** (`experiments/results/noise_sweep.csv`, GHZ rows, depth 1,
   param 0.1, qubit count 2–6) rather than re-running it — the qubit-count-vs-fidelity data needed
   for a ranking was already available.
2. **Derived closed-form fidelity expressions** for the two diagonal channels (phase-flip, phase
   damping) from first principles, by decomposing the GHZ density matrix into population and
   coherence terms and tracking how each is affected by a diagonal Kraus operator.
3. **Fit an empirical geometric decay rate** $r$ per channel (fidelity $\sim A\,r^n$) via a
   log-linear least-squares fit (`src/research_analysis.py::fit_geometric_decay_rate`), turning
   the qualitative "channel X decays faster" observation into a single comparable number.
4. **Verified the closed forms numerically** against direct density-matrix simulation (exact
   agreement to floating-point precision at every qubit count checked).
5. **Ran a new, small, targeted extension sweep** (`experiments/configs/ghz_decay_extension.toml`,
   `experiments/run_ghz_decay_extension.py`) extending qubit count to 7–8 for just the two extreme
   channels (bit-flip, phase damping), to confirm the fitted trend from $n=2$–$6$ actually
   extrapolates past Part 3's cost-driven cap, rather than being an artifact of that window.

## Results

**Ranking (fitted geometric decay rate $r$, param=0.1, depth=1, smaller = faster decay):**

| Channel | Fitted $r$ | $R^2$ |
|---|---|---|
| bit_flip | 0.8977 | 0.9996 |
| depolarizing | 0.9271 | 0.9999 |
| phase_flip | 0.9367 | 0.9915 |
| amplitude_damping | 0.9534 | 0.9993 |
| phase_damping | 0.9767 | 0.9997 |

**Bit-flip is the fastest-decaying channel.** Its fidelity at param=0.1 goes
$0.8200 \to 0.7300 \to 0.6562 \to 0.5905 \to 0.5314$ for $n=2$–$6$, and the extension sweep confirms
the fitted rate correctly predicts $n=7,8$: predicted fidelity $0.4771$/$0.4282$ vs. actual
$0.4783$/$0.4305$ — a small, honestly-reported gap, not exact, but close enough to confirm the
decay really is geometric over this range rather than a fitting coincidence.

**The closed forms hold exactly.** For phase-flip, $F(n)=\frac{1+(1-2p)^n}{2}$; for phase damping,
$F(n)=\frac{1+(\sqrt{1-\lambda})^n}{2}$ — both verified against simulation to floating-point
precision at $n=2,4,6,8$. Both asymptote to fidelity $0.5$, never below.

**Phase damping's $R^2$ is deceptively high (0.9997) within the $n\le6$ window** — its per-qubit
factor $\sqrt{1-p}\approx0.949$ (at $p=0.1$) is so close to 1 that its approach to the $0.5$ floor is
barely visible in that range, making it look like an ordinary, merely-slow decaying channel. This is
the central refinement this analysis contributes beyond Part 3's framing: phase damping is not "the
slowest-decaying channel" in the same sense bit-flip/depolarizing/amplitude-damping decay — it is
mechanistically identical to phase-flip (diagonal, population-preserving), just with a slower
approach to the same $0.5$ floor. Phase-flip's own $R^2$ (0.9915) — the lowest of the five — is a
visible symptom of this: a curve genuinely asymptoting to $0.5$ is not truly a straight line in
log-fidelity space, so the geometric-fit model is a good but imperfect local approximation for it.

## Limitations

- **Qubit count still capped (~8–14).** Density-matrix simulation is $O(4^n)$; the extension sweep
  reaches direct simulation only to $n=8$. The theory-vs-simulation plot in the notebook shows the
  two dephasing channels out to $n=14$, but only via their closed form, not new simulation — the
  other three channels have no such shortcut and were not extrapolated past $n=8$.
- **Only depth=1 studied in the new extension sweep.** Part 3's own depth=1/2/4 data (used for the
  main ranking) suggests the bit-flip-fastest ranking is stable across depth, but the targeted
  extension sweep itself does not re-verify this at $n=7,8$.
- **Closed form derived rigorously only for the two diagonal channels.** Bit-flip, depolarizing,
  and amplitude damping are reported as high-quality empirical geometric fits within the studied
  window, not proven closed forms — deriving one would require tracking how Hamming-weight sectors
  mix under repeated non-diagonal channel applications, which is non-trivial for $n>1$ qubits and
  was out of scope here.
- **Specific to GHZ states.** These findings describe one entangled-state family under independent,
  identical per-qubit noise; other states (W states, cluster states) or correlated noise models
  could show a different channel ranking under the same five channels.

## Conclusion

Bit-flip noise degrades GHZ-state fidelity fastest, and does so because it is not diagonal in the
computational basis — it genuinely destroys the populations that make up half of GHZ fidelity, with
no floor. Phase-flip and phase damping, by contrast, can never push GHZ fidelity below $0.5$: both
are diagonal channels that only ever damp coherence, a hard mathematical floor rather than a
tendency. This distinction — which channels can and cannot destroy which part of a quantum state —
is precisely the kind of noise characterization question a real adaptive-noise-modelling pipeline
needs answered before it can decide where mitigation effort actually pays off, and it generalizes a
finding already seen at small scale in Part 4 (phase-flip's invisibility to the repetition code) to
an arbitrary-size entangled state.
