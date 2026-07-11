# quantum-noise-lab

A practical quantum noise simulation and mitigation lab built with Qiskit — covering
fundamentals, noise models, noise-vs-fidelity experiments, simple error mitigation, and an
original mini research question, with reproducible tests and publication-quality plots throughout.

## Status

Milestone 0 (scaffold) and Part 1 (Qiskit fundamentals) complete.

## Quickstart

```bash
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest tests/
jupyter lab notebooks/01_qiskit_fundamentals.ipynb
```

## Repository structure

```
quantum-noise-lab/
  src/            # modular library: circuit builders, simulation helpers, visualization
  notebooks/      # teaching / exploratory notebooks
  experiments/    # reproducible, config-driven experiment scripts
  figures/        # generated plots
  tests/          # pytest unit tests
  docs/           # architecture notes, math derivations, experiment reports, interview Q&A
```

## Roadmap

- [x] Part 1: Qiskit fundamentals (Bell state, GHZ state, superposition, entanglement)
- [ ] Part 2: Quantum noise models (bit flip, phase flip, depolarizing, amplitude/phase damping)
- [ ] Part 3: Noise comparison experiments (fidelity, success probability, error rates)
- [ ] Part 4: Simple error mitigation (repetition code, majority vote)
- [ ] Part 5: Mini research question
- [ ] Part 6 (optional): ML-based fidelity/error/noise-type prediction

## License

MIT — see [LICENSE](LICENSE).
