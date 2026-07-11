# quantum-noise-lab

A practical quantum noise simulation and mitigation lab built with Qiskit — covering
fundamentals, noise models, noise-vs-fidelity experiments, simple error mitigation, and an
original mini research question, with reproducible tests and publication-quality plots throughout.

## Status

Milestone 0 (scaffold), Part 1 (Qiskit fundamentals), Part 2 (quantum noise models), Part 3 (noise
comparison experiments), and Part 4 (simple error mitigation) complete.

## Quickstart

```bash
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest tests/
jupyter lab notebooks/01_qiskit_fundamentals.ipynb notebooks/02_noise_models.ipynb notebooks/03_noise_experiments.ipynb notebooks/04_error_mitigation.ipynb
python experiments/run_noise_sweep.py       # regenerates experiments/results/noise_sweep.csv
python experiments/run_mitigation_sweep.py  # regenerates experiments/results/mitigation_sweep.csv
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
- [x] Part 2: Quantum noise models (bit flip, phase flip, depolarizing, amplitude/phase damping)
- [x] Part 3: Noise comparison experiments (fidelity, success probability, error rates)
- [x] Part 4: Simple error mitigation (repetition code, majority vote)
- [ ] Part 5: Mini research question
- [ ] Part 6 (optional): ML-based fidelity/error/noise-type prediction

## License

MIT — see [LICENSE](LICENSE).
