# quantum-noise-lab

A practical quantum noise simulation and mitigation lab built with Qiskit — covering
fundamentals, noise models, noise-vs-fidelity experiments, simple error mitigation, an original
mini research question, and an ML fidelity-prediction extension benchmarked against physics, with
reproducible tests and publication-quality plots throughout.

## Status

Milestone 0 (scaffold), Part 1 (Qiskit fundamentals), Part 2 (quantum noise models), Part 3 (noise
comparison experiments), Part 4 (simple error mitigation), Part 5 (mini research question), and
Part 6 (ML fidelity prediction vs. physics baseline) complete.

## Quickstart

```bash
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest tests/
jupyter lab notebooks/01_qiskit_fundamentals.ipynb notebooks/02_noise_models.ipynb notebooks/03_noise_experiments.ipynb notebooks/04_error_mitigation.ipynb notebooks/05_mini_research_question.ipynb notebooks/06_ml_fidelity_prediction.ipynb
python experiments/run_noise_sweep.py          # regenerates experiments/results/noise_sweep.csv
python experiments/run_mitigation_sweep.py     # regenerates experiments/results/mitigation_sweep.csv
python experiments/run_ghz_decay_extension.py  # regenerates experiments/results/ghz_decay_extension.csv
python experiments/run_ml_fidelity.py          # regenerates experiments/results/ml_fidelity_metrics.csv, ml_fidelity_feature_importance.csv
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

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full system map: required QC/math
basics, data-flow diagrams, module reference, and repository structure in depth.

## Roadmap

- [x] Part 1: Qiskit fundamentals (Bell state, GHZ state, superposition, entanglement)
- [x] Part 2: Quantum noise models (bit flip, phase flip, depolarizing, amplitude/phase damping)
- [x] Part 3: Noise comparison experiments (fidelity, success probability, error rates)
- [x] Part 4: Simple error mitigation (repetition code, majority vote)
- [x] Part 5: Mini research question (which noise channel degrades GHZ fidelity fastest, and why)
- [x] Part 6 (optional): ML fidelity prediction vs. physics baseline (interpolation vs.
      extrapolation, feature importance)

## License

MIT — see [LICENSE](LICENSE).
