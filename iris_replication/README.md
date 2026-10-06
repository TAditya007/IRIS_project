# iRIS Replication

Replication of *iRIS: A Deep Policy Learning Framework for Intelligent
Distribution of RIS elements* (IEEE OJ-COMS 2025).

## Result Summary
- Held-out JFI: 0.89 (paper implies > 0.85)
- Held-out sum-rate: 12.17 bps/Hz (paper: 15–25)
- Convergence episode: ~2800 (paper: ~3000)
- 10 of 12 figures reproduced

## Quick Start
```bash
conda env create -f setup/environment.yml
conda activate iris
python setup/verify_setup.py
python train_alg1.py
python -m analysis.make_figures
```

## Structure
- `environment/` — Rician channel, SINR, RIS env, quantization
- `agent/` — DQN agents (buffer, vanilla, and literal Algorithm 1)
- `analysis/` — figure generation, SAYS fitting
- `configs/default.yaml` — all hyperparameters
- `results/REPLICATION_REPORT.md` — full report
- `setup/` — environment setup scripts and guides

## Key Files
- `train_alg1.py` — literal implementation of the paper's Algorithm 1
- `results/REPLICATION_REPORT.md` — full replication write-up
- `commands.txt` — full audit trail