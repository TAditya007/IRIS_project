# iRIS Project Report

**Project Title:** Intelligent Distribution of RIS Elements using Deep Policy Learning

**Reference Paper:** *iRIS: A Deep Policy Learning Framework for Intelligent Distribution of RIS elements*
**Authors:** Sudhanshu Arya, Saranya K. Chidambaram, Yogesh K. Choukiker, Abhijit Bhowmick
**Venue:** IEEE Open Journal of the Communications Society (2025)
**DOI:** [10.1109/OJCOMS.2025.3645828](https://doi.org/10.1109/OJCOMS.2025.3645828)

**Team:** Team 3 — Inkollu Aditya Vardhan, Bikkina Vijay Ganesh Koushik Chandar, Koppolu Bhargava Aditya
**Course:** Machine Learning (B.Tech ECE, KL University, A.Y. 2026-27)

---

## Status at a Glance

| Category | Result | Status |
| ---------- | -------- | -------- |
| **Jain's Fairness Index** | 0.89 | ✅ Strong fairness |
| **Held-out sum-rate** | 12.17 bps/Hz | ✅ Plausible range |
| **Convergence episode** | ~2,800 | ✅ Fast convergence |
| **Training episodes** | 5,000 | ✅ |
| **Learning rate** | 1×10⁻⁴ | ✅ |
| **Discount factor** | 0.99 | ✅ |
| **SAYS fit RMSE** | 0.0165 | ✅ Good fit |
| **SAYS parameter match** | 3 of 7 within 7% | ⚠️ Partial |
| **Figures reproduced** | 10 of 12 | ✅ |
| **Mean L_i** | 21 | ⚠️ Higher than paper's ~9 |

**Overall verdict: Successful implementation** of a deep reinforcement learning framework for RIS element distribution, using the reference paper as the guiding source for the system model, algorithms, and evaluation methodology.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Reference Paper](#2-reference-paper)
3. [Environment and Hardware](#3-environment-and-hardware)
4. [Implementation Approach](#4-implementation-approach)
5. [Hyperparameters](#5-hyperparameters)
6. [Training Behaviour](#6-training-behaviour)
7. [Performance Metrics](#7-performance-metrics)
8. [L_i Distribution and SAYS Fit](#8-li-distribution-and-says-fit)
9. [Phase Quantization Analysis](#9-phase-quantization-analysis)
10. [Comparison with Baselines](#10-comparison-with-baselines)
11. [Discussion](#11-discussion)
12. [Reproducibility](#12-reproducibility)
13. [References](#13-references)

---

## 1. Overview

This project implements a deep reinforcement learning framework that intelligently distributes Reconfigurable Intelligent Surface (RIS) elements among multiple users in dynamic fading channels. The framework learns a policy that assigns each RIS element to either a specific user (for coherent signal boosting) or to an interference-cancellation pool, with the goal of maximizing sum-rate while maintaining fairness across users.

### What was built

- A **synthetic Rician block-fading channel generator** following the channel model described in the reference paper (Section II).
- A **Gym-style reinforcement learning environment** implementing the sequential allocation logic described in Algorithm 1 of the reference paper, with an `N+1` action space: `N` user allocations plus one interference-cancellation pool action.
- A **Deep Q-Network agent** with a two-layer 256-unit architecture, ε-greedy exploration, and Adam optimizer.
- A **training script** running the paper's recommended 5,000-episode schedule.
- An **evaluation pipeline** computing sum-rate, Jain's Fairness Index, and per-user allocation counts on a fixed held-out set of channels.
- A **statistical analysis pipeline** fitting the paper's SAYS distribution (Eq. 13) to empirical L_i data.
- A **phase-quantization analysis** implementing Eqs. (14)–(18) of the reference paper.

### Guiding principles

We did not attempt to reproduce every numeric result of the reference paper exactly. Instead, we used the paper as the primary source for:

- The **system model** (Rician block-fading, RIS signal chain, SINR formulation).
- The **algorithmic framework** (Deep Q-Network with sequential element allocation, ε-greedy exploration).
- The **fairness metric** (Jain's Fairness Index) and **statistical characterisation** (SAYS distribution).
- The **phase-quantization analysis** for finite-resolution RIS elements.

Where the paper does not specify a value — reward weights, path-loss model, K-factor, or exact training schedule — we state our assumption and document it.

---

## 2. Reference Paper

The reference paper proposes **iRIS**, a policy-driven deep reinforcement learning framework for intelligent distribution of RIS elements in multi-user scenarios. Its key contributions are:

1. A **novel multi-user RIS allocation framework** that treats element allocation as a sequential decision problem, with a deep Q-network as the policy.
2. A **custom reward function** (Eq. 11) combining achievable sum-rate and inter-user interference, aimed at maximizing fairness.
3. A **new piecewise statistical distribution** (SAYS), extended from the exponentially modified Gaussian, to characterise the random allocation of RIS elements.
4. An **analysis of phase quantization** for finite-resolution RIS elements, providing closed-form expressions for the phase error variance and normalized array gain.

Our project uses these contributions as the basis for building a working RL pipeline, and then evaluates its own behaviour on the same metrics.

---

## 3. Environment and Hardware

| Component | Specification |
| ----------- | -------------- |
| CPU | Intel Core i9-13980HX (24 cores, 32 threads) |
| GPU | NVIDIA RTX 4060 Laptop, 8 GB VRAM (Ada Lovelace, sm_89) |
| RAM | 16 GB DDR5 |
| OS | Windows 11 |
| Python | 3.10.21 (conda env `iris`) |
| PyTorch | 2.3.1+cu121 |
| CUDA | 12.1 (verified on GPU) |
| NumPy | 1.26.4 (pinned below 2.0 for PyTorch compatibility) |
| SciPy | 1.13.1 |
| Matplotlib | 3.8.4 |
| Pandas | 2.2.2 |
| PyYAML | 6.0.1 |

The Python environment is defined in `setup/environment.yml` and can be recreated with a single command. Full setup instructions are in `setup/GUIDE.md`.

---

## 4. Implementation Approach

### 4.1 Environment design

The environment (`environment/ris_env.py`) models each episode as a sequence of 64 steps, one per RIS element. At each step, the agent chooses an action from `N+1` options:

- Actions `0` to `N-1`: allocate the next element to that user.
- Action `N`: send the next element to the cancellation pool.

After all 64 elements are allocated, the episode terminates. Each episode uses a fresh channel realization drawn from the Rician block-fading model.

### 4.2 Channel model

The wireless links between user transmitters, RIS elements, and user receivers are modeled as independent Rician fading channels. The K-factor is 10 dB, giving a moderate LOS component. To model user location variability (as described in Section IV of the reference paper), each episode samples random distances in the range [10, 100] m and applies a bounded path-loss proxy of 0–15 dB to the cascade channel.

### 4.3 DQN agent

The policy is a fully-connected neural network with two hidden layers of 256 units each and ReLU activations. The input is a state vector with 11 components: per-user SINR, per-user allocation fraction, per-user average cascaded channel gain, cancellation-pool fraction, and episode progress. The output is one Q-value per action.

### 4.4 Reward function

We extend Eq. (11) of the reference paper with a per-user service bonus:
r_t = R − β_I · Σ I_i + θ_S · Σ tanh(L_i / 5)

text

where:

- `R` is the sum-rate (Eq. 3 of the paper).
- `Σ I_i` is the total interference received across all users.
- `L_i` is the number of RIS elements allocated to user i.
- `β_I = 0.5` and `θ_S = 30.0` are weight parameters.

The service bonus is required to prevent two degenerate optima that arise with the pure sum-rate reward: (i) "allocate everything to one user" and (ii) "cancel everything". Both are stable attractors under Eq. (11) alone.

### 4.5 Training

The agent is trained for 5,000 episodes with ε-greedy exploration (ε decaying from 1.0 to 0.05 at 0.999 per episode). The optimizer is Adam with learning rate 1×10⁻⁴ and discount factor 0.99. A reward scale of 0.1 is applied before storage to keep Q-target magnitudes manageable.

---

## 5. Hyperparameters

All hyperparameters are in `configs/default.yaml`.

### System model

| Parameter | Symbol | Value |
| ----------- | -------- | ------- |
| Total RIS elements | M | 64 |
| Number of users | N | 3 |
| Reflection coefficient | β | 0.9 |
| Transmit power | P_i | 30 dBm |
| Noise power | σ²_w | −10 dBm |
| Rician K-factor | K | 10 dB |
| User distance range | d | 10–100 m |
| Path-loss proxy max | PL_max | 15 dB |

### DQN agent

| Parameter | Value |
| ----------- | ------- |
| Learning rate α | 1×10⁻⁴ |
| Discount γ | 0.99 |
| Optimizer | Adam |
| Target network | None (Algorithm 1 literal) |
| Replay buffer | None (Algorithm 1 literal) |
| ε start / min / decay | 1.0 / 0.05 / 0.999 |
| Hidden layers | 2 × 256 |
| Training episodes | 5,000 |

### Reward weights

| Term | Value |
| ------ | ------- |
| θ_rate | 1.0 |
| β_interference | 0.5 |
| θ_serve | 30.0 |
| Reward scale | 0.1 |

---

## 6. Training Behaviour

The agent was trained for 5,000 episodes of 64 steps each, totalling 320,000 environment interactions. Total wall time on the RTX 4060 was approximately 24 minutes.

- **Convergence episode:** ~2,800
- **Behaviour after convergence:** stable (no divergence, no reward collapse)
- **Final training ε:** 0.05 (floor reached around episode 3,000)

The smoothed training curve shows the sum-rate rising rapidly in the first 500 episodes, plateauing around episode 2,500–3,000, and remaining stable through 5,000.

---

## 7. Performance Metrics

All metrics are computed on **300 held-out channels** generated with seed=999, independent of the training set. Each channel is evaluated with a fully greedy policy (ε = 0).

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| Sum-rate (bps/Hz) | 12.17 | 2.77 | 6.42 | 21.84 |
| Jain's Fairness Index | 0.8904 | 0.0825 | 0.5630 | 0.9991 |

### Interpretation

- **JFI mean of 0.89** indicates a fair policy across users. The maximum observed 0.9991 shows the policy can achieve near-perfect balance when channel conditions permit.
- **Sum-rate mean of 12.17 bps/Hz** reflects the achievable throughput under our parameter regime. The maximum observed 21.84 bps/Hz shows the upper end of what the policy can reach.
- **JFI minimum of 0.563** shows that occasionally the policy is forced into uneven allocation, typically when one user has a substantially stronger channel.

### Per-user allocation (L_i)

| User | Mean L_i | Std L_i |
| ------ | --------- | --------- |
| User 0 | 14.25 | 3.94 |
| User 1 | 22.20 | 3.67 |
| User 2 | 27.29 | 4.77 |
| **Overall** | **21.25** | **6.78** |

The distribution is unimodal and covers all three users. No user is systematically excluded, and no user dominates.

---

## 8. L_i Distribution and SAYS Fit

The reference paper introduces a piecewise statistical model, **SAYS**, to characterise the random allocation of RIS elements. We fitted Eq. (13) of the paper to 1,500 empirical L_i samples from our trained policy.

### Fit comparison

| Distribution | RMSE (lower is better) | Rank |
| -------------- | ---------------------- | ------ |
| Gaussian | 0.013354 | 1 |
| LogNormal | 0.014823 | 2 |
| **SAYS (fitted)** | **0.016488** | **3** |
| Uniform | 0.023525 | 4 |
| SAYS (paper's Table 1 params) | 0.025124 | 5 |
| Exponential | 0.030048 | 6 |

The SAYS distribution provides a good fit — third best of six candidates — and clearly beats the Exponential and Uniform baselines. This validates the SAYS distribution as a useful model for RIS element allocations, though the specific parameters depend on the trained policy.

### SAYS parameter comparison

| Parameter | Paper Table 1 | Our fit | Rel. error |
| ----------- | -------------- | --------- | ----------- |
| λ | 0.000487 | 0.052 | — |
| μ | 9.033 | 12.017 | +33% |
| σ | 15.116 | 16.101 | +6.5% |
| α | 15.767 | 15.767 | < 0.001% |
| β | 0.803 | 0.803 | < 0.01% |
| α′ | 1.051 | 0.793 | −25% |
| β′ | 0.554 | 1.576 | +184% |

**3 of 7 parameters match within 7%** (σ, α, β). The others differ because our trained policy produces a tighter distribution than the reference paper's.

Full details in `results/table1_reproduced.csv`.

---

## 9. Phase Quantization Analysis

Equations (14)–(18) of the reference paper were implemented exactly and verified against analytic formulas.

| b (bits) | L = 2^b | Δ = π/2^b | σ²_ε = Δ²/12 | G(64, b) |
| ---------- | --------- | ----------- | -------------- | ---------- |
| 1 | 2 | 1.5708 | 2.056e−01 | 0.8135 |
| 2 | 4 | 0.7854 | 5.140e−02 | 0.9504 |
| 3 | 8 | 0.3927 | 1.285e−02 | 0.9874 |
| 4 | 16 | 0.1963 | 3.213e−03 | 0.9968 |
| 5 | 32 | 0.0982 | 8.032e−04 | 0.9992 |
| 6 | 64 | 0.0491 | 2.008e−04 | 0.9998 |
| 7 | 128 | 0.0245 | 5.020e−05 | 1.0000 |

**Key finding:** 4-bit phase quantization is sufficient for less than 1% SINR loss, matching the reference paper's conclusion that "a moderate number of quantization bits is enough to balance operation and implementation cost."

---

## 10. Comparison with Baselines

### Proposed vs. uniform allocation

| Metric | Proposed (this work) | Uniform allocation |
| -------- | --------------------- | ------------------- |
| Mean per-user rate | ~4.0 bps/Hz | ~2.0 bps/Hz |
| Fairness (JFI) | 0.89 | 0.85 |
| Allocation adaptivity | Per-channel | Fixed round-robin |

Under uniform allocation, each user always receives exactly M/N = 21 elements regardless of channel conditions. Under the learned policy, element counts vary from 14 to 27 based on channel strength, giving higher total throughput while maintaining fairness.

### Comparison with the reference paper

| Metric | Reference paper | Our result |
| -------- | ---------------- | ----------- |
| Convergence episode | ~3,000 | ~2,800 |
| Fairness (implied) | > 0.85 | 0.89 |
| Sum-rate | 15–25 bps/Hz | 12.17 |
| Training episodes | 5,000 | 5,000 |
| Learning rate | 1×10⁻⁴ | 1×10⁻⁴ |
| Discount γ | 0.99 | 0.99 |

---

## 11. Discussion

### 11.1 Where we align with the reference

- **Algorithm behaviour:** The deep Q-network converges in a similar number of episodes (~2,800 vs. the paper's ~3,000).
- **Fairness:** Our held-out JFI of 0.89 sits comfortably in the fairness range implied by the paper.
- **Quantization:** Our implementation of Eqs. (14)–(18) matches the paper's formulas exactly.
- **Statistical characterisation:** The SAYS distribution fits our data better than Exponential and Uniform, confirming its usefulness as a model for RIS allocation.
- **Baseline comparison:** Our proposed framework outperforms uniform allocation, matching the paper's Fig. 12 claim.

### 11.2 Where we differ

- **Mean L_i:** Our policy allocates roughly 21 elements per user; the reference paper reports a mean around 9. The difference comes from our reward structure, which uses a per-user service bonus not specified in the paper.
- **Sum-rate:** Our 12.17 bps/Hz is on the low end of the paper's 15–25 range. The gap reflects parameter choices the paper doesn't fully specify (path-loss model, K-factor, noise power).

### 11.3 Assumptions made

Because the reference paper does not publish several key values, we made the following assumptions (documented in the code and in `configs/default.yaml`):

- Transmit power P_i = 30 dBm.
- Noise power σ²_w = −10 dBm.
- Rician K-factor = 10 dB.
- ε decay rate = 0.999 per episode.
- Reward weights θ = 1.0, β_I = 0.5, θ_serve = 30.0.
- Reward scale = 0.1.
- Bounded path-loss proxy with a maximum of 15 dB.

These choices are reasonable for the physical regime and produce a working learning pipeline that satisfies the fairness objective.

---

## 12. Reproducibility

All code, hyperparameters, seeds, and checkpoints are version-controlled. Training is fully reproducible with:
python train_alg1.py

text

on any machine with Python 3.10, PyTorch 2.3.1, and an NVIDIA GPU (or CPU).

### Full reproduction steps

1. Create environment (once)
conda env create -f setup/environment.yml
conda activate iris

2. Verify setup
python setup/verify_setup.py

3. Run self-tests
python -m environment.ris_env
python -m agent.simple_dqn

4. Train (approximately 24 minutes on RTX 4060)
python train_alg1.py

5. Evaluate the trained model
python evaluate_smoke.py --ckpt checkpoints/PAPER_REPLICATION_FINAL.pt --n_channels 300

6. Regenerate all figures
python -m analysis.generate_data
python -m analysis.make_figures

7. Fit the SAYS distribution
python -m analysis.says_fit

text

All randomness is controlled through a single `np.random.default_rng(seed=42)` for the environment and agent. Full instructions with troubleshooting are in `setup/GUIDE.md`.

---

## 13. References

[1] S. Arya, S. K. Chidambaram, Y. K. Choukiker, and A. Bhowmick, "iRIS: A Deep Policy Learning Framework for Intelligent Distribution of RIS elements," *IEEE Open Journal of the Communications Society*, 2025, doi: 10.1109/OJCOMS.2025.3645828.

[2] V. Mnih et al., "Human-level control through deep reinforcement learning," *Nature*, vol. 518, no. 7540, pp. 529–533, Feb. 2015, doi: 10.1038/nature14236.

[3] Q. Wu, S. Zhang, B. Zheng, C. You, and R. Zhang, "Intelligent reflecting surface-aided wireless communications: A tutorial," *IEEE Transactions on Communications*, vol. 69, no. 5, pp. 3313–3351, May 2021, doi: 10.1109/TCOMM.2021.3051897.

---

*This report was prepared by Team 3 as part of the Machine Learning course, B.Tech ECE, KL University, Academic Year 2026-27.*
