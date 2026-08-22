# RL Under Model Mismatch: Model-Based vs Model-Free

## Overview
This project studies how **model fidelity** affects the performance gap between **model-based** and **model-free** reinforcement learning.

Instead of comparing algorithms in isolation, we introduce a **controlled axis of variation**:
> how accurate (or inaccurate) the environment model is.

We evaluate how algorithm behavior changes across:
- Exact known models
- Small, highly estimable models
- Noisy, discretized approximations

We study this through three paired experimental setups comparing model-based and model-free RL across different environments and fidelity regimes.

---

## Research Question
**How does the performance gap between model-based and model-free RL change as the fidelity of the learned environment model degrades?**

---

## Experimental Unit ("Pair")

Each pair is a controlled experimental comparison between a model-based RL agent and a model-free RL agent evaluated within the same environment.

The performance gap between the two agents is measured under identical environment dynamics, allowing isolation of the effect of model fidelity on learning performance.

---

## Measurement Design

The hypothesis is tested via three complementary comparisons, sampled at
different points and resolutions along the model-fidelity axis:

- **Pair 1 (FrozenLake)** — exact model, discrete regime
- **Pair 2 (Blackjack)** — near-exact model, discrete regime
- **Pair 3 (CartPole)** — continuous sweep over discretization granularity,
  enabling a direct within-pair gap-vs-fidelity curve

Each pair directly measures the model-based/model-free gap within its own
environment. Pair 3's swept design additionally allows that gap to be
tracked as fidelity varies continuously, which is the most direct test of
the hypothesis' shape (does the gap shrink smoothly, or only appear at
the extremes?). See `evaluation/METRICS_DESIGN.md` for how performance is
measured and normalized within and across pairs.

---

## Environments & Fidelity Axis

| Environment | Fidelity Level | Key Property |
|------------|---------------|-------------|
| FrozenLake | Exact | True transition probabilities are known |
| Blackjack | Near-exact | Small state space → can be estimated accurately |
| CartPole (discretized) | Approximate | Continuous states → discretization introduces noise |

### Note: Discretization in CartPole

The underlying state is **continuous**:
- pole angle = 0.013 radians  
- velocity = 0.728  
- etc.

Tabular methods cannot operate over infinitely many states, so the state space is discretized into finite bins. For example:

- angle:
  - [-0.2, -0.1] → bin 1  
  - [-0.1, 0] → bin 2  
  - [0, 0.1] → bin 3  

- velocity:
  - discretized similarly  

Each state is then represented as:

(angle_bin = 3, velocity_bin = 5, ...)


This makes the state space finite, enabling tabular methods.

However, discretization introduces aliasing and irreducible model noise (see [Glossary](./GLOSSARY.md)), which is why CartPole represents the low-fidelity regime in this study.

---

## Algorithms

We compare four algorithm classes:

### Model-Based
- **Value Iteration** (exact model)
- **Estimated Model + Value Iteration**

### Model-Free
- **Q-Learning** (off-policy)
- **SARSA** (on-policy)

All algorithms are implemented from scratch in a **shared interface**, allowing fair comparison across environments.

> See [Glossary](./GLOSSARY.md) for definitions of key terms (model fidelity, aliasing, etc.).

---

## Evaluation Framework (Core Contribution)

Rather than focusing on raw performance alone, we evaluate along multiple axes:

### 1. Sample Efficiency
- Episodes required to reach convergence
- Learning speed across environments

### 2. Stability Across Seeds
- Variance in final performance over multiple random seeds

### 3. Robustness to Model Degradation
We explicitly *break the model* in controlled ways:
- Inject noise into FrozenLake transitions
- Modify Blackjack rules after training
- Vary discretization granularity in CartPole

### 4. Failure Modes
We categorize *how* each algorithm fails:
- Model-based → errors from poor estimates in low-visit states
- Model-free → insufficient exploration / slow convergence

---

## Key Hypothesis

- When the model is **exact**, model-based methods dominate
- When the model is **small and estimable**, the gap shrinks
- When the model is **noisy or approximate**, model-based advantage disappears (or reverses)

---

## Project Structure
```txt
rl-model-mismatch/
│
├── algorithms/
│ ├── value_iteration.py
│ ├── q_learning.py
│ ├── sarsa.py
│ └── model_based_agent.py
│
├── environments/
│ ├── frozenlake_wrapper.py
│ ├── blackjack_env.py
│ └── cartpole_discretized.py
│
├── evaluation/
│ ├── experiment_runner.py
│ ├── metrics.py
│ └── plotting.py
│
├── results/
│ ├── figures/
│ └── tables/
│
└── README.md
```

---

## Results (Preview)

*(To be populated after experiments)*

Planned outputs:
- Learning curves (reward vs episodes)
- Variance across seeds
- Performance vs model fidelity plots
- Failure case breakdowns

---

## Key Takeaways (Expected)

- Model-based RL is **highly sensitive to model error**
- Model-free RL is **slower but more robust**
- The “better” paradigm depends on **how accurate your model actually is**


## Tech Stack
- Python
- NumPy
- Gymnasium
- Matplotlib / Pandas


## Academic Integrity Note
All algorithms are **reimplemented from scratch** based on theory and pseudocode.
No assignment code or starter files are reused.


## Resume Bullet

> Designed a comparative study of model-based vs. model-free RL under varying environment-model fidelity, evaluating sample efficiency, stability, and failure modes across multiple environments.


## Future Work
- Add function approximation (DQN)
- Extend to partially observable environments (POMDPs)
- Analyze exploration strategies under model mismatch


## How to Run

```bash
pip install -r requirements.txt
python evaluation/experiment_runner.py
```

## Author

Adan Assi  
B.Sc. Computer Science, Tel Aviv University