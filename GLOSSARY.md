# Glossary

### Model Fidelity
How accurately a model reflects the **true** environment dynamics.

- **High fidelity** → model closely matches reality  
- **Low fidelity** → model contains approximation errors  

---

### Fidelity Regime
A qualitative category describing the level of model accuracy in a setting.

- **Exact** → true model is known (no estimation error)  
- **Near-exact** → model can be learned with very low error  
- **Approximate / low-fidelity** → model contains structural inaccuracies  

---

### Discretization
The process of mapping continuous variables into a finite set of bins.

Used to make infinite state spaces compatible with tabular methods.

---

### Aliasing
A consequence of discretization where multiple distinct states map to the same discrete representation.

---

### Irreducible Model Noise
Error introduced by limitations of the model representation (e.g., discretization).

Unlike estimation error, this **cannot be reduced with more data**, because the representation itself is lossy.

---

### Model-Based RL
Methods that explicitly learn or use a transition model of the environment.

---

### Model-Free RL
Methods that learn value functions or policies directly without modeling environment dynamics.

---

#### Clarification (important)

The distinction is about **what the algorithm does**, not whether the model is given.

- **Model-based (given model)**  
  The transition model is known in advance and used directly for planning  
  (e.g., Value Iteration).

- **Model-based (learned model)**  
  The model is estimated from experience, then used for planning  
  (e.g., ModelBasedAgent).

- **Model-free**  
  No model is ever built; the agent learns values/policies directly from transitions  
  (e.g., SARSA, Q-learning, TD(0)).

**Key takeaway:**  
Model-based = *build/use a model*  
Model-free  = *no model at all*