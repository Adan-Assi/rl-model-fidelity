# 🧱 Environments in This Project

## 🎯 Core Idea (TL;DR)

An environment is a **hidden Markov Decision Process (MDP) sampler**.

The agent never sees the true dynamics, it only observes transitions:

**$(s_t, a_t) \rightarrow (s_{t+1}, r_t)$**

All learning is done from these samples.

---

## 🎮 Intuition (Game in the Dark)

Think of a game running in the background:

- The game has full internal rules (physics, rewards, transitions)
- The player cannot see the rules or code
- The player only sees:
  - current state (screen)
  - reward (score)
  - next state after actions (pressing a button / moving a joystick)

So learning happens by **interaction, not inspection**.
> You learn the game by playing it, not by reading how it works internally.

---

## 🔗 Interface (What Every Environment Provides)

All environments expose only:

### 1. reset()
Start a new episode:

$s_0$

### 2. step(action)
Return a sampled transition:

$(s_{t+1}, r_t, done)$

---

## 🧠 Key Principle

The environment is a **black-box sampler of an MDP**:

> It generates experience, but never exposes the model itself.

So agents only learn from:

$(s, a, r, s')$

and not from:

- transition probabilities
- reward functions
- full state graph

---

## 📊 Why This Design Exists

This interface allows controlled comparison of:

- Model-Free RL (Q-learning, SARSA)
- Model-Based RL (learned model + planning)
- Exact Planning (Value Iteration with known model)

across environments with different **model fidelity levels**.

---

## 📉 True Model vs Learned Model

There are always two layers:

### 1. True environment model (hidden)
- Exists inside the simulator
- Never accessible

### 2. Learned model (optional)
- Constructed by model-based agents
- Estimated only from samples

---

## 🔥 Key Insight

Even when a true model exists, it is never directly used.

> All algorithms must learn from experience, not from the environment description.

This is what makes model-based RL meaningful here: it must reconstruct structure from data.

---

## 🧩 Connection to `base_env.py`

This interface enforces:

- no access to transition probabilities
- no access to reward function
- only `reset()` and `step()`

So all RL methods operate under the same constraint:
learning from sampled experience only.

---

## 🎮 Environment Notes

### 🃏 Blackjack (Simplified)

This environment intentionally uses a standard RL simplification:

- Ace is always worth 11 (no *usable ace*)
- State = gambler sum only

This keeps the state space small and the MDP clean and fully tabular,  
at the cost of deviating from real Blackjack rules.