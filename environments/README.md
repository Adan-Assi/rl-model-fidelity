# 🧱 Environments in This Project

## 🎯 Core Idea (TL;DR)

An environment is a **hidden Markov Decision Process (MDP) sampler**.

The agent never observes the true dynamics.  
Instead, it interacts through sampled transitions:

**$(s_t, a_t) \rightarrow (s_{t+1}, r_t)$**

All learning is based on these samples.

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

# 🎮 Environment Notes

## 🃏 Blackjack (Simplified)

This environment intentionally uses a standard RL simplification:

- Ace is always worth 11 (no *usable ace*)
- State = gambler sum only

This keeps the state space small and the MDP clean and fully tabular,  
at the cost of deviating from real Blackjack rules.

---

## 🃏 Blackjack (Simplified MDP)

This environment is a tabular MDP modeling a simplified version of Blackjack ('21').

The agent plays against a fixed house policy, where the goal is to maximize the probability of winning without exceeding 21.

- State = gambler's current sum (4–21)
- Actions = {HIT, STAND}
- Reward = 1 only if the agent wins the round, otherwise 0


### 🎴 Game Dynamics

At each step:

- **HIT**: the agent draws a card and updates its sum
- **STAND**: the agent stops, and the house plays its fixed policy

The episode terminates when:
- the gambler busts (>21), or
- the agent stands and the house resolves the game


### 🎲 Deck Models (Stochasticity Control)

The environment supports two card-drawing regimes:

- **Infinite deck (with replacement)**  
  Each draw is independent and identically distributed.

- **Finite deck (52-card shoe, without replacement)**  
  Cards are removed after drawing and the deck is reshuffled when empty.

This allows controlled comparison between stationary and non-stationary dynamics.


### 🧠 House Policy (Environment Dynamics)

The house follows a fixed deterministic policy:

- Start with two cards
- Keep drawing until reaching a threshold (`house_hit_threshold`)
- Stop once threshold is exceeded

This policy defines how the environment resolves outcomes after the agent stops acting.


### 🧩 Design Choices

- **Unified interface**: Implements `reset()` and `step()` consistent with all environments in this project.

- **Encapsulated episode logic**: All game mechanics (drawing, bust detection, house play, reward computation) are fully handled inside the environment.

- **Simplified state space**: Ace is always worth 11 (no usable ace), reducing complexity to a single integer state.

- **Controlled stochasticity**: Deck model and house threshold can be modified to vary difficulty and dynamics.


### 📊 Role in This Project

Blackjack serves as a **controlled stochastic decision-making environment** for studying reinforcement learning under unknown dynamics.

The agent interacts with the environment purely through sampled experience:

- 🧭 Model-Free RL (Q-learning, SARSA)
- 🧠 Model-Based RL (learned dynamics + planning)
- 📉 Evaluation under a fixed but hidden environment model

The true transition dynamics are not directly exposed, requiring agents to learn behavior from interaction rather than explicit rules.

---

## 🧊 FrozenLake (Stochastic Gridworld)

This environment is a wrapper around Gymnasium's `FrozenLake-v1` implementation, adapted to the project's unified `Environment` interface.

It’s a classic tabular MDP used for benchmarking RL algorithms.

The agent navigates a grid from start to goal while avoiding holes.

- State = discrete grid position
- Actions = {LEFT, DOWN, RIGHT, UP}
- Reward = 1 only on reaching the goal, otherwise 0


### 🌊 Stochastic Dynamics (Slippery Mode)

When `is_slippery=True`, the environment becomes stochastic:

- Intended actions may "slip" into neighboring directions
- Transitions are probabilistic rather than deterministic
- This turns the gridworld into a proper stochastic MDP

When `is_slippery=False`, transitions are deterministic.


### 🧠 Why FrozenLake is Special in This Project

FrozenLake is the only environment where the **true transition model is accessible**:

- Full transition probabilities are available via `P(s, a)`
- Exact MDP structure is known
- Enables exact planning methods (e.g., Value Iteration)


### 📊 Role in This Project

FrozenLake acts as the **high-fidelity baseline environment**, allowing comparison between:

- 🧭 Exact Planning (Value Iteration using true model)
- 📉 Model-Free RL (Q-learning, SARSA)
- 🧠 Model-Based RL (learned transition model)

This makes it the reference point for studying how access to the true model affects learning and planning performance.

---

## 🪢 CartPole (Discretized Control Task)

This environment models the classic cart-pole (inverted pendulum) problem in a discretized form.

The goal is to keep a pole balanced upright by moving a cart left or right.

- State = discretized version of:
  - cart position
  - cart velocity
  - pole angle
  - pole angular velocity
- Actions = {push left, push right}
- Reward = 0 while balancing, -1 on failure

### 🎯 Intuition (Balancing a Stick)

Imagine trying to **balance a stick on your hand**:

- If it falls left → move left  
- If it falls right → move right  

The cart is your hand, and the pole is the stick.

### ⚙️ Continuous Dynamics → Discrete States

The underlying system is continuous, but in this project it is discretized:

- Exact values are grouped into bins (ranges)
- The agent observes only a **coarse representation** of the state

So instead of precise physics, the agent sees a simplified version of reality.


### 🎚️ Why This Matters

Discretization controls how much detail the agent sees:

- **More bins** → more precise state → easier learning  
- **Fewer bins** → less information → harder learning  

This introduces approximation error into both learning and planning.


### 📊 Role in This Project

CartPole provides a setting where the environment is **structured but partially abstracted**:

- 🧭 Model-Free RL (Q-learning, SARSA)
- 🧠 Model-Based RL (learned dynamics + planning)
- 📉 Impact of state discretization on performance

It is used to study how **state representation fidelity** affects learning and planning.