"""
Q-learning (model-free, off-policy)
---------------------
Learns the optimal action-value function Q*(s,a) directly from sampled
transitions, without access to the environment model (no `model` dict
anywhere in the file).

Key difference from Value Iteration:
instead of computing expectations over p(s'|s,a), we update from a single
observed transition at time t:

    (s_t, a_t, r_t, s_{t+1})

using the TD (temporal-difference) update:

    Q(s_t, a_t) ← Q(s_t, a_t)
                  + α [ r_t + γ max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) ]

Key idea:

- The term   r_t + γ max_{a'} Q(s_{t+1}, a')   is a sampled approximation
  of the Bellman optimality target at time t.
- Repeated noisy updates average out over time and drive Q_t → Q*, under the
  classical (decaying-stepsize) theorem stated below. With this file's fixed alpha,
  see the note at the end instead.

This is the model-free counterpart of Value Iteration:

- Value Iteration → uses the full model (expectation over all next states)
- Q-learning      → uses a SINGLE sampled transition
  (s_t, a_t, r_t, s_{t+1})

Exploration:
Actions are chosen using an ε-greedy policy:

    with probability ε       → explore (random action)
    with probability 1 - ε   → exploit (greedy action)

Exploration is necessary so that state-action pairs are visited sufficiently
often. Without it, the agent may keep choosing the actions that currently
look best and never discover that another action is better.

ε starts high → more exploration, and gradually decreases over episodes
→ behavior becomes more greedy.

For the classical Q-learning convergence guarantee: 1. every (s,a) pair must
be visited infinitely often, and 2. the learning rate must satisfy the
Robbins-Monro conditions:

    Σ α_t(s,a) = ∞
    Σ α_t²(s,a) < ∞

The first condition requires sufficient exploration; the second requires
the learning rate to decrease appropriately toward zero.

Note: This implementation uses a fixed `alpha`, which is a common practical
choice. Therefore, it does not satisfy the classical theorem's exact
step-size assumptions and does not have its guarantee of Q_t → Q*.

This is a deliberate practical trade-off: constant-step-size Q-learning
is a theoretically studied alternative to diminishing step sizes. It can
converge rapidly to a stationary distribution centered near Q*, with the
residual bias depending on the step size (e.g., Zhang & Xie, 2024).
"""

import random
 
from environments.base_env import Environment
from algorithms.exploration import epsilon_greedy, linear_epsilon_schedule
 
 
def q_learning(
    env: Environment,
    states,
    actions,
    num_episodes: int,
    alpha: float,
    gamma: float,
    max_steps_per_episode: int = 200,
    eps_start: float = 1.0,
    eps_end: float = 0.02,
    seed: int = 0,
):
    rng = random.Random(seed)

    # Initialize Q(s,a) for every state-action pair
    Q = {(s, a): 0.0 for s in states for a in actions}

    episode_rewards, episode_lengths = [], []
 
    for ep in range(num_episodes):
        s = env.reset()
        total_reward, steps = 0.0, 0

        # Compute ε for this episode using a linear decay schedule from eps_start to eps_end
        epsilon = linear_epsilon_schedule(ep, num_episodes, eps_start, eps_end)
 
        for _ in range(max_steps_per_episode):
            steps += 1

            # ----- Observe a transition (s_t, a_t, r_t, s_{t+1}) -----

            # Choose action a_t using the current ε-greedy policy
            a = epsilon_greedy(Q, s, actions, epsilon, rng)

            # Take a_t and observe r_t and s_{t+1}
            s_next, r, done = env.step(a)

            # ----- Update Q(s_t, a_t) using the TD update rule -----

            # Bellman optimality target: r_t + γ max_a' Q(s_{t+1}, a')
            best_next = max(Q[(s_next, a2)] for a2 in actions)

            # TD error = reward + discounted best next-state value - current Q-value
            td_error = r + gamma * best_next - Q[(s, a)]

            # Update only the visited (s_t, a_t) pair
            Q[(s, a)] += alpha * td_error

            total_reward += r

            # Move to the next state
            s = s_next

            if done:
                break

        # Record the total reward and length of this episode
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)

    # After sufficient exploration and updates, Q should approach Q*.
    return Q, episode_rewards, episode_lengths