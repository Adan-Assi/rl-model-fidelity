"""
Q-learning (model-free, off-policy TD control)
---------------------
Q-learning learns the optimal action-value function Q*(s,a) directly from sampled
transitions, without access to an environment model.

This is the model-free counterpart of Value Iteration:
instead of computing expectations over p(s'|s,a), we update from a single sampled transition:

    (s_t, a_t, r_t, s_{t+1})

using the TD (temporal-difference) update:

    Q(s_t, a_t) ← Q(s_t, a_t)
                  + α [ r_t + γ max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) ]

------------------------------------------------------------
Key idea

- The target:
      r_t + γ max_{a'} Q(s_{t+1}, a')
  is a sampled approximation of the Bellman optimality backup.

- Each update is noisy, but repeated updates average out over time,
  and push Q_t toward Q*, under standard stochastic approximation conditions.

------------------------------------------------------------
Value Iteration vs Q-learning

- Value Iteration:
    uses full expectation over p(s'|s,a)

- Q-learning:
    uses a single sampled transition (s_t, a_t, r_t, s_{t+1})

So instead of planning with the model, we learn directly from experience.

------------------------------------------------------------
Exploration (ε-greedy)

Actions are chosen using an ε-greedy policy over Q:

    with probability ε       → random action (exploration)
    with probability 1 - ε   → argmax_a Q(s,a) (exploitation)

ε typically starts high and decays over time, so early learning explores
more, and later behavior becomes increasingly greedy.

Exploration is necessary so that all state-action pairs are sufficiently visited;
otherwise the agent may overfit to early estimates and miss better actions.

------------------------------------------------------------
Convergence (classical setting)

For convergence to Q*, two conditions are required:

1. Every (s,a) pair is visited infinitely often
2. Learning rates satisfy Robbins-Monro conditions:

    Σ α_t(s,a) = ∞
    Σ α_t²(s,a) < ∞

These ensure:
- enough exploration over time
- but decreasing variance so updates stabilize

------------------------------------------------------------
Practical note (this implementation)

This implementation uses a constant learning rate α.

So:
- it does not satisfy the full theoretical assumptions
- but works as a practical stochastic approximation method
- converging to a neighborhood of Q*, depending on step size

This is a standard trade-off in practice between stability and strict convergence guarantees.
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