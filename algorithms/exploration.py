"""
Shared epsilon-greedy exploration for table-based algorithms (e.g., Q-learning, SARSA).

Therefore, these algorithms use the same exploration strategy so that their comparison
is fair: differences in learning behavior or performance can be attributed
to their update rules (e.g. off-policy vs. on-policy), rather than to different
exploration schemes.
"""

import random
from typing import Dict, Sequence, Tuple


def epsilon_greedy(Q: Dict[Tuple, float], s, actions: Sequence, epsilon: float,
                    rng: random.Random):
    """
    Choose an action using an epsilon-greedy policy.

    With probability epsilon, choose a uniformly random action.
    Otherwise, choose a greedy action according to Q.

    If multiple actions have the same highest Q-value, break the tie
    uniformly at random. This avoids systematically favoring whichever
    action happens to appear first, which is especially important early
    in training when Q-values may all be equal (e.g. all zeros).
    """
    if rng.random() < epsilon:
        return rng.choice(actions)
    q_values = [Q[(s, a)] for a in actions]
    best = max(q_values)
    best_actions = [a for a, q in zip(actions, q_values) if q == best]
    return rng.choice(best_actions)


"""
Linear epsilon decay schedule.

ε controls the exploration-exploitation tradeoff in an ε-greedy policy:

    with probability ε       → explore (choose a random action)
    with probability 1 - ε   → exploit (choose the current best action)

We start with a high ε because the Q-values are initially unreliable, so
the agent needs to explore and discover which actions are good. As learning
progresses, ε decreases so the agent increasingly exploits what it has learned.

Sufficient exploration is important for Q-learning/SARSA: their convergence
theory requires state-action pairs to be visited sufficiently often.

This function linearly decreases ε from eps_start to eps_end over the
training episodes.
"""
def linear_epsilon_schedule(episode: int, num_episodes: int,
                             eps_start: float = 1.0, eps_end: float = 0.02) -> float:
    frac = episode / max(1, num_episodes - 1)
    return eps_start + (eps_end - eps_start) * frac
