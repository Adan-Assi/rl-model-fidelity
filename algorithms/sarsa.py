"""
SARSA (on-policy, model-free TD control)
---------------------
SARSA learns the action-value function Q(s,a) directly from sampled
transitions, without access to an environment model.

It is very similar to Q-learning, but with one crucial difference:

    Q-learning uses the greedy next action (off-policy)
    SARSA uses the actual next action taken by the current policy (on-policy)

So SARSA learns:

    “What is the value of the policy I am actually following?”

instead of:

    “What would happen if I acted optimally from now on?”

------------------------------------------------------------
Core update rule (TD control)

From a sampled transition:

    (s_t, a_t, r_t, s_{t+1}, a_{t+1})

SARSA performs:

    Q(s_t, a_t) ← Q(s_t, a_t)
                  + α [ r_t + γ Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) ]

where the next action a_{t+1} is drawn from the same behavior policy
(e.g. ε-greedy w.r.t. Q_t).

------------------------------------------------------------
Policy (ε-greedy)

Actions are selected using an ε-greedy policy over Q:

    with probability ε       → random action (exploration)
    with probability 1 - ε   → argmax_a Q(s,a) (exploitation)

ε is typically decayed over time to reduce exploration gradually.

------------------------------------------------------------
Key difference from Q-learning

- Q-learning:
    uses max_a Q(s',a)
    → learns optimal policy independently of behavior

- SARSA:
    uses Q(s',a') where a' is actually taken
    → learns values of the behavior policy itself

This makes SARSA typically more conservative, since it accounts for
the effects of exploration during learning.
"""

import random
from typing import Dict, Tuple

from environments.base_env import Environment
from algorithms.exploration import epsilon_greedy, linear_epsilon_schedule


def sarsa_control(
    env: Environment,
    states,
    actions,
    num_episodes: int,
    alpha: float,
    gamma: float = 1.0,
    eps_start: float = 1.0,
    eps_end: float = 0.02,
    seed: int = 0,
):
    rng = random.Random(seed)
    # Start with an arbitrary initialization
    Q = {(s, a): 0.0 for s in states for a in actions}

    # Episode loop:
    # Each episode is one full trajectory of the environment,
    # starting from reset() and ending when done=True.
    # Inside each episode, multiple time steps (t=0,1,2,...) occur.
    for ep in range(num_episodes):
        epsilon = linear_epsilon_schedule(ep, num_episodes, eps_start, eps_end)

        # Initialize the environment and get the initial state
        s = env.reset()

        # Choose initial action a using ε-greedy policy
        a = epsilon_greedy(Q, s, actions, epsilon, rng)
        done = False

        # Time step loop (where “t” lives)
        while not done:
            s_next, r, done = env.step(a)

            # Terminal case (no bootstrap)
            if done:
                Q[(s, a)] += alpha * (r - Q[(s, a)])
                break
            
            # Next action (on-policy continuation: a_{t+1} = \pi(s_{t+1}; Q_t))
            a_next = epsilon_greedy(Q, s_next, actions, epsilon, rng)

            # TD target
            target = r + gamma * Q[(s_next, a_next)]

            # Update Q(s_t, a_t) using the TD update rule
            Q[(s, a)] += alpha * (target - Q[(s, a)])

            # Move to the next state and action
            s, a = s_next, a_next

    return Q

"""
After SARSA finishes learning, it produces a Q(s,a) table representing how good each action
is in each state.

To extract a policy from it, we define:
    π(s) = argmax_a Q(s,a)

i.e. for each state, we pick the action with the highest Q-value,
which becomes the learned greedy policy.
"""
def greedy_policy_from_Q(Q, states, actions):
    return {s: max(actions, key=lambda a: Q[(s, a)]) for s in states}
