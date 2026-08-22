"""
TD(0) Policy Evaluation (model-free, on-policy prediction)
----------------------------------------------------------

TD(0) estimates the value function V^π(s) for a fixed policy π directly
from sampled experience, without requiring a model of the environment.

It's a temporal-difference method that combines:
- Monte Carlo ideas (learning from sampled returns)
- Dynamic Programming ideas (bootstrapping)

----------------------------------------------------------
Core idea (TD learning)

Instead of computing the full expectation:

    V(s) = E_π[r + γ V(s')]

TD(0) uses a single sampled transition:

    (s_t, a_t, r_t, s_{t+1})

and performs a bootstrap update:

    target = r_t + γ V(s_{t+1})

----------------------------------------------------------
Update rule

TD error:

    δ_t = r_t + γ V(s_{t+1}) - V(s_t)

Value update:

    V(s_t) ← V(s_t) + α_t(s_t) · δ_t

Equivalent form:

    V(s_t) ← (1 - α_t) V(s_t) + α_t [r_t + γ V(s_{t+1})]

----------------------------------------------------------
Key properties

- Model-free: does not require p(s'|s,a) or r(s,a)
- Online: updates after every transition
- Bootstrapped: uses current estimate V(s_{t+1})
- On-policy: evaluates a fixed policy π (no off-policy correction here)

----------------------------------------------------------
DP vs TD(0)

- Value Iteration (DP):
    uses full expectation over next states (requires model)

- TD(0):
    replaces expectation with a single sampled transition

----------------------------------------------------------
Convergence (theoretical setting)

TD(0) converges to V^π under standard stochastic approximation
conditions:

1. All states are visited sufficiently often under policy π
2. Step sizes satisfy Robbins-Monro conditions:
       Σ α_t(s) = ∞
       Σ α_t²(s) < ∞

In practice, constant step sizes are often used,
leading to convergence to a neighborhood of V^π.

----------------------------------------------------------
Notes

- "0" refers to using zero additional real rewards before bootstrapping.
- This implementation assumes a fixed policy π (no exploration logic included).

----------------------------------------------------------
Precondition on `states` (important)

V is initialized once, up front, for exactly the state set passed in via
`states`, every V(s) and V(s_next) lookup during training uses direct
dict indexing, not a defensive .get(..., default). This means `states`
must be the environment's true, complete enumeration of non-terminal
states (e.g. FrozenLakeEnv.states, BlackjackEnv.states,
CartPoleDiscretizedEnv.states restricted to non-failure indices),
not a hand-picked or partial subset.

If `states` is incomplete or mismatched against what the environment
actually produces, this will raise a KeyError rather than silently
falling back to a default value. A loud KeyError on a mismatched state set is
preferable to a quietly-corrupted value function.
"""

from typing import Callable, Dict, Hashable, Iterable

from environments.base_env import Environment


def td0_evaluate(
    env: Environment,
    policy: Callable[[Hashable], int], # on-policy action selection function
    states: Iterable[Hashable],
    num_episodes: int,
    alpha: float,
    gamma: float = 1.0,
) -> Dict[Hashable, float]:
    """Estimate V^policy(s) for each s in `states` via tabular TD(0).

    `states` must be the environment's complete, true set of non-terminal
    states.
    """

    V = {s: 0.0 for s in states}

    for _ in range(num_episodes):
        # Reset the environment and start a new episode
        s = env.reset() # utilizing the basic reset method of the environment to start a new episode
        done = False

        # Run the episode until termination (time steps)
        while not done:
            # 1. Observe transition
            a = policy(s) # Select action according to the provided policy
            s_next, r, done = env.step(a)

            # 2. Update value function using TD(0) update rule
            bootstrap = 0.0 if done else V[s_next] # no bootstrap if terminal state, otherwise use value of next state
            target = r + gamma * bootstrap

            # 3. Update the value function for the current state
            V[s] += alpha * (target - V[s]) # update rule for TD(0)
            s = s_next

    return V