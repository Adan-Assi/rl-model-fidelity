"""
Value Iteration (VI) - General Form
---------------------
Solves the Bellman optimality equation iteratively.

  Start with any initial guess: V_0(s) = 0
  Then iterate:
      V_{n+1}(s) = max_a ∑_{s'} p(s'|s,a) [ r(s,a,s') + γ V_n(s') ]

  Note (DP setting): Value Iteration assumes a KNOWN MDP, i.e., we have
  access to the transition probabilities p(s'|s,a) and reward function
  r(s,a,s'). Therefore, it is a Dynamic Programming (DP) and model-based
  method.

  Same pattern as policy evaluation (agents/td0.py), but instead of
  following a fixed policy, we act GREEDILY w.r.t. the current estimate.

  V_n(s) = "best value assuming I only look n steps ahead"

  First iteration: ignores the future completely, only immediate reward
  matters. Later iterations: future value gets folded in as V_n propagates
  backward through the model.

  Note: This is the fully general form where reward may depend on (s,a,s').
  If reward depends only on (s,a), it can be pulled outside the sum,
  yielding the simpler, commonly seen version.
"""

from typing import Dict, List, Tuple

"""
The KNOWN MDP assumption shows up in the `model` argument:

`model` is a dictionary mapping each `(state, action)` tuple to a list of
possible outcomes:

    (state, action) -> [(prob, next_state, reward, done), ...]

The `(state, action)` tuples are immutable dictionary keys that identify each
state-action pair. The model is fixed throughout the algorithm, with the
transition probabilities and rewards already given.

value_iteration() receives this complete model up front and NEVER interacts
with an Environment, making it model-based. In contrast, model-free
algorithms such as `algorithms/td0.py` and `algorithms/sarsa.py` learn from
individual transitions obtained through `env.step()`.
"""
def value_iteration(
    states,
    actions,
    model: Dict[Tuple, List[Tuple[float, object, float, bool]]],
    gamma: float,
    tolerance: float = 1e-4,
    max_iterations: int = 10_000,
):
    # V_0(s) = 0: Initial guess of the value of every state is 0 (arbitrary).
    V = {s: 0.0 for s in states}

    # Iterate until convergence or until we reach the maximum number of iterations.
    for it in range(1, max_iterations + 1):
        V_new = {}
        max_delta = 0.0

        # Update the value of each state based on the Bellman optimality equation.
        for s in states:

            """
            The Bellman optimality update is equivalent to V_{n+1}​(s) = max_a ​Q_n​(s,a).

            To perform  max_a Q_n​(s,a), we first need to calculate ​Q_n​(s,a) for 
              EACH available ACTION, then compare them and take the maximum.  
            """
            q_values = [_q(model, s, a, V, gamma) for a in actions]
            best = max(q_values) if q_values else 0.0 # if-else to handle empty case (e.g., no available actions)

            V_new[s] = best # stores the newly calculated value for state s
            max_delta = max(max_delta, abs(best - V[s])) # keeps track of the maximum change in value across all states

        V = V_new # updates the value function to the newly calculated values for the next iteration.

        """
        Convergence check: if the maximum change in value across all states is
        below the tolerance, we treat V as having converged (V ≈ V*).

        In this case, the greedy policy w.r.t. V is the optimal policy.
        """
        if max_delta < tolerance:
            return V, _greedy_policy(states, actions, model, V, gamma), it

    # If we reach the maximum number of iterations without converging, we return:
    # - the last computed value function V (which is an approximation of V*),
    # - the greedy policy based on that value function,
    # - and the maximum number of iterations reached.
    return V, _greedy_policy(states, actions, model, V, gamma), max_iterations


# Helper function to compute the Q-value for a given state-action pair.
def _q(model, s, a, V, gamma):
    transitions = model.get((s, a), [])
    return sum(p * (r + (0.0 if done else gamma * V[s_next]))
               for p, s_next, r, done in transitions)


# Helper function to compute the greedy policy based on the current value function V.
# Greedy (w.r.t V): pick argmax_a Q(s,a), i.e., the action that currently looks best according to V.
def _greedy_policy(states, actions, model, V, gamma):
    policy = {}
    for s in states:
        q_values = {a: _q(model, s, a, V, gamma) for a in actions}

        """
        `q_values` = {action: Q-value}.
        `max(q_values, key=q_values.get)` looks at the Q-values to compare the
        actions, then returns the action (key) with the highest Q-value.
        """
        policy[s] = max(q_values, key=q_values.get)
    return policy