"""
Model-Based RL (learned model + planning via Value Iteration)
---------------------
Learns an explicit model of the environment from sampled transitions,
then uses Value Iteration to compute a value function and greedy policy.

Unlike model-free methods such as Q-learning and SARSA, this approach
does NOT update value estimates directly from individual transitions.
Instead, it separates the problem into two phases:

    1. Model Learning (from experience)
    2. Planning (using the learned model)

From observed transitions (s_t, a_t, r_t, s_{t+1}), the agent estimates:

    p(s'|s,a)    transition probabilities
    r(s,a,s')    rewards

These estimates are stored in an explicit model of the form:

    (state, action) -> [(prob, next_state, reward, done), ...]

which matches the input expected by `value_iteration()`.

Planning step:
At chosen intervals, the agent constructs its current model estimate
and solves the Bellman optimality equation via Value Iteration:

    V_{n+1}(s) = max_a Σ_{s'} p̂(s'|s,a) [ r̂(s,a,s') + γ V_n(s') ]

The resulting greedy policy is then used for future interaction.

Key idea: this agent learns a model from data, then plans as if that model were correct.
That creates a critical dependency:

    performance ≈ quality of the learned model

If the model is accurate → behaves like optimal planning
If the model is biased/noisy → planning propagates those errors

Which is exactly the axis studied in this project:
    model fidelity and its effect on performance.

Exploration:

This agent acts epsilon-greedily during data collection (see act()),
using the same decaying-epsilon idea as algorithms/q_learning.py. This
matters more here than it might look: acting purely greedily w.r.t. an
early, sparse policy can silently starve half the model of data (the
other action for a state may never get sampled again once the greedy
policy commits), causing the "model has converged" signal to fire against
an incomplete model rather than an accurate one. This is a real failure
mode observed empirically on Blackjack's short episodes (the agent
"converged" after 20 hands with a degenerate always-HIT, 0%-win-rate
policy), not just a theoretical concern -- CartPole's long episodes
(up to 2000 steps each) happened to mask it, since sheer step-volume
compensated for the lack of explicit exploration.

Separation of concerns:

- `ModelBasedAgent` handles:
    learning the model + planning

- `run_model_based()` handles:
    interaction loop, replanning schedule, epsilon schedule, and
    stopping criterion

This allows flexible experimentation with different replanning and
exploration strategies.
"""

import random
from collections import defaultdict
from typing import Callable

from environments.base_env import Environment
from algorithms.value_iteration import value_iteration


class ModelBasedAgent:
    def __init__(self, states, actions, gamma: float, tolerance: float = 1e-2, seed: int = 0):
        self.states = states
        self.actions = actions
        self.gamma = gamma
        self.tolerance = tolerance
        self.rng = random.Random(seed)

        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.reward_sums = defaultdict(lambda: [0.0, 0])  # (s,a) -> [sum, count]
        self.done_flags = defaultdict(lambda: defaultdict(bool))

        self.V = {s: self.rng.uniform(0, 0.1) for s in states}
        self.policy = {s: self.rng.choice(actions) for s in states}
        self.last_convergence_iters = None

    def act(self, s, epsilon: float = 0.0):
        """
        Choose an action epsilon-greedily: with probability `epsilon`,
        explore with a uniformly random action; otherwise follow the
        current learned policy. See the module docstring's Exploration
        section for why this can't safely default to pure greedy.
        """
        if self.rng.random() < epsilon:
            return self.rng.choice(self.actions)
        return self.policy.get(s, self.rng.choice(self.actions))

    def observe(self, s, a, s_next, r, done):
        """Update the learned model with a single observed transition (s,a,s',r,done)."""

        self.transition_counts[(s, a)][s_next] += 1
        self.reward_sums[(s, a)][0] += r
        self.reward_sums[(s, a)][1] += 1
        self.done_flags[(s, a)][s_next] = done

    def replan(self):
        """
        Build the current estimated MDP from accumulated experience and
        re-solve it with Value Iteration.
        """

        model = {}
        for (s, a), next_counts in self.transition_counts.items():
            total = sum(next_counts.values())
            r_sum, r_count = self.reward_sums[(s, a)]
            avg_r = r_sum / r_count if r_count else 0.0
            model[(s, a)] = [
                (count / total, s_next, avg_r, self.done_flags[(s, a)][s_next])
                for s_next, count in next_counts.items()
            ]

        self.V, self.policy, iters = value_iteration(
            self.states, self.actions, model, self.gamma, self.tolerance
        )
        self.last_convergence_iters = iters
        return iters


def run_model_based(
    env: Environment,
    states,
    actions,
    gamma: float,
    replan_trigger: Callable[[int], bool],
    max_episodes: int,
    no_learning_threshold: int,
    tolerance: float = 1e-2,
    max_steps_per_episode: int = 2000,
    epsilon_fn: Callable[[int], float] = lambda episode: 1.0 / (episode + 1),
    min_episodes_before_convergence_check: int = 0,
    seed: int = 0,
):
    """
    Drives `env` with a ModelBasedAgent, replanning whenever `replan_trigger`
    (given the completed-episode count) says to. Stops once
    `no_learning_threshold` consecutive replans all converge in a single
    value-iteration sweep; the same "model has stopped changing"
    criterion the original CartPole task used, expressed generically here.

    `epsilon_fn` controls exploration during data collection (see the
    module docstring); the default harmonic decay (1/(episode+1)) matches
    algorithms/q_learning.py's schedule. `min_episodes_before_convergence_check`
    guards against declaring convergence before the model has seen enough
    data, important for short-episode environments like Blackjack, where
    a handful of episodes can trivially "converge" against an almost
    entirely unvisited model (see the module docstring's Exploration note).
    """
    agent = ModelBasedAgent(states, actions, gamma, tolerance, seed)
    episode_lengths = []
    consecutive_single_sweep = 0
    episode = 0

    while episode < max_episodes and consecutive_single_sweep < no_learning_threshold:
        s = env.reset()
        steps, done = 0, False
        epsilon = epsilon_fn(episode)
        while not done and steps < max_steps_per_episode:
            a = agent.act(s, epsilon)
            s_next, r, done = env.step(a)
            agent.observe(s, a, s_next, r, done)
            s = s_next
            steps += 1

        episode += 1
        episode_lengths.append(steps)

        if replan_trigger(episode):
            iters = agent.replan()
            if episode < min_episodes_before_convergence_check:
                consecutive_single_sweep = 0
            else:
                consecutive_single_sweep = consecutive_single_sweep + 1 if iters == 1 else 0

    return agent, episode_lengths