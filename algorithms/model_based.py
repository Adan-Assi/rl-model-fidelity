"""
Model-based RL agent

Learns a model of the environment from sampled transitions, then uses
Value Iteration to plan with the learned model.

Unlike model-free methods such as Q-learning and SARSA, this agent does
not learn Q-values directly from transitions. Instead, it estimates:

    p(s'|s,a)    transition probabilities
    r(s,a,s')    rewards

and stores them as an explicit model in the same format expected by
`value_iteration()`:

    (state, action) -> [(prob, next_state, reward, done), ...]

The agent periodically re-estimates this model from accumulated experience
and re-runs Value Iteration to update its value function and greedy policy.

This is still model-based even though the MDP is NOT given up front:
the model is learned from experience rather than provided directly.

`run_model_based()` separates the learning/planning logic from the
experimental schedule: the caller decides when to re-plan and when to
stop, allowing the same agent to be reused across different environments.
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

    def act(self, s):
        """Choose the action currently prescribed by the learned policy."""

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
    seed: int = 0,
):
    """
    Drives `env` with a ModelBasedAgent, replanning whenever `replan_trigger`
    (given the completed-episode count) says to. Stops once
    `no_learning_threshold` consecutive replans all converge in a single
    value-iteration sweep -- the same "model has stopped changing"
    criterion the original CartPole task used, expressed generically here.
    """
    agent = ModelBasedAgent(states, actions, gamma, tolerance, seed)
    episode_lengths = []
    consecutive_single_sweep = 0
    episode = 0

    while episode < max_episodes and consecutive_single_sweep < no_learning_threshold:
        s = env.reset()
        steps, done = 0, False
        while not done and steps < max_steps_per_episode:
            a = agent.act(s)
            s_next, r, done = env.step(a)
            agent.observe(s, a, s_next, r, done)
            s = s_next
            steps += 1

        episode += 1
        episode_lengths.append(steps)

        if replan_trigger(episode):
            iters = agent.replan()
            consecutive_single_sweep = consecutive_single_sweep + 1 if iters == 1 else 0

    return agent, episode_lengths
