"""
environments/frozen_lake_env.py
------------------------------

Wrapper around Gymnasium's FrozenLake-v1 environment, adapted to this
project's unified `Environment` interface.

State = discrete grid position  
Actions = {LEFT, DOWN, RIGHT, UP}  

The environment defines a stochastic MDP where transitions may "slip"
(i.e., not follow the intended direction), depending on the
`is_slippery` setting.

------------------------------------------------------------
Design choices

* Unified interface:
  Exposes reset()/step() returning (state, reward, done),
  so the same agents can run across different environments.

* Exact model access (unique to this environment):
  Unlike other environments in this project, FrozenLake exposes its
  true transition dynamics via `transition_model()`.

  This is intentional: FrozenLake represents the high-fidelity regime,
  where the model is known exactly.

* Controlled stochasticity:
  The `is_slippery` flag controls whether transitions are deterministic
  or stochastic, allowing variation in environment difficulty.

------------------------------------------------------------
Notes

FrozenLake is the only environment in this project where the true MDP
is directly accessible.

This makes it the reference point for comparing:

- Exact planning (Value Iteration with true model)
- Model-free learning (Q-learning, SARSA)

against settings where the model must be learned or approximated.
"""

import gymnasium as gym

from .base_env import Environment


class FrozenLakeEnv(Environment):
    # Class-level constant for the action space, following Gymnasium's convention.
    actions = (0, 1, 2, 3)

    # instance-level attributes:
    def __init__(self, is_slippery: bool = True, seed: Optional[int] = None):
        self._env = gym.make("FrozenLake-v1", is_slippery=is_slippery) # its like loading the game from Gym
        self.states = tuple(range(self._env.observation_space.n)) # states are the discrete grid positions
        self._seed = seed

    def reset(self) -> int:
        s, _ = self._env.reset(seed=self._seed)
        self._seed = None  # ensure seeding happens only once (deterministic start)
        return s # return the initial state after reset

    def step(self, action: int):
        s_next, r, terminated, truncated, _ = self._env.step(action)
        return s_next, r, (terminated or truncated)

    def transition_model(self):
        """
        Return the EXACT transition model of the environment:

            (state, action) -> [(prob, next_state, reward, done), ...]

        This is directly extracted from Gymnasium's internal representation.

        Unlike other environments in this project, this is the TRUE model,
        not an estimate. This enables exact planning (e.g., Value Iteration)
        and serves as the high-fidelity baseline in the study.
        """

        # raw = Gymnasium's internal transition probabilities
        # structure: dict[int, dict[int, list[tuple]]]
        # which is a dict over states,
        # each state maps to a dict over actions,
        # each action maps to a list of possible outcomes.
        raw = self._env.unwrapped.P

        model = {}

        for s, action_dict in raw.items(): # int, dict[int, list[tuple]]
            for a, transitions in action_dict.items(): # int, list[tuple]

                # Flattening (s,a) structure for algorithm-friendly and consistent MDP interface
                model[(s, a)] = [
                    (prob, s_next, float(r), done)
                    for prob, s_next, r, done in transitions
                ]

        return model