"""
environments/base_env.py
-------------------------
Minimal MDP interaction interface shared by all environments.

Each environment defines a hidden Markov Decision Process (MDP)
with transition dynamics p(s'|s,a) and rewards r(s,a,s'), but these
are never exposed to the agent.

The agent only interacts through:

    reset() → start episode and return initial state
    step(a) → sample (s', r, done) from the environment

This enforces a strict sampling-only setting, where all learning
(model-free or model-based) must be done from experience rather
than direct access to the environment dynamics.
"""

# Abstract base class for all environments in this project
from abc import ABC, abstractmethod

from typing import Hashable, Tuple


class Environment(ABC):
    """Common interface every environment in this project implements."""

    #: tuple of all legal actions, fixed per-environment (e.g. (0, 1))
    actions: tuple

    # @abstractmethod: Every subclass MUST implement this
    @abstractmethod
    def reset(self) -> Hashable:
        """Start a new episode and return the initial state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, action) -> Tuple[Hashable, float, bool]:
        """Apply `action` from the current state.

        Returns (next_state, reward, done).
        """
        raise NotImplementedError
