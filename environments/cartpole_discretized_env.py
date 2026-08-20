"""
environments/cartpole_discretized_env.py
----------------------------------------

Discretized version of the CartPole (inverted pendulum) control problem.

State = discretized bins of (x, x_dot, theta, theta_dot)  
where:
    x = cart position ("Where is the cart along the track?")
    x_dot = cart velocity ("How fast (and in which direction) is the cart moving?")
    theta = pole angle ("How far is the pole from vertical?")
    theta_dot = pole angular velocity ("How fast (and in which direction) is the pole rotating?")

Actions = {push left, push right}  

The underlying system follows standard continuous cart-pole physics,
but is mapped into a finite state space via discretization.

------------------------------------------------------------
User-configurable inputs:
- discretization granularity (number of bins per state dimension)
- physical parameters of the system (gravity, masses, etc.)
- episode step limit (max_steps)
- random seed

Internal state (not configurable):
- continuous state tuple (x, x_dot, theta, theta_dot)
- discretized state index used by the agent
- current step count within the episode

------------------------------------------------------------
Design choices

* Unified interface:
  Implements reset()/step() → (state, reward, done),
  allowing direct comparison with other environments.

* Discretized state space:
  Continuous variables are grouped into bins, producing a finite,
  tabular state space required by the implemented RL algorithms.

  The number of bins per dimension is configurable:
      (x_bins, x_dot_bins, theta_bins, theta_dot_bins)

* Fidelity control (core of this environment):
  Discretization introduces approximation error (state aliasing).

  - Finer bins → more accurate model (higher fidelity)
  - Coarser bins → more noise (lower fidelity)

  This makes CartPole the low-fidelity regime in the project.

* Episode length cap (max_steps):
  A successfully-balancing policy can keep the pole upright indefinitely,
  so `step()` alone cannot guarantee episode termination. `max_steps`
  truncates the episode after a fixed number of steps if failure hasn't
  already occurred. This replicates what Gymnasium's `TimeLimit` wrapper
  does automatically for registered environments (e.g. FrozenLake-v1);
  since this environment is hand-built with no such wrapper, truncation
  is implemented explicitly here via `_step_count`. Truncation and
  failure are both signaled as `done=True`; callers that need to
  distinguish "timed out" from "failed" should check
  `step_count >= max_steps` after `step()` returns `done=True`.

* Reward convention (matches Gymnasium CartPole-v1):
  +1.0 for every step taken, including the terminal step; reward does
  not distinguish failure from truncation, it simply stops accruing once
  the episode ends. This matches Gymnasium's own convention, consistent
  with reusing Gymnasium's failure thresholds (|x| > 2.4, |theta| > 12°)
  elsewhere in this file. Standard benchmarks (e.g. "solved" = average
  reward >= 195 over 100 episodes) are therefore directly comparable,
  modulo the effect of discretization on achievable performance.

* Initial state distribution (matches Gymnasium CartPole-v1):
  All four state dimensions (x, x_dot, theta, theta_dot) are drawn
  independently and uniformly from [-0.05, 0.05] at reset(), matching
  Gymnasium's own initialization rather than randomizing position only.

------------------------------------------------------------
Notes

Unlike FrozenLake (exact model) and Blackjack (small estimable model),
CartPole's true dynamics are continuous.

Discretization is therefore an approximation, not an exact representation.

This makes it ideal for studying how model-based methods degrade when
the learned model cannot perfectly capture the true environment.

Default physical parameters (mass_pole=0.1, half_length=0.5) match
Gymnasium CartPole-v1's defaults (masspole=0.1, length=0.5), consistent
with this project's use of Gymnasium's own failure thresholds and reward
convention elsewhere in this file.
"""

import math
import random
from typing import Optional, Tuple

from .base_env import Environment


class CartPoleDiscretizedEnv(Environment):
    actions = (0, 1)  # push left, push right

    """
    Constructor for CartPoleDiscretizedEnv.

    Parameters:
    - x_bins, x_dot_bins, theta_bins, theta_dot_bins:
    number of bins for discretizing each state dimension

    - gravity, force_mag, tau, mass_cart, mass_pole, half_length:
    physical parameters of the cart-pole system

    - max_steps:
    number of steps after which an episode is truncated (done=True)
    if the pole hasn't already fallen. Without this, a policy that
    balances successfully would never produce a terminal step.

    - seed:
    random seed for reproducibility
    """
    def __init__(
        self,
        x_bins: int = 3, 
        x_dot_bins: int = 3,
        theta_bins: int = 6,
        theta_dot_bins: int = 3,
        gravity: float = 9.8,
        force_mag: float = 10.0,
        tau: float = 0.02,
        mass_cart: float = 1.0,
        mass_pole: float = 0.1,
        half_length: float = 0.5, # usully called length, but this is half the pole length
        max_steps: int = 2000,
        seed: Optional[int] = None,
    ):
        self.x_bins, self.x_dot_bins = x_bins, x_dot_bins
        self.theta_bins, self.theta_dot_bins = theta_bins, theta_dot_bins
        self.gravity, self.force_mag, self.tau = gravity, force_mag, tau
        self.mass_cart, self.mass_pole = mass_cart, mass_pole
        self.mass = mass_cart + mass_pole # total mass of the system
        self.half_length = half_length
        self.pole_mass_length = mass_pole * half_length # used in the equations of motion for the cart-pole system
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self.num_states = x_bins * x_dot_bins * theta_bins * theta_dot_bins + 1 # +1 to account for the failure state (e.g., when the pole falls over or the cart goes out of bounds)
        self.failure_state = self.num_states - 1 # the last state index
        self.states = tuple(range(self.num_states))

        self._state_tuple = (0.0, 0.0, 0.0, 0.0) # continuous state representation (x, x_dot, theta, theta_dot)
        self._step_count = 0 # steps taken so far in the current episode


    # -- physics: standard Euler-integrated cart-pole dynamics -----------
    # Simulates one time-step of the true continuous dynamics (physics),
    # given the current state and chosen action.
    def _simulate(self, action: int, state_tuple: Tuple[float, float, float, float]):
        x, x_dot, theta, theta_dot = state_tuple
        costheta, sintheta = math.cos(theta), math.sin(theta)
        force = self.force_mag if action > 0 else -self.force_mag

        temp = (force + self.pole_mass_length * theta_dot ** 2 * sintheta) / self.mass
        theta_acc = (
            (self.gravity * sintheta - temp * costheta)
            / (self.half_length * (4 / 3 - self.mass_pole * costheta ** 2 / self.mass))
        )
        x_acc = temp - self.pole_mass_length * theta_acc * costheta / self.mass

        return (
            x + self.tau * x_dot,
            x_dot + self.tau * x_acc,
            theta + self.tau * theta_dot,
            theta_dot + self.tau * theta_acc,
        )

    # -- discretization: uniform bins across a physically-sensible range
    #    for each dimension, count controlled by the constructor ----------
    @staticmethod
    def _uniform_bin(value, lo, hi, n_bins):
        """
        Maps a continuous value into a discrete, ordered bin index.

        The interval [lo, hi] is divided into n_bins equal-sized ranges:

            bin 0      → lowest values (≤ lo)
            bin 1..k   → increasing value ranges
            bin n-1    → highest values (≥ hi)

        This produces an ordered discretization where:
            lower values → lower indices
            higher values → higher indices

        Out-of-range values are clipped to boundary bins.
        """
        if value <= lo:
            return 0
        if value >= hi:
            return n_bins - 1
        width = (hi - lo) / n_bins
        return min(n_bins - 1, int((value - lo) / width)) # return the bin index, ensuring it doesn't exceed n_bins - 1


    """
    Failure conditions (Gymnasium CartPole standard):

    This environment follows the official CartPole termination rules used in
    Gymnasium / OpenAI Gym benchmarks:

    - |x| > 2.4          → cart moves outside track bounds
    - |theta| > 12°      → pole exceeds stability limit

    These thresholds are standard benchmark definitions, not arbitrary
    project-specific parameters.

    Termination is triggered during discretization, where continuous states
    are mapped into a finite MDP representation. This is separate from
    truncation, which is driven by max_steps in step() and is NOT a
    physical failure.
    """
    def _discretize(self, state_tuple) -> int:
        x, x_dot, theta, theta_dot = state_tuple
        one_deg = math.pi / 180 # convert degrees to radians
        fail_theta = 12 * one_deg

        if x < -2.4 or x > 2.4 or theta < -fail_theta or theta > fail_theta:
            return self.failure_state

        x_bin = self._uniform_bin(x, -2.4, 2.4, self.x_bins) # split [−2.4,2.4] into x_bins equal-width bins
        x_dot_bin = self._uniform_bin(x_dot, -2.0, 2.0, self.x_dot_bins) # split [−2.0,2.0] into x_dot_bins equal-width bins
        theta_dot_bin = self._uniform_bin(theta_dot, -50 * one_deg, 50 * one_deg, self.theta_dot_bins) # split [−50°,50°] into theta_dot_bins equal-width bins
        theta_bin = self._uniform_bin(theta, -fail_theta, fail_theta, self.theta_bins) # split [−12°,12°] into theta_bins equal-width bins

        # Flatten 4D discretized state (x, x_dot, theta, theta_dot) into a single
        # integer index for tabular RL.

        # Row-major encoding of a 4D grid:
        # x varies fastest, then x_dot, theta, and theta_dot define progressively larger blocks.
        #
        # Produces a unique index in:
        #   [0, x_bins * x_dot_bins * theta_bins * theta_dot_bins - 1]
        #
        # The last index (num_states - 1) is reserved for the failure state,
        # triggered when physical termination conditions are met.
        #
        # Enables standard tabular RL methods (Q-learning, SARSA, Value Iteration).
        return (
            x_bin
            + x_dot_bin * self.x_bins
            + theta_bin * self.x_bins * self.x_dot_bins
            + theta_dot_bin * self.x_bins * self.x_dot_bins * self.theta_bins
        )

    # -- Environment interface -------------------------------------------
    def reset(self) -> int:
        # Matches Gymnasium CartPole-v1: all four state dimensions drawn
        # independently and uniformly from [-0.05, 0.05].
        low, high = -0.05, 0.05
        x0 = low + self.rng.random() * (high - low) # x0 ∼ Uniform(-0.05, 0.05)
        x_dot0 = low + self.rng.random() * (high - low)
        theta0 = low + self.rng.random() * (high - low)
        theta_dot0 = low + self.rng.random() * (high - low)

        self._state_tuple = (x0, x_dot0, theta0, theta_dot0)
        self._step_count = 0
        return self._discretize(self._state_tuple)

    def step(self, action: int) -> Tuple[int, float, bool]:
        self._state_tuple = self._simulate(action, self._state_tuple)
        self._step_count += 1

        s_next = self._discretize(self._state_tuple) # discretize the new continuous state into a finite MDP state
        failed = s_next == self.failure_state # check if the new state is a failure state
        truncated = (not failed) and (self._step_count >= self.max_steps) # check if the episode has reached the maximum step limit without failure
        done = failed or truncated # episode ends if either failure or truncation occurs)

        reward = 1.0  # Gymnasium CartPole-v1 convention: +1 for every step
                       # taken, including the terminal step (no failure
                       # penalty; reward simply stops accruing once done)
        return s_next, reward, done