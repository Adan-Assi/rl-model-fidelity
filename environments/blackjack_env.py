"""
environments/blackjack_env.py
------------------------------
Object-oriented implementation of a simplified Blackjack ('21') MDP.

State = gambler's running sum  
Actions = {HIT, STAND}  
The house rules (i.e., how the environment resolves the game after an action) are fixed:
    hit below a threshold, otherwise stand.
Ace is fixed at 11 (no "usable ace"), so the state is a single integer.

------------------------------------------------------------
Design choices

* Unified interface:
  Implemented as an `Environment` with reset()/step(), enabling the same
  agents to run across different environments.

* Configurable deck model:
  - Infinite deck (with replacement, default)
  - Finite 52-card shoe (without replacement)

  This allows controlled variation in environment structure and estimation difficulty.

* Encapsulated episode logic:
  All game mechanics (drawing, bust detection, house play, reward)
  are handled inside `step`.

------------------------------------------------------------
Notes

This is a deliberately simplified version of Blackjack:
no usable-ace logic and no dealer-visible card.

This keeps the state space small and the MDP fully tabular,
at the cost of realism.
"""


import random
from enum import IntEnum
from typing import Tuple

# imports the shared Environment interface (reset/step contract) from the current package
from .base_env import Environment


class Action(IntEnum):
    HIT = 0
    STAND = 1


# --- Infinite-deck model (with replacement) ---
# 13 ranks, each equally likely:
# 2–9 face value, {10, J, Q, K} all mapped to 10 (repeated entries preserve correct probability mass), Ace fixed at 11.
_RANK_VALUES = list(range(2, 11)) + [10, 10, 10] + [11]

# --- Finite-deck model (without replacement) ---
# Concrete 52-card shoe: 13 ranks × 4 suits.
# Cards are drawn without replacement; deck is reshuffled when exhausted.

# --- Finite-deck model (without replacement) ---
# Concrete 52-card shoe: 13 ranks × 4 suits (hearts, diamonds, clubs, spades).
# Cards are drawn without replacement (once a card is drawn its removed),and the deck is reshuffled when exhausted.
# This models a real shuffled deck where past draws slightly affect future probabilities (unlike the infinite-deck model).
_FULL_SHOE = [v for v in _RANK_VALUES for _suit in range(4)]


class BlackjackEnv(Environment):
    actions = (Action.HIT, Action.STAND)

    # all possible non-terminal gambler sums (4–21) where the agent still chooses action
    states = tuple(range(4, 22)) 


    """
    Initializes the environment by separating two categories of fields:

    User-configurable inputs (external “game settings”):
    - Parameters that define how the environment behaves from the outside
    - Example: house_hit_threshold (house rule difficulty), with_replacement (deck type), seed (randomness control)
    - These are the only values the user can modify when creating the environment instance

    Internal state (hidden “game memory”):
    - Variables used during execution to track the current episode and randomness
    - Example: rng (random number generator), _shoe (deck when using finite model), _gambler_sum (current hand total)
    - These are managed internally by the environment and are not directly configurable

    Together, these define the setup phase of the environment before any reset()/step() interaction begins.
    """
    def __init__(self, house_hit_threshold: int = 15, with_replacement: bool = True,
                 seed: Optional[int] = None):
        self.house_hit_threshold = house_hit_threshold
        self.with_replacement = with_replacement
        self.rng = random.Random(seed)
        self._shoe = []  # only populated when with_replacement (infinite deck) is False
        self._gambler_sum = None # current sum of the gambler's hand, updated during episode execution


    # -- card dealing ----------------------------------------------------
    def _reshuffle(self): # (like reshuffling a physical deck before a new round)
        # Start with a full fresh deck again, without destroying the original
        self._shoe = _FULL_SHOE.copy()

        # Shuffle the fresh deck in-place (random.shuffle from Python stdlib)
        self.rng.shuffle(self._shoe)

    def _draw(self) -> int:
        if self.with_replacement:
            return self.rng.choice(_RANK_VALUES) # draw a card randomly from the infinite deck (with replacement)
        if not self._shoe:
            self._reshuffle() # reshuffle the shoe if it's empty (all cards drawn)
        return self._shoe.pop() # take the top card and remove it from the deck

    # Simulates the house (dealer) turn using a fixed policy:
    # start with 2 cards, then repeatedly hit until reaching the threshold
    # (this defines how the environment resolves the game after the agent acts)
    def _play_house(self) -> int:
        total = self._draw() + self._draw() # initial house hand total (two cards)

        # house policy: keep drawing while weak
        while total <= self.house_hit_threshold:
            total += self._draw() # hit = draw another card and add to total

        return total # final house score after policy terminates


    # -- Environment interface -------------------------------------------
    def reset(self) -> int: # Start a new Blackjack game and give the initial state
        while True: # keep drawing until a valid starting hand is dealt (not a bust)
            total = self._draw() + self._draw() # initial gambler hand total (two cards)
            if total <= 21: # valid starting state (not a bust)
                self._gambler_sum = total
                return total
            # Avoid invalid initial states (e.g., immediate bust like 11+11=22),
            # so the game always starts in a state where the agent still has a choice

    def step(self, action: Action) -> Tuple[int, float, bool]:
        if action == Action.HIT:
            self._gambler_sum += self._draw()
            done = self._gambler_sum > 21
            return self._gambler_sum, 0.0, done # reward is 0.0 for non-terminal states (HIT) and terminal busts (gambler_sum > 21)

        # STAND: resolve against the house and end the episode.
        house_total = self._play_house()
        won = house_total > 21 or self._gambler_sum > house_total # gambler wins if house busts or gambler has higher total
        reward = 1.0 if won else 0.0  # draw counts as non-win
        return self._gambler_sum, reward, True # (next_state, reward, done)
