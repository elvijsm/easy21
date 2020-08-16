#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base utilities for the game of easy21.

This module stores various utility classes and functions for the module easy21.

"""
import numpy as np
import core.params as params
from collections import namedtuple
from typing import Union

### The actions possible in the game:
### draw a card (hit) or stay with the current sum (stick).
HIT = 0
STICK = 1
### The size of the state space: 10 dealer scores, 21 player scores, 2 actions.
### Plus one for the first two to avoid subtracting 1 everywhere
SDIM = (11,22,2)

### Named tuple storing the dealer sum, the player sum, and the current action.
StateAction = namedtuple('StateAction', ['ds', 'ps', 'a'])
StateAction.__doc__ = """
A tuple storing the current state and action.

ds : int -- The current sum of the dealer.
ps : int -- The current sum of the player.
a : int -- Action chosen by the player.
"""

### Classes
class State:
    """
    A class storing a state of a single round of Easy21.

    A state consists of a tuple of (dealer sum, player sum). The class
    provides the functionality for transitioning between states via the
    hit_dealer() and hit_player() methods.

    Attributes
    ----------
    dealer_sum : int
        Current dealer sum.
    player_sum : int
        Current player sum.

    """

    def __init__(self, dealer_sum: int, player_sum: int = 0):
        """Initialize the dealer and player sums."""
        self.dealer_sum = dealer_sum
        self.player_sum = player_sum
        self.terminal = False

    def get(self) -> (int, int):
        """Return the current sum of the dealer and the player."""
        return self.dealer_sum, self.player_sum

    def get_dealer_sum(self) -> int:
        """Return the current sum of the dealer."""
        return self.dealer_sum

    def get_player_sum(self) -> int:
        """Return the current sum of the player."""
        return self.player_sum

    def hit_dealer(self):
        """Draw a card for the dealer and mark state as terminal if the dealer busts."""
        if not self.is_terminal():
            self.dealer_sum += hit_rigged(self.dealer_sum) if params.rigged else hit()
            if self.dealer_is_bust():
                self.terminal = True
    def hit_player(self):
        """Draw a card for the player and mark state as terminal if the player busts."""
        if not self.is_terminal():
            self.player_sum += hit()
            if self.player_is_bust():
                self.terminal = True

    def player_is_bust(self) -> bool:
        """Return True if the player's sum is outside [1,21]."""
        return not (1 <= self.player_sum <= 21)

    def dealer_is_bust(self) -> bool:
        """Returns True if the dealer's sum is outside [1,21]."""
        return not (1 <= self.dealer_sum <= 21)

    def is_terminal(self) -> bool:
        """Return True if the state is terminal."""
        return self.terminal

    def set_terminal(self):
        """Mark the state as terminal."""
        self.terminal = True


class Episode:
    """
    A class storing a single round of Easy21.

    Attributes
    ----------
    history : list(StateAction)
        The sequence of state-action pairs in the round.
    outcome : int
        The outcome of the round (i.e. reward).
    finished : bool
        State of the round, True as soon as a terminal state is reached.

    """

    def __init__(self, state: State = None, action: int = None):
        """Initialize episode history."""
        if state is None or action is None:
            self.history = []
        else:
            self.history = [StateAction(state.get_dealer_sum(), state.get_player_sum(), action)]
        self.outcome = 0
        self.finished = False

    def advance(self, state: State, action: int):
        """Take `action` starting from `state` and append to history."""
        if not self.finished:
            self.history.append(StateAction(*state.get(), action))
            s, r = step(state, action)
            if s.is_terminal():
                self.finished = True
                self.outcome = r

    def discard_duplicate_states(self):
        """Keep only one instance of each visited state."""
        self.history = set(self.history)

    def get_history(self) -> list:
        """Return the history of the round."""
        return self.history

    def get_outcome(self) -> int:
        """Return the outcome of the round."""
        return self.outcome

    def get_final_state(self) -> (list, int):
        """Return the history and the outcome of the round."""
        if self.finished:
            return self.history, self.outcome

    def is_finished(self) -> bool:
        """Return True if the round is finished and False otherwise."""
        return self.finished


class ModelFreeControl:
    """
    A base class for applying model-free reinforcement learning to the game of Easy21.

    Attributes
    ----------
    dealer_policy : str
        The policy followed by the dealer, one of "easy21", "smart", "random",
        "mia". Default is "easy21".
    dealer_stick_sum : int
        Dealer's sum used by the dealer to decide on the next action (only for
        "easy21" and "smart" policies). Default is 17.
    prob_black : float
        Probability of drawing a black (positive) card. Default is 2/3.
    N0 : float
        Normalization constant of epsilon in eps-greedy. Default is 100.0.
    dt0 : float
        Smallest allowed stepsize. Default is 0.0.
    rigged : bool
        Use a rigged version of drawing cards that make it impossible for the
        dealer to go bust. Default is False.
    Q : np.ndarray or int
        Initial state-action value function. Default is 0.
    N : np.ndarray or int
        Initial array storing the number of times each state has been visited.
        Default is 0.

    """

    def __init__(self, dealer_policy: str, dealer_stick_sum: int,
                 prob_black: float, N0: float, dt0: float, rigged: bool,
                 Q_init: Union[np.ndarray, int],
                 N_init: Union[np.ndarray, int]):
        """Initialize variables and parameters common to all algorithms."""
        valid_policies = ["easy21", "smart", "random", "mia"]
        assert dealer_policy in valid_policies, \
            "Dealer policy '{0}' not in {1}".format(dealer_policy, valid_policies)
        params.dealer_policy = dealer_policy
        params.dealer_stick_sum = dealer_stick_sum
        params.prob_black = prob_black
        params.N0 = N0
        params.dt0 = dt0
        params.rigged = rigged
        self.Q = Q_init * np.ones(SDIM) if type(Q_init) is int else Q_init.copy()
        self.N = N_init * np.ones(SDIM) if type(N_init) is int else N_init.copy()


    def update(self, episode: Episode) -> None:
        """
        Update the value function based on the outcome of an `episode`.

        Raises NotImplementedError.
        """
        raise NotImplementedError

    def play_round(self) -> None:
        """Play a single round of the game and update the value function."""
        self.update(self.simulate_episode())

    def play(self, num_rounds: int) -> None:
        """Play the game for `num_rounds` rounds."""
        for t in range(num_rounds):
            self.play_round()
            if (t % 1000000 == 0 and t > 0):
                print("{0} million games played.".format(int(t / 1000000)))


    def eps_greedy(self, s: State, eps0: float = None) -> int:
        """
        Choose an action for the player using an epsilon-greedy policy.

        Value of epsilon decays with the number of visits to the state by
        default but can also be kept at a constant value of `eps0`.

        Parameters
        ----------
        s : State
            Current state.
        eps0 : float, optional
            Use constant epsilon if provided. Default is None.

        Returns
        -------
        int
            Chosen action.

        """
        Q_hit, Q_stick = self.Q[s.get()]

        eps = eps0 if eps0 is not None else params.N0 / (params.N0 + sum(self.N[s.get()]))
        # The factor 0.5 comes from m (size of the action space) = 2
        # But it's actually not so important, it's effectively a rescaling of N0
        choose_greedy = np.random.rand(1)[0] < 1 - 0.5 * eps
        greedy, explore = (HIT, STICK) if Q_hit >= Q_stick else (STICK, HIT)
        return greedy if choose_greedy else explore


    def alpha(self, sa: StateAction) -> float:
        """Calculate the step-size based on the number of visits to the state `sa`."""
        return params.dt0 + 1.0 / max(self.N[sa], 1)


    def simulate_episode(self) -> Episode:
        """Simulate and return a single round of the game."""
        s = State(hit(True), hit(True)) # Initialize round and episode
        episode = Episode()

        while not s.is_terminal():
            a = self.eps_greedy(s)  # Choose action
            episode.advance(s, a)   # Take step

        return episode


    def get_Q(self) -> np.ndarray:
        """Return the Q-value array."""
        return self.Q

    def get_N(self) -> np.ndarray:
        """Return the array storing the number of visits to each state."""
        return self.N

    def get_params(self) -> dict:
        """
        Return the parameters used by the algorithm. Calls params21.get_params.
        """
        return params.get_params()

    def print_params(self) -> None:
        """
        Print the parameters used by the algorithm. Calls params21.print_params.
        """
        params.print_params()





### Functions
def hit(initial: bool = False) -> int:
    """
    Draw a single card.

    Parameters
    ----------
    initial : bool, optional
        First draw (meaning, only black cards can be drawn). Default is False.

    Returns
    -------
    int
        Value of card drawn.

    """
    draw = np.random.randint(1,11) # Uniform random integer between 1 and 10
    # Black if first draw or P(black) otherwise
    black = initial or np.random.rand(1)[0] > 1 - params.prob_black
    return draw if black else -draw


def hit_rigged(initial: bool = False, dealer_current: int = None) -> int:
    """
    Draw a single card. Like hit() but heavily biased toward the dealer.

    Dealer cannot go bust.

    Parameters
    ----------
    initial : bool, optional
        First draw (meaning, only black cards can be drawn). Default is False.
    dealer_current : int, optional
        Current dealer score. Default is None.

    Returns
    -------
    int
        Value of card drawn.

    """
    draw = hit(initial)
    if not initial and dealer_current is not None: # Only for dealer
        if (dealer_current <= 11):
            # No chance of going bust in the negative direction
            return abs(draw)
        elif (dealer_current + draw > 21):
            # No chance of going bust in the positive direction
            return -draw
        else:
            return draw
    else:
        return draw


def dealer_action(s: State) -> int:
    """Choose an action for the dealer based on the current state `s`."""
    if params.dealer_policy == "easy21":
        # Stick on 17 and above
        return HIT if 0 < s.get_dealer_sum() < params.dealer_stick_sum else STICK
    if params.dealer_policy == "random":
        return HIT if np.random.rand(1)[0] >= 0.5 else STICK
    if params.dealer_policy == "mia":
        # Stick after the initial draw
        return STICK
    if params.dealer_policy == "smart":
        # Stick if exceeding player score, accept draw near 21
        cutoff = params.dealer_stick_sum
        if (0 < s.get_dealer_sum() < cutoff and s.get_dealer_sum() <= s.get_player_sum()) or \
           (21 > s.get_dealer_sum() >= cutoff and s.get_dealer_sum() < s.get_player_sum()):
            return HIT
        else:
            return STICK

def play_out_round(s: State) -> State:
    """
    Make moves for the dealer until a terminal state is reached; return state.
    """
    while dealer_action(s) == HIT and not s.is_terminal():
        s.hit_dealer()

    s.set_terminal()   # Round ends with dealer's turn
    return s

def step(s: State, a: int) -> (State, int):
    """
    Advance the episode by one step: (S, A) -> (R, S').

    If the player hits, the episode either terminates if the player goes bust
    or continues otherwise. If the player sticks, the dealer makes his moves
    and the episode terminates.

    Parameters
    ----------
    s : State
        Current state.
    a : int
        Action taken by player.

    Returns
    -------
    (State, int)
        Terminal state and reward.

    """
    if (a == HIT):
        s.hit_player()
        return (s, -1) if s.player_is_bust() else (s, 0) # 0 reward because can hit again
    else: # Player sticks
        s = play_out_round(s)
        if s.dealer_is_bust() or s.get_dealer_sum() < s.get_player_sum():
            reward = 1
        else:
            reward = 0 if s.get_dealer_sum() == s.get_player_sum() else -1

        return s, reward