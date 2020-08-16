#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of the game Easy21.

This module contains functions and classes for applying model-free reinforcement
learning to the game of Easy21.

The available algorithms are Monte Carlo, forward-view Sarsa(lambda) and
backward-view Sarsa(lambda). Forward-view Sarsa(lambda) can be used together
with linear function approximation.

All algorithms assume no discounting of future rewards.

"""

import numpy as np
import core.params as params
from core.examples import do_assignment, run_examples
from core.base import State, Episode, ModelFreeControl, StateAction
from core.base import hit, step, HIT, STICK, SDIM
from core.plotting import plot_Q_and_N
from core import lfa
from typing import Union, Type

DO_ASSIGNMENT = False
RUN_EXAMPLES = False
SAVE_PLOTS = False

def main():
    """
    Main method that runs an example simulation using the MonteCarlo algorithm.

    Edit the global parameter `DO_ASSIGNMENT` to run simulations for the
    assignment, `RUN_EXAMPLES` to run some example simulations with different
    parameters for the algorithms and set `SAVE_PLOTS` to True to save figures.
    """
    np.random.seed(132624)

    print("Running an example of Monte Carlo control.")
    example_run(algorithm=MonteCarlo, num_rounds=100000,
                title="example", save_plots=SAVE_PLOTS)

    if DO_ASSIGNMENT:
        do_assignment(num_mc_rounds=1000000, save_plots=SAVE_PLOTS)

    if RUN_EXAMPLES:
        run_examples(num_rounds=100000, save_plots=SAVE_PLOTS)

    return 0


class MonteCarlo(ModelFreeControl):
    """
    A class for applying model-free Monte Carlo learning to the game of Easy21.

    Extends ModelFreeControl. The default parameters are as per the instructions
    of the Easy21 assignment.

    Attributes
    ----------
    dealer_policy : str
        The policy followed by the dealer, one of "easy21", "smart", "random",
        "mia". Default is "easy21".
    dealer_stick_sum : int
        Dealer's sum used by the dealer to decide on the next action (only for
        "easy21" and "smart" policies). Default is 17.
    first_visit : bool
        Use first-visit value-function updates, as opposed to every-visit.
        Default is False.
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

    def __init__(self, dealer_policy: str = "easy21", dealer_stick_sum: int = 17,
                 first_visit: bool = False, prob_black: float = 2./3, N0: float = 100.,
                 dt0: float = 0., rigged: bool = False,
                 Q_init: Union[np.ndarray, int] = 0,
                 N_init: Union[np.ndarray, int] = 0):
        """Initialize variables and parameters of the algorithm."""
        super().__init__(dealer_policy, dealer_stick_sum, prob_black, N0, dt0,
                         rigged, Q_init, N_init)
        params.first_visit = first_visit

    def update(self, episode: Episode) -> None:
        """
        Update the value function based on the outcome of an `episode`.

        After a round of the game, go through sequence of visited states and
        update action-values and the array storing the number of visits to
        each state.
        """
        if (params.first_visit):
            # First visit MC: update states only once per episode
            # Order does not matter because:
            # a) reward comes only at the end; b) there is no discounting
            episode.discard_duplicate_states()

        r = episode.get_outcome()
        for sa in episode.get_history():
            dt = self.alpha(sa)
            self.Q[sa] += dt * (r - self.Q[sa])
            self.N[sa] += 1




class SarsaFWD(ModelFreeControl):
    """
    A class for applying model-free forward-view SARSA(lambda) to the game of Easy21.

    Extends ModelFreeControl. The default parameters are as per the instructions
    of the Easy21 assignment.

    Can operate in two modes: with and without using Linear Function Approximation.

    Attributes
    ----------
    lmbda : float
        Value of the lambda parameter. Default is 0.0.
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
    use_linear_func_approx : bool
        Use linear function approximation. Default is False.
    eps0 : float
        Constant epsilon for eps-greedy policy. Only used when
        `use_linear_func_approx` = True. Default is 0.05.
    rigged : bool
        Use a rigged version of drawing cards that make it impossible for the
        dealer to go bust. Default is False.
    Q : np.ndarray or int
        Initial state-action value function. Default is 0.
    N : np.ndarray or int
        Initial array storing the number of times each state has been visited.
        Default is 0.
    w : np.ndarray or int
        Initial approximation to state-action value function. Only used when
        `use_linear_func_approx` = True. Default is 0.


    """

    def __init__(self, lmbda: float = 0.0, dealer_policy: str = "easy21",
                 dealer_stick_sum: int = 17, prob_black: float = 2./3,
                 N0: float = 100., dt0: float = 0., eps0: float = 0.05,
                 use_linear_func_approx: bool = False, rigged: bool = False,
                 Q_init: Union[np.ndarray, int] = 0,
                 N_init: Union[np.ndarray, int] = 0,
                 w_init: Union[np.ndarray, int] = 0):
        """Initialize variables and parameters of the algorithm."""
        super().__init__(dealer_policy, dealer_stick_sum, prob_black, N0, dt0,
                         rigged, Q_init, N_init)
        params.lmbda = lmbda
        self.use_lfa = use_linear_func_approx
        if use_linear_func_approx:
            assert params.dt0 > 0, "Must set step-size when using function approximation"
            self.w = w_init * np.ones(3 * 6 * 2) if type(w_init) is int else w_init.copy()
            params.eps0 = eps0
            self.activations = lfa.get_activations()


    def qt_nstep(self, state_action_seq: list) -> list:
        """Calculate undiscounted n-step returns for every state in `state_action_seq`."""
        qtn = []
        if len(state_action_seq) > 1:
            for sa in state_action_seq[1:]:
                qtn.append(self.qt_target(sa))

        return qtn


    def qt_target(self, sa: StateAction) -> float:
        """Calculate the TD-target for the state-action `sa`."""
        return np.dot(self.activations[sa], self.w) if self.use_lfa else self.Q[sa]


    def qt_lambda(self, state_action_seq: list, reward: int) -> float:
        """
        Calculate the lambda-weighted return for the first state in `state_action_seq`.

        Parameters
        ----------
        state_action_seq : list
            Sequence of state-actions until the end of the episode.
        reward : int
            The ultimate outcome of the episode.

        Returns
        -------
        qtl : float
            Lambda-weighted return for the first state in `state_action_seq`.

        """
        qtl = 0.0
        weights_sum = 0.0
        lambda_power = 1.0

        targets = self.qt_nstep(state_action_seq)
        for target in targets:
            weights_sum += lambda_power
            qtl += lambda_power * target
            lambda_power *= params.lmbda
        weights_sum *= 1.0 - params.lmbda
        qtl *= 1.0 - params.lmbda
        # The final return gets the remaining weight
        qtl += reward * (1.0 - weights_sum)

        return qtl


    def update(self, episode: Episode) -> None:
        """
        Update the value function based on the outcome of an `episode`.

        After a round of the game, go through sequence of visited states and
        update action-values (or their approximating function) and the array
        storing the number of visits to each state.
        """
        state_action_seq = episode.get_history()
        reward = episode.get_outcome()
        if self.use_lfa:
            dt = self.alpha(None) # Constant stepsize
            for k, sa in enumerate(state_action_seq):
                dw = self.activations[sa]
                q_hat = np.dot(dw, self.w)
                self.w += dt \
                    * (self.qt_lambda(state_action_seq[k:], reward) - q_hat) * dw
                self.N[sa] += 1  # This is tracked just for plotting purposes
        else:
            for k, sa in enumerate(state_action_seq):
                dt = self.alpha(sa)
                self.Q[sa] += dt \
                    * (self.qt_lambda(state_action_seq[k:], reward) - self.Q[sa])
                self.N[sa] += 1


    def eps_greedy(self, s: State) -> int:
        """
        Choose an action for the player using an epsilon-greedy policy.

        Use constant epsilon with linear function approximation and the method
        of the base class otherwise.
        """
        eps0 = None
        if self.use_lfa:
            # Set up Q entries to be able to use base method
            ds, ps = s.get_dealer_sum(), s.get_player_sum()
            s_HIT = StateAction(ds, ps, HIT)
            s_STICK = StateAction(ds, ps, STICK)
            self.Q[s_HIT] = np.dot(self.activations[s_HIT], self.w)
            self.Q[s_STICK] = np.dot(self.activations[s_STICK], self.w)
            eps0 = params.eps0
        return super().eps_greedy(s, eps0)


    def alpha(self, sa: StateAction) -> float:
        """
        Calculate the step-size to be used.

        Step-size is constant when using linear function approximation and
        chosen based on the number of visits to the state `sa` otherwise.
        """
        return params.dt0 if self.use_lfa else super().alpha(sa)


    def get_Q(self) -> np.ndarray:
        """Return the Q-value array."""
        if self.use_lfa:
            for sa in self.activations:
                self.Q[sa] = np.dot(self.activations[sa], self.w)
        return self.Q





class SarsaBWD(ModelFreeControl):
    """
    A class for applying model-free backward-view SARSA(lambda) to the game of Easy21.

    Extends ModelFreeControl. The default parameters are as per the instructions
    of the Easy21 assignment.

    Attributes
    ----------
    lmbda : float
        Value of the lambda parameter. Default is 0.0.
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
        Smallest allowed step-size. Default is 0.0.
    rigged : bool
        Use a rigged version of drawing cards that make it impossible for the
        dealer to go bust. Default is False.
    Q : np.ndarray or int
        Initial state-action value function. Default is 0.
    N : np.ndarray or int
        Initial array storing the number of times each state has been visited.
        Default is 0.
    E : np.ndarray or int
        Initial eligibility traces. Default is 0.

    """

    def __init__(self, lmbda: float = 0.0, dealer_policy: str = "easy21",
                 dealer_stick_sum: int = 17, prob_black: float = 2./3,
                 N0: float = 100., dt0: float = 0., rigged: bool = False,
                 Q_init: Union[np.ndarray, int] = 0,
                 N_init: Union[np.ndarray, int] = 0,
                 E_init: Union[np.ndarray, int] = 0):
        """Initialize variables and parameters of the algorithm."""
        super().__init__(dealer_policy, dealer_stick_sum, prob_black, N0, dt0,
                         rigged, Q_init, N_init)
        self.E = E_init * np.ones(SDIM) if type(E_init) is int else E_init.copy()
        params.lmbda = lmbda


    def update_eligibility_trace(self, sa: StateAction) -> None:
        """Update eligibility traces upon visiting state `sa`."""
        self.E *= params.lmbda
        self.E[sa] += 1


    def update(self, sa: StateAction, target: float) -> None:
        """Update the value function of state `sa` toward the `target`."""
        dt = self.alpha(sa)
        self.update_eligibility_trace(sa)
        self.Q[sa] += dt * (target - self.Q[sa]) * self.E[sa]
        self.N[sa] += 1

    def simulate_episode(self) -> None:
        """
        Simulate a single round of the game.

        Although the game is episodic, in backward-view Sarsa the parameters
        are updated after every step, i.e. after every action taken.
        """

        s = State(hit(True), hit(True)) # Initialize round and episode
        a = self.eps_greedy(s)          # Choose action

        while not s.is_terminal():
            s0 = s.get()
            s, r = step(s, a)
            # If state is terminal, next action is irrelevant
            a2 = None if s.is_terminal() else self.eps_greedy(s)
            target = r if s.is_terminal() else self.Q[(*s.get(), a2)]
            self.update((*s0, a), target)
            a = a2

    def play_round(self) -> None:
        """Play a single round of the game and update the value function."""
        self.simulate_episode()


    def get_E(self) -> np.ndarray:
        """Return the eligibility trace array."""
        return self.E



def example_run(algorithm: Type[ModelFreeControl], num_rounds: int,
                parameters: dict = {}, title: str = "", dir_path: str = "img",
                save_plots: bool = False) -> None:
    """
    Wrapper function for calling any of the control algorithms.

    Parameters
    ----------
    algorithm : Type[ModelFreeControl]
        The algorithm to be used. One of MonteCarlo, SarsaFWD, SarsaBWD.
    num_rounds : int
        Number of episodes to run the control algorithm for.
    parameters : dict, optional
        Parameters of the algorithm passed as a dictionary. When empty, the
        default parameters are used. Default is {}.
    title : str, optional
        Title for the figures. Default is "".
    dir_path : str, optional
        Path of directory to save figure in, relative to base path.
        Default is 'img'.
    save_plots : bool, optional
        Save the figures in the `dir_path` folder. Default is False.


    """
    assert algorithm in [MonteCarlo, SarsaFWD, SarsaBWD], \
        "Invalid algorithm {0}".format(algorithm)

    alg = algorithm(**parameters)
    alg.play(num_rounds)
    plot_Q_and_N(alg.get_Q(), alg.get_N(), title, dir_path, save_plots)



if __name__ == "__main__":
    main()