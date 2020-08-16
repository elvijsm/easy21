#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameters for the game of easy21.

This module stores various parameters used by the module easy21.

The parameters are intialized upon creating an instance of one of the control
algorithms.

"""

N0: float = None              # eps-greedy time-variation constant
eps0: float = None            # constant epsilon for eps-greedy
dt0: float = None             # step-size time-variation constant
dealer_policy: str = None     # dealer policy
dealer_stick_sum: int = None  # dealer is biased towards sticking at this sum and above
prob_black: float = None      # probability of drawing a black, as opposed to a red card
rigged: bool = None           # use rigged rules (dealer cannot go bust)
first_visit: bool = None      # first-visit as opposed to every-visit Monte Carlo
lmbda: float = None           # lambda for Sarsa

def get_params() -> dict:
    """Return a dictionary containing the parameters of the game."""
    conf = {}

    conf["N0"] = N0
    conf["dt0"] = dt0
    conf["dealer_policy"] = dealer_policy
    conf["dealer_stick_sum"] = dealer_stick_sum
    conf["prob_black"] = prob_black
    conf["rigged"] = rigged
    if first_visit is not None:
        conf["first_visit"] = first_visit
    if eps0 is not None:
        conf["eps0"] = eps0
    if lmbda is not None:
        conf["lambda"] = lmbda

    return conf

def print_params() -> None:
    """Print the parameters of the game using simple formatting."""
    print("Parameters:")
    for k, v in get_params().items():
        print("  {0} = {1}".format(k, v))
