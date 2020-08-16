#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linear Function Approximation for the game of easy21.

This module stores various helper functions for using linear function
approximation in the module easy21.

"""

import numpy as np
from core.base import HIT, STICK, StateAction

_ds_map = {}
_ps_map = {}
_a_map = {}
_INITIALIZED = False

def __setup():
    # Cuboid encodings of each feature
    # Dealer card values: {[1,4], [4,7], [7,10]}
    _ds_map.update(dict.fromkeys([1, 2, 3], [1,0,0]))
    _ds_map[4] = [1,1,0]
    _ds_map.update(dict.fromkeys([5, 6], [0,1,0]))
    _ds_map[7] = [0,1,1]
    _ds_map.update(dict.fromkeys([8, 9, 10], [0,0,1]))

    # Player score values: {[1,6], [4,9], [7,12], [10,15], [13,18], [16,21]}
    _ps_map.update(dict.fromkeys([1, 2, 3], [1,0,0,0,0,0]))
    _ps_map.update(dict.fromkeys([4, 5, 6], [1,1,0,0,0,0]))
    _ps_map.update(dict.fromkeys([7, 8, 9], [0,1,1,0,0,0]))
    _ps_map.update(dict.fromkeys([10, 11, 12], [0,0,1,1,0,0]))
    _ps_map.update(dict.fromkeys([13, 14, 15], [0,0,0,1,1,0]))
    _ps_map.update(dict.fromkeys([16, 17, 18], [0,0,0,0,1,1]))
    _ps_map.update(dict.fromkeys([19, 20, 21], [0,0,0,0,0,1]))

    # Actions: {hit, stick}
    _a_map[0], _a_map[1] = [1, 0], [0, 1]


    # Convert to numpy arrays
    for k in _ds_map:
        _ds_map[k] = np.array(_ds_map[k])
    for k in _ps_map:
        _ps_map[k] = np.array(_ps_map[k])
    for k in _a_map:
        _a_map[k] = np.array(_a_map[k])

    global _INITIALIZED
    _INITIALIZED = True


def encode_features(sa: StateAction) -> np.ndarray:
    """Compute and return the feature vector for the state-action `sa`."""
    if not _INITIALIZED:
        __setup()

    phi = np.zeros((3,6,2))
    # np.ix_ makes sure dimensions are preserved when slicing
    phi[np.ix_(_ds_map[sa.ds] == 1, _ps_map[sa.ps] == 1, _a_map[sa.a] == 1)] = 1
    return phi.flatten()


def get_activations() -> dict:
    """Compute and return the activations of every possible state-action."""
    if not _INITIALIZED:
        __setup()

    activations = {}
    for a in [HIT, STICK]:
        for ps in range(1,22):
            for ds in range(1, 11):
                activations[(ds,ps,a)] = encode_features(StateAction(ds, ps, a))
    return activations