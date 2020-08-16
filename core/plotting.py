#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting utilities for the game of easy21.

This module stores various plotting utility functions for the module easy21.

"""
import matplotlib.pyplot as plt
import numpy as np

def plot_Q_and_N(Q: np.ndarray, N: np.ndarray, title: str = "",
                 dir_path: str = "img", savefig: bool = False):
    """
    Wrapper function for plotting the action-value function array.

    Parameters
    ----------
    Q : np.ndarray
        Array storing the action-value function.
    N : np.ndarray
        Array storing the number of visits to each state.
    title : str, optional
        Title for the panels. Default is "".
    dir_path : str, optional
        Path of directory to save figure in, relative to base path.
        Default is 'img'.
    savefig : bool, optional
        Save the plot in the `dir_path` directory. Default is False.

    """
    # Remove information-less row and column
    Q = Q[1:,:,:]
    Q = Q[:,1:,:]
    N = N[1:,:,:]
    N = N[:,1:,:]
    Ntot = np.sum(N, axis = 2)

    # Plot optimal value function
    plot_panel(np.max(Q, axis = 2), Ntot, indx=1, title=title,
               dir_path=dir_path, savefig=savefig)
    # Plot HIT value function
    plot_panel(Q[:,:,0], np.divide(N[:,:,0], Ntot), indx=2, title=title,
               dir_path=dir_path, savefig=savefig)
    # Plot STICK value function
    plot_panel(Q[:,:,1], np.divide(N[:,:,1], Ntot), indx=3, title=title,
               dir_path=dir_path, savefig=savefig)

    print("Results of '{0}'".format(title))
    print("   Reward per game:", np.sum(np.multiply(Q, N))/np.sum(N))
    print("   Reward per game with optimal policy:",
          np.sum(np.multiply(np.max(Q, axis = 2), np.sum(N, axis = 2)))
          / np.sum(N))
    # Map back to original index since we cut the array in the beginning
    indx = np.array(np.unravel_index(np.argmax(Q, axis=None), Q.shape))
    indx[:-1] += 1
    print("   Best state = {0} with Q = {1}".format(tuple(indx), np.max(Q)))
    indx = np.array(np.unravel_index(np.argmin(Q, axis=None), Q.shape))
    indx[:-1] += 1
    print("   Worst state = {0} with Q = {1}".format(tuple(indx), np.min(Q)))


def plot_panel(Q: np.ndarray, N: np.ndarray, indx: int,
               title: str = "", dir_path: str = "img",
               savefig: bool = False) -> None:
    """
    Plot the action-value function and the number of visits to each state.

    Plots a 2D figure using matplotlib imshow.

    Parameters
    ----------
    Q : np.ndarray
        Array storing the action-value function.
    N : np.ndarray
        Array storing the number of visits to each state.
    indx : int
        Panel index. 1: optimal value function, 2: hit, 3: stick
    title : str, optional
        Title for the panel. Default is "".
    dir_path : str, optional
        Path of directory to save figure in, relative to base path.
        Default is "img".
    savefig : bool, optional
        Save the plot in the `dir_path` directory. Default is False.

    """
    extent = [0.5,21.5,0.5,10.5]
    xticks = [1,3,5,7,9,11,13,15,17,19,21]
    yticks = [1,4,7,10]
    paren = {1: "s", 2: "s, hit", 3: "s, stick"}
    suffix = {1: "optimal", 2: "hit",  3: "stick"}


    fig = plt.figure(figsize=(16,11))
    ax = fig.add_subplot(111, frameon=False)
    ax.xaxis.set_label_coords(0.45, -0.05)
    ax.yaxis.set_label_coords(0.11, 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("agent sum")
    ax.set_ylabel("dealer showing")
    ax = fig.add_subplot(211)
    cax = ax.imshow(Q, vmin = -1, vmax = 1, extent = extent, origin = "lower")
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_title(title)
    cbar = fig.colorbar(cax)
    cbar.set_label("Q({})".format(paren[indx]), rotation = 90)
    ax = fig.add_subplot(212)
    if indx == 1:
        cax = ax.imshow(np.log(N + 1), extent = extent, origin = "lower")
    else:
        cax = ax.imshow(N * 100, extent = extent, vmin = 0, vmax = 100,
                        origin = "lower")

    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    cbar = fig.colorbar(cax)
    if indx == 1:
        cbar.set_label("log(N({0}))".format(paren[indx]), rotation = 90)
    else:
        cbar.set_label("N({0}) (%)".format(paren[indx]), rotation = 90)
    if savefig:
        dir_path = "." if not dir_path else dir_path.rstrip("/")
        plt.savefig("{0}/{1}_{2}.pdf".format(dir_path, title, suffix[indx]),
                    tight_layout=True, bbox_inches='tight')
        plt.close()


def plot_mse(lambdas: np.ndarray, mse: np.ndarray,
             title: str = "", dir_path: str = "img/assignment",
             savefig: bool = False) -> None:
    """Plot Mean Squared Error versus lambda. Optionally save the figure."""
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111)
    ax.plot(lambdas, mse, 'o-')
    ax.set_xlabel("lambda")
    ax.set_ylabel("MSE")
    ax.set_title(title)
    if savefig:
        dir_path = "." if not dir_path else dir_path.rstrip("/")
        plt.savefig("{0}/MSE_vs_lambda_{1}.pdf".format(dir_path, title),
                    tight_layout=True, bbox_inches='tight')
        plt.close()

def plot_mse_evolution(episodes: np.ndarray, mse0: np.ndarray,
                       mse1: np.ndarray, title: str = "",
                       dir_path: str = "img/assignment",
                       savefig: bool = False) -> None:
    """
    Plot Mean Squared Error versus episode number for lambda = 0 and 1.
    Optionally save the figure.
    """
    fig = plt.figure(figsize=(12,7))
    ax = fig.add_subplot(111)
    ax.plot(episodes, mse0, 'bo-')
    ax.plot(episodes, mse1, 'ro-')
    ax.set_xlabel("Episode number")
    ax.set_ylabel("MSE")
    ax.legend(["lambda = 0", "lambda = 1"])
    ax.set_title(title)
    if savefig:
        dir_path = "." if not dir_path else dir_path.rstrip("/")
        plt.savefig("{0}/MSE_vs_episode_{1}.pdf".format(dir_path, title),
                    tight_layout=True, bbox_inches='tight')
        plt.close()

