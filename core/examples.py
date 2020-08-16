#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Examples of Easy21.

A simple collection of various example runs for the game of Easy21.
"""
import numpy as np
from core.plotting import plot_Q_and_N, plot_mse, plot_mse_evolution

def do_assignment(num_mc_rounds: int, num_sarsa_rounds: int = 1000,
                  save_plots: bool = False) -> None:
    """
    Simple wrapper function for producing the figures for the assignment.

    Parameters
    ----------
    num_mc_rounds : int
        Number of episodes to run Monte Carlo control for.
    num_sarsa_rounds : int, optional
        Number of episodes to run Sarsa(lambda) for for each lambda.
        Default is 1000.
    save_plots : bool, optional
        Save the figures in the ./img folder. Default is False.

    """
    from easy21 import MonteCarlo, SarsaFWD
    np.random.seed(132624)

    print("Monte Carlo control for {0} rounds".format(num_mc_rounds))
    mc = MonteCarlo()
    mc.play(num_mc_rounds)
    plot_Q_and_N(mc.get_Q(), mc.get_N(), "MC", dir_path="img/assignment",
                 savefig=save_plots)

    print("Sarsa(lambda) for {0} rounds for each lambda".format(
        num_sarsa_rounds
    ))
    mse = []
    lambdas = np.linspace(0, 1, 11)
    for lmbda in lambdas:
        # Compute mean squared error (compared to Monte Carlo) after
        # `num_sarsa_rounds` rounds for each value of lambda
        sarsa = SarsaFWD(lmbda = lmbda)
        sarsa.play(num_sarsa_rounds)
        dQ = sarsa.get_Q() - mc.get_Q()
        mse.append(np.mean(np.multiply(dQ, dQ)))

    plot_mse(lambdas, mse, "Sarsa", dir_path="img/assignment",
             savefig=save_plots)

    episodes, mse0, mse1 = [], [], []
    sarsa0 = SarsaFWD(lmbda = 0.0)
    sarsa1 = SarsaFWD(lmbda = 1.0)
    batch = 10
    for i in range(int(num_sarsa_rounds / batch) + 1):
        sarsa0.play(batch)
        sarsa1.play(batch)
        dQ = sarsa0.get_Q() - mc.get_Q()
        mse0.append(np.mean(np.multiply(dQ, dQ)))
        dQ = sarsa1.get_Q() - mc.get_Q()
        mse1.append(np.mean(np.multiply(dQ, dQ)))
        episodes.append(i * batch)

    plot_mse_evolution(episodes, mse0, mse1, "Sarsa",
                       dir_path="img/assignment", savefig=save_plots)

    print("Sarsa(lambda) with function approximation")
    mse = []
    for lmbda in lambdas:
        sarsa = SarsaFWD(lmbda = lmbda, use_linear_func_approx=True, dt0=0.01)
        sarsa.play(num_sarsa_rounds)
        dQ = sarsa.get_Q() - mc.get_Q()
        mse.append(np.mean(np.multiply(dQ, dQ)))

    plot_mse(lambdas, mse, "Sarsa_lfa",
             dir_path="img/assignment", savefig=save_plots)

    mse0, mse1 = [], []
    sarsa0 = SarsaFWD(lmbda = 0.0, use_linear_func_approx=True, dt0=0.01)
    sarsa1 = SarsaFWD(lmbda = 1.0, use_linear_func_approx=True, dt0=0.01)
    for i in range(int(num_sarsa_rounds / batch) + 1):
        sarsa0.play(batch)
        sarsa1.play(batch)
        dQ = sarsa0.get_Q() - mc.get_Q()
        mse0.append(np.mean(np.multiply(dQ, dQ)))
        dQ = sarsa1.get_Q() - mc.get_Q()
        mse1.append(np.mean(np.multiply(dQ, dQ)))

    plot_mse_evolution(episodes, mse0, mse1, "Sarsa_lfa",
                       dir_path="img/assignment", savefig=save_plots)



def run_examples(num_rounds: int = 10000, save_plots: bool = False) -> None:
    """A simple collection of various examples, each run for `num_rounds`."""
    from easy21 import MonteCarlo, SarsaFWD, SarsaBWD, example_run

    np.random.seed(132624)

    # Some Monte Carlo examples
    example_run(algorithm=MonteCarlo, num_rounds=num_rounds,
                parameters={},
                title="MC", dir_path="img/examples",
                save_plots=save_plots)

    example_run(algorithm=MonteCarlo, num_rounds=num_rounds,
                parameters={"first_visit": True},
                title="MC_first-visit", dir_path="img/examples",
                save_plots=save_plots)

    example_run(algorithm=MonteCarlo, num_rounds=num_rounds,
                parameters={"dealer_policy": "smart"},
                title="MC_dealer-smart", dir_path="img/examples",
                save_plots=save_plots)

    example_run(algorithm=MonteCarlo, num_rounds=num_rounds,
                parameters={"dealer_policy": "random"},
                title="MC_dealer-random", dir_path="img/examples",
                save_plots=save_plots)

    example_run(algorithm=MonteCarlo, num_rounds=num_rounds,
                parameters={"dealer_policy": "mia"},
                title="MC_dealer-mia", dir_path="img/examples",
                save_plots=save_plots)

    example_run(algorithm=MonteCarlo, num_rounds=num_rounds,
                parameters={"prob_black": 0.1},
                title="MC_Pblack-0.1", dir_path="img/examples",
                save_plots=save_plots)

    example_run(algorithm=MonteCarlo, num_rounds=num_rounds,
                parameters={"prob_black": 0.9},
                title="MC_Pblack-0.9", dir_path="img/examples",
                save_plots=save_plots)

    example_run(algorithm=MonteCarlo, num_rounds=num_rounds,
                parameters={"dealer_policy": "smart", "prob_black": 0.1},
                title="MC_dealer-smart_Pblack-0.1",
                dir_path="img/examples", save_plots=save_plots)

    example_run(algorithm=MonteCarlo, num_rounds=num_rounds,
                parameters={"dealer_policy": "smart", "prob_black": 0.9},
                title="MC_dealer-smart_Pblack-0.9",
                dir_path="img/examples", save_plots=save_plots)

    example_run(algorithm=MonteCarlo, num_rounds=num_rounds,
                parameters={"rigged": True},
                title="MC_rigged", dir_path="img/examples",
                save_plots=save_plots)


    # Some forward-view Sarsa(lambda) examples
    example_run(algorithm=SarsaFWD, num_rounds=num_rounds,
                parameters={"lmbda": 0.5},
                title="SarsaFWD_lambda-0.5", dir_path="img/examples",
                save_plots=save_plots)

    example_run(algorithm=SarsaFWD, num_rounds=num_rounds,
                parameters={"lmbda": 0.5, "dealer_policy": "smart"},
                title="SarsaFWD_lambda-0.5_dealer-smart",
                dir_path="img/examples", save_plots=save_plots)

    example_run(algorithm=SarsaFWD, num_rounds=num_rounds,
                parameters={"lmbda": 0.5, "dealer_policy": "random"},
                title="SarsaFWD_lambda-0.5_dealer-random",
                dir_path="img/examples", save_plots=save_plots)

    example_run(algorithm=SarsaFWD, num_rounds=num_rounds,
                parameters={"lmbda": 0.5, "dealer_policy": "mia"},
                title="SarsaFWD_lambda-0.5_dealer-mia",
                dir_path="img/examples", save_plots=save_plots)

    example_run(algorithm=SarsaFWD, num_rounds=num_rounds,
                parameters={"lmbda": 0.5, "rigged": True},
                title="SarsaFWD_lambda-0.5_rigged",
                dir_path="img/examples", save_plots=save_plots)

    example_run(algorithm=SarsaFWD, num_rounds=num_rounds,
                parameters={"lmbda": 0.0,
                            "use_linear_func_approx" : True,
                            "dt0" : 0.01},
                title="SarsaFWD_lambda-0.0_lfa", dir_path="img/examples",
                save_plots=save_plots)

    example_run(algorithm=SarsaFWD, num_rounds=num_rounds,
                parameters={"lmbda": 0.5,
                            "use_linear_func_approx" : True,
                            "dt0" : 0.01},
                title="SarsaFWD_lambda-0.5_lfa", dir_path="img/examples",
                save_plots=save_plots)

    example_run(algorithm=SarsaFWD, num_rounds=num_rounds,
                parameters={"lmbda": 1.0,
                            "use_linear_func_approx" : True,
                            "dt0" : 0.01},
                title="SarsaFWD_lambda-1.0_lfa", dir_path="img/examples",
                save_plots=save_plots)


    # Some backward-view Sarsa(lambda) examples
    example_run(algorithm=SarsaBWD, num_rounds=num_rounds,
                parameters={"lmbda": 0.5},
                title="SarsaBWD_lambda-0.5", dir_path="img/examples",
                save_plots=save_plots)

    example_run(algorithm=SarsaBWD, num_rounds=num_rounds,
                parameters={"lmbda": 0.5, "dealer_policy": "smart"},
                title="SarsaBWD_lambda-0.5_dealer-smart",
                dir_path="img/examples", save_plots=save_plots)

    example_run(algorithm=SarsaBWD, num_rounds=num_rounds,
                parameters={"lmbda": 0.5, "dealer_policy": "random"},
                title="SarsaBWD_lambda-0.5_dealer-random",
                dir_path="img/examples", save_plots=save_plots)

    example_run(algorithm=SarsaBWD, num_rounds=num_rounds,
                parameters={"lmbda": 0.5, "dealer_policy": "mia"},
                title="SarsaBWD_lambda-0.5_dealer-mia",
                dir_path="img/examples", save_plots=save_plots)

    example_run(algorithm=SarsaBWD, num_rounds=num_rounds,
                parameters={"lmbda": 0.5, "rigged": True},
                title="SarsaBWD_lambda-0.5_rigged",
                dir_path="img/examples", save_plots=save_plots)
