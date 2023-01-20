#!/usr/bin/env python

import simulation_tools as simu
import arviz as az
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import tensorflow as tf
import tensorflow_probability as tfp
from scipy.spatial import distance_matrix


def logit_experiment(
    radius, nTrain=50, nTest=5, nItems=15, nMCMC=500, nBurn=100, nChains=25
):
    """Run a simulation experiment for a logistic regression model."""
    tfd = tfp.distributions
    tfb = tfp.bijectors
    dtype = tf.float64

    print("-" * 80)
    print(
        f"Running logit experiment with {radius=}, {nTrain=}, {nItems=}, {nMCMC=}, {nBurn=}"
    )

    # experiment parameters
    rng_u = np.random.default_rng(seed=26121991)
    nTrials = nTrain + nTest
    nPoints = np.repeat(nItems, nTrials)  # points per tiral
    minx, maxx = -2, 2  # box extent

    # sample points
    lons = [rng_u.uniform(minx, maxx, size=n) for n in nPoints]  # X
    lats = [rng_u.uniform(minx, maxx, size=n) for n in nPoints]  # Y
    points = [np.vstack([lons[i], lats[i]]).T for i in range(nTrials)]

    # make quality data for DPP
    Xs = []
    for i in range(nTrials):
        Xs.append(
            np.vstack(
                (
                    np.ones(nPoints[i]),
                    np.sqrt(lons[i] ** 2 + lats[i] ** 2),
                    lats[i],
                    lons[i],
                )
            ).T
        )
    Zs = []
    for i in range(nTrials):
        Zs.append([distance_matrix(points[i], points[i])])

    # poisson thinning
    rng_t = np.random.default_rng(seed=18842524)
    beta0 = -5.0
    beta1 = 2.5

    labels = [np.ones(shape=n).astype(bool) for n in nPoints]  # all labels are 1
    probs = [np.exp(beta0 + X[:, 1] * beta1) for X in Xs]
    unifs = [rng_t.uniform(0, 1, size=n) for n in nPoints]
    retain = [(unifs[i] < probs[i]) for i in range(nTrials)]
    labels = [labels[i] * retain[i] for i in range(nTrials)]

    # sample matern
    labels = [
        simu.sample_matern_iii(Xs[i][:, 2], Zs[i][0], radius, labels=labels[i])
        for i in range(nTrials)
    ]

    # make tensors of data
    X_train = tf.constant(np.vstack(Xs[:nTrain]), dtype=dtype)
    y_train = tf.constant(np.hstack(labels[:nTrain]), dtype=dtype)
    X_test = tf.constant(np.vstack(Xs[nTrain:]), dtype=dtype)
    y_test = tf.constant(np.hstack(labels[nTrain:]), dtype=dtype)

    # save true test outcomes
    folder = "simulations/data/logit/"
    np.savetxt(
        folder + f"y_test_{nItems}_items_rad{radius:.1f}.csv", y_test, delimiter=","
    )
    y_test_saved = np.genfromtxt(
        folder + f"y_test_{nItems}_items_rad{radius:.1f}.csv",
        delimiter=",",
        usemask=False,
    )

    # declare prior
    nDims = X_train.shape[-1]
    means_prior = np.zeros(nDims)
    sigma_prior = 10

    beta_prior = tfd.MultivariateNormalDiag(
        loc=means_prior, scale_diag=np.ones(nDims) * sigma_prior, name="beta_prior"
    )

    def make_logit_logposterior(prior, X_train, y_train):
        """Functional closure for loglikelihood function."""

        def _logposterior(beta):
            """Compute linear response and evaluate."""
            return simu.logit_log_prob(beta, X_train, y_train) + prior.log_prob(beta)

        return _logposterior

    # get log posterior
    log_posterior = make_logit_logposterior(beta_prior, X_train, y_train)

    # setup NUTS for Bayesian Inference
    rng_p = np.random.default_rng(seed=3545412091)
    initial_state = rng_p.uniform(size=(nChains, nDims)) * 5

    # define nuts kernel
    nuts_kernel = tfp.mcmc.NoUTurnSampler(
        target_log_prob_fn=log_posterior, step_size=np.float64(0.1)
    )

    # nuts adaptive kernel
    nuts_adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
        nuts_kernel,
        num_adaptation_steps=int(nBurn * 0.8),
        step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(
            step_size=new_step_size
        ),
        step_size_getter_fn=lambda pkr: pkr.step_size,
        log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
        target_accept_prob=0.7,
    )

    @tf.function
    def run_chain(initial_state, num_results, num_burnin_steps):
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=nuts_adaptive_kernel,
            trace_fn=lambda current_state, kernel_results: kernel_results,
        )

    # run chain
    print("Running MCMC...")
    samples, kernel_results = run_chain(
        initial_state, num_results=nMCMC, num_burnin_steps=nBurn
    )
    print("Run completed")

    # summaries
    results = samples.numpy().T
    data = az.convert_to_dataset(
        {
            "beta0": results[0, :],
            "beta1": results[1, :],
            "beta2": results[2, :],
            "beta3": results[3, :],
        }
    )

    # get log accept ratio
    log_accept_ratio = kernel_results.inner_results.log_accept_ratio
    p_accept = tf.math.exp(
        tfp.math.reduce_logmeanexp(tf.minimum(log_accept_ratio, 0.0))
    )
    print("Acceptance ratio: \t %.3f" % p_accept.numpy())

    # save traceplot
    axes = az.plot_trace(data, combined=True, compact=True, figsize=(12, 12))
    fig = axes.ravel()[0].figure
    fig.savefig(f"simulations/plots/logit/traceplot_{nItems}_{radius}.png")

    # autocorrelation plots
    axes = az.plot_autocorr(data.sel(chain=[0]))
    fig = axes.ravel()[0].figure
    fig.savefig(f"simulations/plots/logit/autrocorrelation_{nItems}_{radius}.png")

    # save r-hat
    az.rhat(data).to_netcdf(f"simulations/data/logit/rhat_{nItems}_{radius:.2f}.json")

    # collect betas
    betas = samples[-1, :, :]

    # make predictions
    rng = np.random.default_rng(20122221201)
    p_pred = simu.logit_pred_prob(betas, X_test).numpy()
    predictions = rng.binomial(1, p=p_pred)  # sample predicted labels

    # save predictions
    folder = "simulations/data/logit/"
    np.savetxt(
        folder + f"pred_{nItems}_items_rad{radius:.1f}.csv", predictions, delimiter=","
    )

    # reshape predictions and print MCC
    predictions = predictions.reshape(nChains, nItems, nTest)
    y_test = y_test.numpy().reshape(nItems, nTest).T
    mcc, std = simu.summarize_mcc(predictions, y_test)

    print(f"Mean MCC is {mcc:.2f} (SE {std:.4f})")
    print("-" * 80)


if __name__ == "__main__":
    logit_experiment(float(sys.argv[1]))
