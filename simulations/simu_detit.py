import detit.models as dtm
import detit.quality
import detit.similarity
import detit.data
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


def detit_experiment(
    radius, nTrain=50, nTest=5, nItems=15, nMCMC=500, nBurn=100, nChains=25
):
    """Run a simulation experiment for a logistic regression model."""
    tfd = tfp.distributions
    tfb = tfp.bijectors
    dtype = tf.float64

    print("-" * 80)
    print(
        f"Running detit experiment with {radius=}, {nTrain=}, {nItems=}, {nMCMC=}, {nBurn=}"
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

    # make arrays
    Zs = [np.array(Z).T for Z in Zs]  # entries must be (nItems, nItems, nDims)

    # train-test split
    Xs_train = Xs[:nTrain]
    Zs_train = Zs[:nTrain]
    labels_train = labels[:nTrain]
    Xs_test = Xs[nTrain:]
    Zs_test = Zs[nTrain:]
    labels_test = labels[nTrain:]

    dsTrain = detit.data.Dataset.fromNumpy(
        labels=labels_train, Xs=Xs_train, Zs=Zs_train
    )

    # specify L-ensemble via quality and similarity
    quality = detit.quality.Exponential()
    similarity = detit.similarity.RBF()
    ensemble = dtm.Ensemble(quality, similarity)

    # prior over beta
    nDimsQual = Xs_train[0].shape[-1]
    nDimsSimi = Zs_train[0].shape[-1]
    nDims = nDimsQual + nDimsSimi
    prior_loc = tf.constant(np.zeros(nDimsQual), dtype=dtype)
    prior_scale = tf.constant(np.repeat(10, nDimsQual), dtype=dtype)
    prior_beta = tfd.Normal(loc=prior_loc, scale=prior_scale)
    prior_beta = tfd.Independent(distribution=prior_beta, reinterpreted_batch_ndims=1)

    # prior over ell (log-normal)
    loc = tf.constant([-1], dtype=dtype)
    scale = tf.constant([1.25], dtype=dtype)
    prior_ell = tfd.Normal(loc, scale, validate_args=True, allow_nan_stats=True)
    prior_ell = tfd.Independent(distribution=prior_ell, reinterpreted_batch_ndims=1)

    # get a joint prior
    joint_prior = tfd.JointDistributionSequential(
        [prior_beta, prior_ell], batch_ndims=0, use_vectorized_map=True
    )
    prior = tfd.Blockwise(joint_prior)

    # get our log-probability function
    log_posterior = ensemble.likelihoodClosure(dsTrain, nDimsQual, prior)

    # sample initial MCMC state
    rng_p = np.random.default_rng(seed=3545412092)  # dedicated rng
    lows = np.array([-2, -2, -2, -2.5])
    upps = np.array([2, 2, 2, -1.5])
    params_init = rng_p.uniform(lows, upps, size=(nChains, nDims))
    initial_state = tf.constant(params_init, dtype=dtype)

    # transition kernel
    hcm_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=log_posterior, step_size=0.15, num_leapfrog_steps=1
    )

    # adaptive rule for inner kernel stepsize
    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=hcm_kernel, num_adaptation_steps=int(nBurnin * 0.8)
    )

    # @tf.function - tracing not supported yet...
    def run_chain(initial_state, num_results, num_burnin_steps):
        return tfp.mcmc.sample_chain(
            num_results=num_results,
            num_burnin_steps=num_burnin_steps,
            current_state=initial_state,
            kernel=adaptive_hmc,
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
            #'beta3': results[3, :],
            "logls": results[3, :],
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
    fig.savefig(f"simulations/plots/detit/traceplot_{nItems}_{radius}.png")

    # autocorrelation plots
    axes = az.plot_autocorr(data.sel(chain=[0]))
    fig = axes.ravel()[0].figure
    fig.savefig(f"simulations/plots/detit/autrocorrelation_{nItems}_{radius}.png")

    # save r-hat
    az.rhat(data).to_netcdf(f"simulations/data/detit/rhat_{nItems}_{radius:.2f}.json")

    # collect betas
    betas = samples[-1, :, :]

    # make predictions
    rng = np.random.default_rng(20122221201)
    p_pred = simu.clogit_pred_prob(betas, X_test).numpy()

    # collect betas
    params = samples[-1, :, :]
    print(params.shape)

    # make testing data to tensorts and expand dims for batch dimension
    Xs_test = detit.data.toTensors(Xs_test)
    Zs_test = detit.data.toTensors(Zs_test)
    Xs_test = [tf.expand_dims(X, axis=0) for X in Xs_test]
    Zs_test = [tf.expand_dims(Z, axis=0) for Z in Zs_test]

    # form predictions
    nTest = nTrials - nTrain
    predictions = np.zeros(shape=(nChains, nItems, nTest))

    # form [nChains, nTest, nItems, nItems] tensor of L-ensembles
    Ls = [
        ensemble(params[:, :nDimsQual], params[:, nDimsQual:], X, Z)
        for X, Z in zip(Xs_test, Zs_test)
    ]

    for chain in range(nChains):
        for trial in range(nTest):
            L = Ls[trial][chain, 0, ...].numpy()
            K = np.eye(nItems) - np.linalg.inv(np.eye(nItems) + L)
            selection = simu.dpp_sampler_generic_kernel(K, rng)
            # label selected items with ones
            for item in selection:
                predictions[chain, item, trial] = 1

    # summarize results
    mcc, std = simu.summarize_mcc(predictions, y_test.numpy())

    print(f"Mean MCC is {mcc:.2f} (SE {std:.4f})")
    print("-" * 80)


if __name__ == "__main__":
    detit_experiment(float(sys.argv[1]))
