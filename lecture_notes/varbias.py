import numpy as np

# set the seed
rng = np.random.default_rng(100)

n_samples = 10 # number of samples per trial
n_trials = 10**4 # number of independent trials

# default ddof = 0 is a biased estimator
# change to ddof = 1 to see unbiased estimator
ddof = 0  

# store the sample variances
sample_variances = np.zeros(n_trials)
for trial in range(n_trials):
    samples = rng.normal(loc=0.0, scale=1.0, size=n_samples) # sample from N(0,1)
    sample_variances[trial] = np.var(samples, ddof=ddof)

print("Mean of Sample Variances of size {} with ddof = {}: {:.4f}".format(n_samples, ddof, sample_variances.mean()))
