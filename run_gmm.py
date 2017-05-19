import numpy as np
import glob
import os
from sklearn.mixture import GaussianMixture


def get_components_min_bic(data, end_early=False, delta=2):
    """
    Get the number of number of gmm components which result in lowest bic score
    :param data: LI/BI coordinates of all sampled trees
    :param end_early: End on asymptotic convergence (default False)
    :param delta: Percent change threshold for convergence
    :return: Number of components
    """
    min_bic = 0
    prev_bic = 10000
    bic_clusters = 1

    size = 50 if data.size/2 > 50 else int(data.size/2 - 1)

    for i in range(size):
        gmm = GaussianMixture(n_components=i+1, n_init=2, covariance_type='full').fit(data)
        bic = gmm.bic(data)
        # Check for convergence
        if end_early and (prev_bic/bic) - 1 > - delta:
            return i + 1
        elif bic < min_bic:
            bic_clusters = i+1
            min_bic = bic
        prev_bic = bic
    return bic_clusters


def run_gmm(data):
    """
    Runs gmm using BIC to determine number of components, and return associated data
    :param data: LI/BI coordinates for all sampled trees
    :return: Weight, mean, covariance for each cluster, and assignments for each sampled tree
    """
    num_components = get_components_min_bic(data)
    gmm = GaussianMixture(n_components=num_components, n_init=2, covariance_type="full").fit(data)
    return gmm.weights_, gmm.means_, gmm.covariances_, gmm.predict(data)


def run_gmm_file(fn=None, remove_lin=False, path=""):
    """
    Run GMM clustering on a given tumor index file
    :param fn: Name of tumor index file
    :param remove_lin: Remove linear sampled trees from clustering
    :param path: Data output path
    :return: None 
    """
    if not fn:
        fn = input("File Name:\n")

    data = np.loadtxt("{}".format(fn), delimiter=",")
    if remove_lin:
        data = (data[data[:, 1] != 0])
        if data.size == 0:
            return
    gmm_results = run_gmm(data)
    f = os.path.basename(fn)
    f = f[0:len(f) - 10]
    extension = "_no_linear" if remove_lin else ""

    np.savetxt("{}{}{}.assigns.csv".format(path, f, extension), gmm_results[3])
    np.save("{}{}{}.weights".format(path, f, extension), gmm_results[0])
    np.save("{}{}{}.means".format(path, f, extension), gmm_results[1])
    np.save("{}{}{}.covariances".format(path, f, extension), gmm_results[2])
    np.save("{}{}{}.assigns".format(path, f, extension), gmm_results[3])


def run_gmm_directory(directory=None):
    """
    Runs gmm clustering on a directory containing tumor index files
    :return: None
    """
    if not directory:
        directory = input("Directory:\n")
    os.chdir(directory)
    for file in glob.glob("*.index.csv"):
        run_gmm_file(file, True, path="../test/")

if __name__ == "__main__":
    run_gmm_directory()

