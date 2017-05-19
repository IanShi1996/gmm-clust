import itertools
import os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl


def load_gmm_results(fn, path="", lin=False):
    """
    Load pre-processed gmm clustering result data
    :param fn: Name of tumor sample
    :param path: OS path to data files
    :param lin: Use files with linear trees removed (default False)
    :return: NumPy representations of all gmm data files
    """
    extension = "_no_linear" if lin else ""
    gmm_weights = np.load("{}/{}{}.weights.npy".format(path, fn, extension))
    gmm_means = np.load("{}/{}{}.means.npy".format(path, fn, extension))
    gmm_covariances = np.load("{}/{}{}.covariances.npy".format(path, fn, extension))
    gmm_assigns = np.load("{}/{}{}.assigns.npy".format(path, fn, extension))

    return gmm_weights, gmm_means, gmm_covariances, gmm_assigns


def plot_clustering(trees, assignments, weights, means, covariances, title):
    """
    Plot gmm clustering results using pyplot (Adapted from sklearn example)
    :param trees: Coordinates (LI vs BI) for all sampled trees
    :param assignments: Cluster assignments for all trees
    :param weights: Cluster weights for all clusters
    :param means: Cluster means in LI/BI space for all clusters
    :param covariances: Covariance matrices for all clusters
    :param title: Title of plot
    :return: None
    """
    subplot = plt.subplot(1, 1, 1)
    colour_iter = itertools.cycle(['black', 'navy', 'blue', 'gold', 'brown'])
    for i, (weight, mean, covariance, colour) in enumerate(zip(weights, means, covariances, colour_iter)):
        # Note: Eigenvector v is direction of highest variance, w is direction of variance orthogonal to v
        v, w = linalg.eigh(covariance)
        v = np.multiply(2.,  np.sqrt(2.) * np.sqrt(v))
        u = w[0] / linalg.norm(w[0])
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi
        # Plot ellipses
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=colour)
        ell.set_clip_box(subplot.bbox)
        ell.set_alpha(weight)
        subplot.add_artist(ell)

        # Label amplitude, cluster number, and number of assignments
        subplot.text(mean[0], mean[1], str(weight))
        subplot.text(mean[0], mean[1] + 0.0025, str(len(trees[assignments == i])))
        subplot.text(mean[0], mean[1] + 0.005, i)
        if not np.any([assignments == i]):
            continue
        # Plot constituents
        plt.scatter(trees[assignments == i, 0], trees[assignments == i, 1], s=15, color=colour)

    plt.xticks()
    plt.yticks()
    plt.xlabel("Linearity Index (LI)", fontsize=13)
    plt.ylabel("Branching Index (BI)", fontsize=13)
    plt.title(title, fontsize=17)


def plot_clustering_tumor(linear=False):
    """
    Plot the gmm clustering of an individual tumor
    :return: None
    """
    path = input("Path:\n")
    directory = os.path.dirname(path)
    fn = os.path.basename(path)
    gmm_results = load_gmm_results(fn, directory, linear)
    data = np.loadtxt("{}/{}.index.csv".format(directory, fn), delimiter=",")
    if linear:
        # Remove linear elements for now
        data = (data[data[:, 1] != 0])
    title = 'GMM Clustering on {} ({} clusters)'.format(fn[0:4], len(gmm_results[0]))
    plot_clustering(data, gmm_results[3], gmm_results[0], gmm_results[1], gmm_results[2], title)
    plt.show()


if __name__ == "__main__":
    plot_clustering_tumor()
