import itertools
import glob, os
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl

from operator import itemgetter
from sklearn import mixture
from sklearn.mixture import GaussianMixture

colour_iter = itertools.cycle(['black', 'navy', 'cornflowerblue', 'gold', 'brown'])

def load_gmm_results(fn, lin=False):
    if (lin):
        gmm_weights = np.load("../data/gmm_output/"+fn+"_nolin.weights.npy") 
        gmm_means = np.load("../data/gmm_output/"+fn+"_nolin.means.npy")
        gmm_covars = np.load("../data/gmm_output/"+fn+"_nolin.covars.npy")
        gmm_assigns= np.load("../data/gmm_output/"+fn+"_nolin.assigns.npy")
    else:
        gmm_weights = np.load("../data/gmm_output/"+fn+".weights.npy") 
        gmm_means = np.load("../data/gmm_output/"+fn+".means.npy")
        gmm_covars = np.load("../data/gmm_output/"+fn+".covars.npy")
        gmm_assigns= np.load("../data/gmm_output/"+fn+".assigns.npy")
        
    return (gmm_weights, gmm_means, gmm_covars, gmm_assigns)
    
def plot_clustering(X, Y_, weights, means, covariances, title):
    ''' (NP Array, NP Array, NP Array, NP Matrix, NP Matrix, String) -> None

    Plot GMM clustering result
    '''
    splot = plt.subplot(1, 1, 1)
    for i, (weight, mean, covar, colour) in enumerate(zip(weights, means, covariances, colour_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])

        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        # Plot ellipses
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=colour)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(weight)
        splot.add_artist(ell)
        # Label amplitude and num clusters
        splot.text(mean[0] + 0.005, mean[1] + 0.005, str(weight))
        splot.text(mean[0] + 0.005, mean[1] + 0.0025, str(len(X[Y_==i])))
        if not np.any(Y_ == i):
            continue
        # Plot constituents
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], s=15, color=colour)
        splot.text(mean[0] + 0.005, mean[1] + 0.0075, i)

    plt.xticks()
    plt.xlabel("Linearity Index (LI)", fontsize=13)
    plt.yticks()
    plt.ylabel("Branching Index (BI)", fontsize=13)
    plt.title(title, fontsize=17)

def plot_clustering_file():
    fn = input("Filename:\n")
    lin = False
    results = load_gmm_results(fn[0:len(fn)-10], lin)
    data = np.loadtxt("../data/index/"+fn, delimiter=",")
    if (lin):
        data = (data[data[:,1] != 0])
    title = 'GMM Clustering on {} ({} clusters)'.format(fn[0:4], len(results[0]))
    plot_clustering(data, results[3], results[0], results[1], results[2], title)
    plt.show()


def get_cluster_sizes(assignments, k):
    ''' (NP Array, Number) -> Array
    Get number of trees per GMM cluster sorted in descending order
    '''
    num_trees = [0] * k
    for assignment in assignments:
        num_trees[assignment] += 1
    return sorted(num_trees, reverse=True)

def get_median_clust_size():
    dn = input("Directory:\n")
    os.chdir(dn)

    cluster_sizes = {i: [] for i in range(1, 51)}
    medians = {i: None for i in range(1, 51)}

    for file in glob.glob('*.csv'):
        X = np.loadtxt(file, delimiter=',')
        bic_result = get_min_bic_clusters(X)
        gmm = mixture.GaussianMixture(n_components=bic_result[0], covariance_type='full').fit(X)
        
        cluster_sizes[gmm.n_components].append(get_cluster_sizes(gmm.predict(X), gmm.n_components))
    for k in cluster_sizes:
        medians[k] = np.median(cluster_sizes[k], axis=0)

    return medians

def plot_medians(medians):
    for key in medians:
        if not np.any(np.isnan(medians[key])):        
            for i in range(len(medians[key])):
                if (i is not 0):
                    s = 0
                    for j in range(i):
                        s += medians[key][j]
                    plt.bar(key, medians[key][i], bottom = s)
                else:
                    plt.bar(key, medians[key][i])
    plt.show()

def plot_descending_clust_size():
    dn = input("Directory:\n")
    os.chdir(dn)

    num_tumors = len(glob.glob("*.csv"))
    num_assignments = {i: [] for i in range(1, 51)}

    for file in glob.glob("*.csv"):
        X = np.loadtxt(file, delimiter=",")
        bic_result = get_min_bic_clusters(X)
        gmm = mixture.GaussianMixture(n_components=bic_result[0], covariance_type="full").fit(X)
        num_assignments[bic_result[0]].append(get_cluster_sizes(gmm.predict(X), bic_result[0]))

    for k in num_assignments:
        num_assignments[k].sort(reverse=True)

    it = 0
    for j in range(1, 51):
        for l in range(len(num_assignments[j])):
            for m in range(len(num_assignments[j][l])):
                if m is 0:
                    plt.bar(it, num_assignments[j][l][m], color='seagreen', edgecolor='black', linewidth='1px') 
                else:
                    s = 0
                    for n in range(m):
                        s += num_assignments[j][l][n]
                    plt.bar(it, num_assignments[j][l][m], color='mediumseagreen', edgecolor='black', linewidth='1px', bottom = s) 
            it += 1
    plt.show()

def get_bad_clust():
    dn = input("Directory:\n")
    os.chdir(dn)

    num_tumors = len(glob.glob("*.csv"))
    num_assignments = {i: [] for i in range(1, 51)}

    for file in glob.glob("*.csv"):
        X = np.loadtxt(file, delimiter=",")
        bic_result = get_min_bic_clusters(X)
        gmm = mixture.GaussianMixture(n_components=bic_result[0], covariance_type="full").fit(X)
        if sum(get_cluster_sizes(gmm.predict(X), bic_result[0])) != 2500:
            print(file)

if __name__ == "__main__":
    plot_clustering_file()
