from scipy import linalg
from scipy import stats
import numpy as np
import glob, os

from sklearn.mixture import GaussianMixture

def get_bic_clusters(X, end_early=False, delta=2):
    min_bic = 0
    prev_bic = 10000
    bic_clusters = 1
    for i in range (50):
        gmm = GaussianMixture(n_components=i+1, n_init = 2, covariance_type='full').fit(X)
        bic = gmm.bic(X)
        if (bic < min_bic):
            bic_clusters = i+1
            min_bic = bic
        elif (end_early and prev_biv*100/bic < delta):
            bic_clusters = i+1
            min_bic = bic
        prev_bic = bic
    return bic_clusters

def gmm(X, k):
    gmm = GaussianMixture(n_components=k, n_init=2, covariance_type="full").fit(X)
    assign = gmm.predict(X)
    return (gmm.weights_, gmm.means_, gmm.covariances_, assign)

def run_gmm_directory():
    dn = input("Directory:\n")
    os.chdir(dn)
    for file in glob.glob("*.csv"):
        run_gmm_file(file)

def run_gmm_file(fn=None):
    if not fn:
        fn = input("File Name:\n")
    X = np.loadtxt(fn, delimiter=",")
    gmm_results = gmm(X, get_bic_clusters(X))
    f = os.path.basename(fn)[0:len(fn)-4]
    print(f)

    np.savetxt("data/gmm_output/{}_assigns.csv".format(f), gmm_results[3])
    np.save("data/gmm_output/{}_weights".format(f), gmm_results[0])
    np.save("data/gmm_output/{}_means".format(f), gmm_results[1])
    np.save("data/gmm_output/{}_covars".format(f), gmm_results[2])
    np.save("data/gmm_output/{}_assigns".format(f), gmm_results[3])
    
if __name__ == "__main__":
    run_gmm_directory()
    
    
        
