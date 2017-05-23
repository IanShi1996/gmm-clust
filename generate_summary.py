import json
import numpy as np
import os
import glob


def write_json(file_name, data):
    """
    Write generated json data to file
    :param file_name: Output file name
    :param data: Tree summary data to write
    :return: None
    """
    os.chdir("result")
    f = open(file_name + "_clust.summ.json", "w")
    json.dump(data, f)
    f.close()


def round_array(f):
    """
    Rounds array of floats
    :param f: Array of floating point numbers
    :return: Rounded float
    """
    return list(map(lambda x: float(str("%.3f" % f[x])), f))


class SubCluster:
    """
    Represent a sub-cluster defined by a shared tree structure
    """
    def __init__(self, structure, assignments):
        """
        Create sub-cluster object
        :param structure: Shared tree structure
        :param assignments: Trees assigned to sub-cluster
        """
        self.tree_structure = structure
        self.trees = assignments
        self.num_trees = len(assignments)
        self.mean_node = int(self.calc_num_node())

        self.mean_cp = round_array(self.calc_mean_node_cp())
        self.std_cp = round_array(self.calc_std_node_cp())
        self.mean_ssm = round_array(self.calc_mean_node_ssm())
        self.std_ssm = round_array(self.calc_std_node_ssm())
        self.mean_cnv = round_array(self.calc_mean_node_cnv())
        self.std_cnv = round_array(self.calc_std_node_cnv())
        self.mean_llh = "%.3f" % self.calc_mean_llh()
        self.std_llh = "%.4f" % self.calc_std_llh()

    def calc_num_node(self):
        """
        Calculates the mean number of nodes per sampled tree
        :return: Mean number of nodes
        """
        mean_nodes = sum(map(lambda x: len(self.trees[x]["populations"]), self.trees)) / len(self.trees)
        if not mean_nodes.is_integer():
            print("Non whole number of nodes in sub-cluster: " + self.tree_structure)
        return mean_nodes

    def calc_mean_node_cp(self):
        """
        Calculate mean cellular prevalence across nodes rounded to 3 decimals
        :return: Mean cellular prevalence of nodes
        """
        cp_means = {i: 0 for i in range(self.mean_node)}

        for j in range(self.mean_node):
            for tree in self.trees:
                cp_means[j] += self.trees[tree]["populations"][str(j)]["cellular_prevalence"][0]
            # Round to 3 points
            cp_means[j] = cp_means[j] / len(self.trees)
        return cp_means

    def calc_std_node_cp(self):
        """
        Calculate standard deviation of cellular prevalence across nodes
        :return: Standard deviation of cellular prevalence of nodes
        """
        cps = {i: [] for i in range(self.mean_node)}
        cp_stds = {k: 0 for k in range(self.mean_node)}
        for j in range(self.mean_node):
            for tree in self.trees:
                cps[j].append(self.trees[tree]["populations"][str(j)]["cellular_prevalence"][0])
            # Round to 4 points
            cp_stds[j] = np.std(cps[j])
        return cp_stds

    def calc_mean_node_ssm(self):
        """
        Calculate mean number of ssms per node across trees in sub-cluster
        :return: Mean ssms for all nodes
        """
        ssm_means = {i: 0 for i in range(self.mean_node)}
        for j in range(self.mean_node):
            for tree in self.trees:
                ssm_means[j] += self.trees[tree]["populations"][str(j)]["num_ssms"]
            ssm_means[j] = ssm_means[j] / len(self.trees)
        return ssm_means

    def calc_std_node_ssm(self):
        """
        Calculate standard deviation of number of ssms per node across trees in a sub-cluster
        :return: Standard deviation of number of ssms for all nodes
        """
        ssms = {i: [] for i in range(self.mean_node)}
        ssm_stds = {j: 0 for j in range(self.mean_node)}
        for k in range(self.mean_node):
            for tree in self.trees:
                ssms[k].append(self.trees[tree]["populations"][str(k)]["num_ssms"])
            ssm_stds[k] = np.std(ssms[k])
        return ssm_stds

    def calc_mean_node_cnv(self):
        """
        Calculate mean copy number variation for all nodes across trees
        :return: Mean cnv for all trees
        """
        cnv_means = {i: 0 for i in range(self.mean_node)}
        for j in range(self.mean_node):
            for tree in self.trees:
                cnv_means[j] += self.trees[tree]["populations"][str(j)]["num_cnvs"]
            cnv_means[j] = cnv_means[j] / len(self.trees)
        return cnv_means

    def calc_std_node_cnv(self):
        """
        Calculate standard deviation of copy number variation for all nodes across trees
        :return: Standard deviation of number of cnvs across all trees
        """
        cnvs = {i: [] for i in range(self.mean_node)}
        cnv_stds = {j: 0 for j in range(self.mean_node)}
        for k in range(self.mean_node):
            for tree in self.trees:
                cnvs[k].append(self.trees[tree]["populations"][str(k)]["num_cnvs"])
            cnv_stds[k] = np.std(cnvs[k])
        return cnv_stds

    def calc_mean_llh(self):
        """
        Calculate the mean llh

        :return: Mean llh
        """
        return sum(map(lambda x: self.trees[x]["llh"], self.trees)) / len(self.trees)

    def calc_std_llh(self):
        """
        Calculate the standard deviation of llh across trees
        :return: Standard deviation of llh
        """
        return np.std([self.trees[tree]["llh"] for tree in self.trees])


class Cluster:
    """
    Represents a cluster of sampled trees
    """
    def __init__(self, name, trees, mean):
        """
        Create cluster object
        :param name: Name of cluster
        :param trees: Tree summaries of assignments to cluster
        :param mean: GMM clustering mean of cluster
        """
        self.name = name
        self.trees = trees

        self.li = mean[0]
        self.bi = mean[1]
        self.ci = 1 - (self.li + self.bi)

        self.subclusters = self.get_tree_structures()

    def get_tree_structures(self):
        """
        Create sub-cluster objects for each unique tree structure in a cluster
        :return: Returns array of unique tree structures
        """
        unique_structures = set(map(lambda x: str(self.trees[x]["structure"]), self.trees))
        structures = []
        for j, unique_structure in enumerate(unique_structures):
            # Get all tree structures matching unique structure
            a = {t: self.trees[t] for t in self.trees if str(self.trees[t]["structure"]) == unique_structure}
            # Create sub-cluster object
            structures.append(SubCluster(unique_structure, a))
        return structures


class Tumor:
    """
    Represents a tumor containing n sampled trees assigned to clusters determined by gmm clustering
    """
    def __init__(self, summaries, assignments, means):
        """
        Create Tumor object
        :param summaries: JSON object containing tree summaries
        :param assignments: Cluster assignments of each sampled tree
        :param means: GMM cluster mean of each cluster
        """
        self.trees = summaries
        self.assignments = assignments
        self.num_clusters = len(set(self.assignments))
        self.clusters = []

        for c_num in range(self.num_clusters):
            # Assign all trees by cluster number
            trees = {k: v for k, v in self.trees.items() if assignments[int(k)] == c_num}
            self.clusters.append(Cluster(c_num, trees, means[c_num]))

        self.summary_file = self.generate_summary_file()

    def generate_summary_file(self):
        """
        Generate object containing cluster summary trees to convert to JSON
        :return: JSON structured tree summary object
        """
        json_output = {}
        for cluster in self.clusters:
            for i, sub_cluster in enumerate(cluster.subclusters):
                name = str(cluster.name) + chr(i + 97) if (len(cluster.subclusters) > 1) else str(cluster.name)
                json_output[name] = {
                    "branching_index": cluster.bi,
                    "populations": {},
                    "llh": sub_cluster.mean_llh,
                    "linearity_index": cluster.li,
                    "structure": eval(sub_cluster.tree_structure),
                    "coclustering_index": cluster.ci
                }
                for node in range(sub_cluster.mean_node):
                    json_output[name]["populations"][str(node)] = {
                        "num_ssms": sub_cluster.mean_ssm[node],
                        "num_cnvs": sub_cluster.mean_cnv[node],
                        "cellular_prevalence": sub_cluster.mean_cp[node]
                    }
        return {"trees": json_output}

if __name__ == "__main__":
    os.chdir("../data/summ_files/summ_files")
    for file in glob.glob("*.summ.json"):
        base = os.path.basename(file)
        base = base[0:len(base) - 10]
        assigns = np.loadtxt("../../dump/" + base + ".assigns.csv")
        gmm_means = np.load("../../dump/" + base + ".means.npy")

        json_file = open(file, "r")
        tree_sums = json.load(json_file)["trees"]
        json_file.close()

        jp = Tumor(tree_sums, assigns, gmm_means)
        write_json("test.json", jp.summary_file)
        break



