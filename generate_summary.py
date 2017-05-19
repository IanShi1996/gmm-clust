import json
import numpy as np
import os
import glob


class Cluster:
    def __init__(self, num, trees, mean):
        self.num = num
        self.trees = trees
        self.li = mean[0]
        self.bi = mean[1]
        self.ci = 1 - (self.li + self.bi)

        self.mean_node = int(self.calc_mean_node())
        self.tree_structure = self.calc_tree_structures()
        self.mean_cp = self.calc_mean_node_cp()
        self.mean_ssm = self.calc_mean_node_ssm()
        self.mean_cnv = self.calc_mean_node_cnv()
        self.mean_llh = self.calc_mean_llh()


    def calc_mean_node(self):
        mean = sum(map(lambda x: len(self.trees[x]["populations"]), self.trees))
        return mean / len(self.trees)

    def calc_tree_structures(self):
        structures = [self.trees[next(iter(self.trees))]["structure"]]
        for tree in self.trees:
            if self.trees[tree]["structure"] not in structures:
                structures.append(self.trees[tree]["structure"])
        if len(structures) > 1:
            print("More than 1 structure")
        return structures

    def calc_mean_node_cp(self):
        cp_means = {i: 0 for i in range(self.mean_node)}
        for j in range(self.mean_node):
            for tree in self.trees:
                try:
                    cp_means[j] += self.trees[tree]["populations"][str(j)]["cellular_prevalence"][0]
                except KeyError:
                    pass
            cp_means[j] = cp_means[j] / len(self.trees)
        return cp_means

    def calc_mean_node_ssm(self):
        ssm_means = {i: 0 for i in range(self.mean_node)}
        for j in range(self.mean_node):
            for tree in self.trees:
                try:
                    ssm_means[j] += self.trees[tree]["populations"][str(j)]["num_ssms"]
                except KeyError:
                    pass
            ssm_means[j] = ssm_means[j] / len(self.trees)
        return ssm_means

    def calc_mean_node_cnv(self):
        cnv_means = {i: 0 for i in range(self.mean_node)}
        for j in range(self.mean_node):
            for tree in self.trees:
                try:
                    cnv_means[j] += self.trees[tree]["populations"][str(j)]["num_cnvs"]
                except KeyError:
                    pass
            cnv_means[j] = cnv_means[j] / len(self.trees)
        return cnv_means

    def calc_mean_llh(self):
        mean_llh = 0
        for tree in self.trees:
            llh = self.trees[tree]["llh"]
            '''
            num_ssm = 0
            for j in range(self.mean_node):
                num_ssm += self.trees[tree]["populations"][str(j)]["num_ssms"]
            llh = llh / num_ssm
            llh = llh / math.log(2)
            mean_llh += -llh
            '''
            mean_llh += llh
        return mean_llh / len(self.trees)


class Tumor:
    def __init__(self, file_name, assignments, means):
        self.file_name = os.path.basename(file_name)
        self.file_name = self.file_name[0:len(self.file_name) - 10]

        f = open(file_name, "r")
        self.trees = json.load(f)["trees"]
        f.close()

        self.assignments = assignments
        self.num_clusters = len(set(self.assignments))
        self.clusters = []

        for c_num in range(self.num_clusters):
            # Get all trees with cluster num c_num
            try:
                trees = {k: v for k, v in self.trees.items() if assignments[int(k)] == c_num}
                self.clusters.append(Cluster(c_num, trees, means[c_num]))
            except:
                pass

        self.summ_file = self.generate_summary_file()
        self.write_json()

    def generate_summary_file(self):
        j = {}
        for c in self.clusters:
            n = str(c.num)
            j[n] = {}
            j[n]["branching_index"] = c.bi
            j[n]["populations"] = {}
            for node in range(c.mean_node):
                j[n]["populations"][str(node)] = {}
                j[n]["populations"][str(node)]["num_ssms"] = c.mean_ssm[node]
                j[n]["populations"][str(node)]["num_cnvs"] = c.mean_cnv[node]
                j[n]["populations"][str(node)]["cellular_prevalence"] = [c.mean_cp[node]]
            j[n]["llh"] = c.mean_llh
            j[n]["linearity_index"] = c.li
            j[n]["structure"] = c.tree_structure[0]
            j[n]["coclustering_index"] = c.ci
        out = {"trees": j}
        return out

    def write_json(self):
        os.chdir("result")
        f = open(self.file_name + "_clust.summ.json", "w")
        json.dump(self.summ_file, f)

    def get_tree(self, num):
        return self.trees[str(num)]

    def print_all_num_nodes(self):
        for cluster in self.clusters:
            print("Cluster {}: {} nodes".format(cluster.num, cluster.mean_node))

    def print_all_structure(self):
        for cluster in self.clusters:
            print("Cluster {}: {}".format(cluster.num, cluster.tree_structure))

    def print_all_mean_node_cp(self):
        for cluster in self.clusters:
            print("Cluster {}: {}".format(cluster.num, cluster.mean_cp))

    def print_all_mean_node_ssm(self):
        for cluster in self.clusters:
            print("Cluster {}: {}".format(cluster.num, cluster.mean_ssm))

    def print_all_mean_node_cnv(self):
        for cluster in self.clusters:
            print("Cluster {}: {}".format(cluster.num, cluster.mean_cnv))

    def print_all_mean_llh(self):
        for cluster in self.clusters:
            print("Cluster {}: {}".format(cluster.num, cluster.mean_llh))

if __name__ == "__main__":
    os.chdir("data/summ_files/summ_files")
    for file in glob.glob("*.summ.json"):
        a = np.loadtxt("../../gmm_output/" + file[0: len(file) - 10] + ".assigns.csv")
        c = np.load("../../gmm_output/" + file[0: len(file) - 10] + ".means.npy")

        jp = Tumor(file, a, c)
        os.chdir("..")



