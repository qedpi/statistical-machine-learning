from collections import defaultdict

from graphviz import Digraph
import numpy as np


class BayesNet:
    def __init__(self, rvs):
        self.k = len(rvs)
        self.rvs = rvs
        self.G = None
        self.Grev = None

    def add_edges(self, edges):
        self.G = edges

        Grev = defaultdict(list)
        for vi, vfs in edges.items():
            for vf in vfs:
                Grev[vf].append(vi)
        self.Grev = Grev

        print(f"graph: {self.G}\nreverse graph: {self.Grev}")

    def learn_params(self, data):
        assert data.shape[-1] == self.k, f"#cols = {data.shape[-1]} differs from #rvs = {self.k}"

    def show_graph(self):
        dot = Digraph()
        for v in self.rvs:
            dot.node(str(v), str(v))
        edges = []
        # for vi, vfs
        dot.edges(f"{vi}{vf}" for vi, vf in self.G)

        print(dot.source)
        file_name = "myplot"
        dot.render(file_name, view=True)


# generate binary RVs: 100 samples of 4 binary rv
data = np.random.randint(2, size=(100, 4))
print(data)

rvs = ['a', 'b', 'c', 'd']
# rvs = [1, 2]
# edges = {('a', 'b'), ('a', 'c'), ('b', 'd'), ('c', 'd')}
edges = {0: (1, 2), 1: (3,), 2: (3, )}

BN = BayesNet(rvs)
BN.add_edges(edges)
BN.learn_params(data)
# BN.show_graph()