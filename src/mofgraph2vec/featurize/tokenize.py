from collections import Counter

class WeisfeilerLehmanMachine:
    """ Weisfeiler Lehman extractor class. """
    def __init__(self, graph, features, iterations):
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = list(Counter([str(v) for k, v in features.items()]).keys())
        self.step = 0
        self.do_recursions()

    def do_a_recursion(self):
        """ The method does a single WL recursion. """
        def get_new_features(feature_set, node):
            nebs = self.graph.neighbors(node)
            degs = [self.features[int(neb)] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            feature_set[node] = features
            return feature_set
        
        feature_storage = {}
        new_features = {}
        for node in self.nodes:
            feature_storage = get_new_features(feature_storage, node)
            new_features = get_new_features(new_features, node)
        new_subgraph = list(Counter(list(new_features.values())))
        self.extracted_features += new_subgraph

        self.step += 1
        return feature_storage

    def do_recursions(self):
        """ The method does a series of WL recursions. """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()