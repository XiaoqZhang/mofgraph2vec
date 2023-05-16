# update node hash by gathering information from neighborhoods
import hashlib
from collections import Counter
from loguru import logger

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, nodes_idx, linker_idx, iterations, use_hash, writing_style, mode):
        """
        Initialization method which also executes feature extraction.
        :param graph: The networkx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes_idx = nodes_idx
        self.linker_idx = linker_idx
        self.nodes = self.graph.nodes()
        self.extracted_features = [] #list(Counter([str(v) for k, v in features.items()]).keys())
        self.hash = use_hash
        self.writing_style = writing_style
        self.mode = mode
        #self.length = len(Counter(self.extracted_features).values())
        self.step = 0
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        def get_new_features(feature_set, node):
            nebs = self.graph.neighbors(node)
            degs = [self.features[int(neb)] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            if self.hash == True:
                hash_object = hashlib.md5(features.encode())
                hashing = hash_object.hexdigest()
                feature_set[node] = hashing
            else:
                feature_set[node] = features
            return feature_set
        
        feature_storage = {}
        new_features = {}
        for node in self.nodes:
            feature_storage = get_new_features(feature_storage, node)
            if self.mode == "all":
                new_features = get_new_features(new_features, node)
            if self.mode == "scaffold":
                if (node in self.nodes_idx) or (node in self.linker_idx):
                    new_features = get_new_features(new_features, node)
            if self.mode == "node":
                if node in self.nodes_idx:
                    new_features = get_new_features(new_features, node)
            if self.mode == "linker":
                if node in self.linker_idx:
                    new_features = get_new_features(new_features, node)
        
        if self.writing_style == "paragraph":
            new_subgraph = list(Counter(list(new_features.values())))
            self.extracted_features += new_subgraph
        elif self.writing_style == "sentence":
            for index in range(len(self.nodes)):
                insert_pos = self.step + 1 + index * (self.step+2)
                self.extracted_features.insert(insert_pos, new_features[index])
        self.step += 1
        #self.length = len(Counter(new_features.values()).values())
        return feature_storage

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()