# update node hash by gathering information from neighborhoods
import hashlib

class WeisfeilerLehmanMachine:
    """
    Weisfeiler Lehman feature extractor class.
    """
    def __init__(self, graph, features, iterations, use_hash, writing_style):
        """
        Initialization method which also executes feature extraction.
        :param graph: The networkx graph object.
        :param features: Feature hash table.
        :param iterations: Number of WL iterations.
        """
        self.iterations = iterations
        self.graph = graph
        self.features = features
        self.nodes = self.graph.nodes()
        self.extracted_features = [str(v) for k, v in features.items()]
        self.hash = use_hash
        self.writing_style = writing_style
        #self.length = len(Counter(self.extracted_features).values())
        self.step = 0
        self.do_recursions()

    def do_a_recursion(self):
        """
        The method does a single WL recursion.
        :return new_features: The hash table with extracted WL features.
        """
        new_features = {}
        for node in self.nodes:
            nebs = self.graph.neighbors(node)
            degs = [self.features[neb] for neb in nebs]
            features = [str(self.features[node])]+sorted([str(deg) for deg in degs])
            features = "_".join(features)
            if self.hash == True:
                hash_object = hashlib.md5(features.encode())
                hashing = hash_object.hexdigest()
                new_features[node] = hashing
            else:
                new_features[node] = features
        if self.writing_style == "paragraph":
            self.extracted_features += list(new_features.values())
        elif self.writing_style == "sentence":
            for index in range(len(self.nodes)):
                insert_pos = self.step + 1 + index * (self.step+2)
                self.extracted_features.insert(insert_pos, new_features[index])
        self.step += 1
        #self.length = len(Counter(new_features.values()).values())
        return new_features

    def do_recursions(self):
        """
        The method does a series of WL recursions.
        """
        for _ in range(self.iterations):
            self.features = self.do_a_recursion()