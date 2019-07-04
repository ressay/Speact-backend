"""
RDF Graph class
"""

class Graph(object):
    def __init__(self,triplets) -> None:
        """
        graph constructor
        :param triplets: contains 3-tuples subject predicate object as RDF triplets
        """
        super().__init__()
        self.triplets = []
        self.id_triplets = []
        self.nodes = [None]
        self.edges = [None]
        self.node_id = {}
        self.edge_id = {}
        self.init_triplets(triplets)

    def _add_node(self,node):
        """
        adds node's URI map to local integer id
        :param node (string): URI of the node
        :return:
        """
        if node not in self.node_id:
            self.node_id[node] = len(self.nodes)
            self.nodes.append(node)

    def _add_edge(self,edge):
        """
        adds edge's URI map to local integer id
        :param (str) edge: URI of the edge
        :return:
        """
        if edge not in self.edge_id:
            self.edge_id[edge] = len(self.edges)
            self.edges.append(edge)

    def add_triplet(self,s,p,o):
        """
        adds triplet to the graph
        :param (str) s: URI of the subject node
        :param (str) p: URI of the predicate edge
        :param (str) o: URI of the object node
        :return:
        """
        self._add_node(s)
        self._add_node(o)
        self._add_edge(p)
        self.id_triplets.append((self.node_id[s], self.edge_id[p], self.node_id[o]))

    def init_triplets(self,triplets):
        """
        creates a graph with list of triplets given in parameters
        :param (iterable) triplets: list of triplets
        :return:
        """

        self.triplets = list(triplets)
        self.triplets.sort()
        for s,p,o in self.triplets:
            self.add_triplet(s,p,o)
        # print(self.triplets)

    def add_all(self,triplets):
        for s,p,o in triplets:
            self.add_triplet(s,p,o)

    def get_encoded_triplets(self, triplets):
        return [self.get_encoded_triplet(t) for t in triplets]

    def get_encoded_triplet(self,t):
        """
        map triplet URIs to their ids
        :param t:
        :return: tuple of integers, id of each URI given in input
        """
        s,p,o = t
        return self.node_id[s],self.edge_id[p],self.node_id[o]

    def get_decoded_triplet(self,t):
        """
        map triplet ids to their URIs
        :param t:
        :return: tuple of integers, URI of each id given in input
        """
        s,p,o = t
        return self.nodes[s],self.edges[p],self.nodes[o]

    def get_encoded_list_nodes(self,nodes):
        """
        map triplet URIs to their ids
        :param t:
        :return: tuple of integers, id of each URI given in input
        """
        return [self.node_id[n] for n in nodes]

# TRIPLETS:
# (47, 2, 3) (rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#root_directory'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#Directory'))
# (47, 8, 48) (rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#root_directory'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#has_name'), rdflib.term.Literal('Simulation'))
# (49, 2, 3) (rdflib.term.BNode('N892ca78a28f74be1ac2637ebd37023f9'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#Directory'))
# (49, 8, 50) (rdflib.term.BNode('N892ca78a28f74be1ac2637ebd37023f9'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#has_name'), rdflib.term.Literal('downloads'))
# (47, 9, 49) (rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#root_directory'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#contains_file'), rdflib.term.BNode('N892ca78a28f74be1ac2637ebd37023f9'))
# (51, 2, 3) (rdflib.term.BNode('N6cc3e3c977e544b18620a1901ceca857'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#Directory'))
# (51, 8, 52) (rdflib.term.BNode('N6cc3e3c977e544b18620a1901ceca857'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#has_name'), rdflib.term.Literal('home'))
# (47, 9, 51) (rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#root_directory'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#contains_file'), rdflib.term.BNode('N6cc3e3c977e544b18620a1901ceca857'))
# (53, 2, 3) (rdflib.term.BNode('N010261f5527e499099b9f3e37990d9ac'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#Directory'))
# (53, 8, 54) (rdflib.term.BNode('N010261f5527e499099b9f3e37990d9ac'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#has_name'), rdflib.term.Literal('studies'))
# (47, 9, 53) (rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#root_directory'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#contains_file'), rdflib.term.BNode('N010261f5527e499099b9f3e37990d9ac'))
# (55, 2, 3) (rdflib.term.BNode('Nc071e28df013440b869c1bafb4fb9201'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#Directory'))
# (55, 8, 56) (rdflib.term.BNode('Nc071e28df013440b869c1bafb4fb9201'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#has_name'), rdflib.term.Literal('Second'))
# (53, 9, 55) (rdflib.term.BNode('N010261f5527e499099b9f3e37990d9ac'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#contains_file'), rdflib.term.BNode('Nc071e28df013440b869c1bafb4fb9201'))
# (57, 2, 3) (rdflib.term.BNode('Ne88ceaf4b785400e871539663a5a1449'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#Directory'))
# (57, 8, 58) (rdflib.term.BNode('Ne88ceaf4b785400e871539663a5a1449'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#has_name'), rdflib.term.Literal('First'))
# (53, 9, 57) (rdflib.term.BNode('N010261f5527e499099b9f3e37990d9ac'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#contains_file'), rdflib.term.BNode('Ne88ceaf4b785400e871539663a5a1449'))
# (59, 2, 3) (rdflib.term.BNode('N662245e128334b0f8fe8861cc1813840'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#Directory'))
# (59, 8, 60) (rdflib.term.BNode('N662245e128334b0f8fe8861cc1813840'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#has_name'), rdflib.term.Literal('work'))
# (47, 9, 59) (rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#root_directory'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#contains_file'), rdflib.term.BNode('N662245e128334b0f8fe8861cc1813840'))
# (61, 2, 3) (rdflib.term.BNode('N9322ca5dc9384bc6bb68bdc2ce141113'), rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#Directory'))
# (61, 8, 62) (rdflib.term.BNode('N9322ca5dc9384bc6bb68bdc2ce141113'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#has_name'), rdflib.term.Literal('Dialogue Manager'))
# (59, 9, 61) (rdflib.term.BNode('N662245e128334b0f8fe8861cc1813840'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#contains_file'), rdflib.term.BNode('N9322ca5dc9384bc6bb68bdc2ce141113'))
# (21, 10, 57) (rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#User'), rdflib.term.URIRef('http://www.semanticweb.org/ressay/ontologies/2019/2/untitled-ontology-7#Change_directory'), rdflib.term.BNode('Ne88ceaf4b785400e871539663a5a1449'))


