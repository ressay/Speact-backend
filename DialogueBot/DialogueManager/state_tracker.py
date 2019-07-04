import math

import numpy as np
import rdflib
from keras import Model
import Ontologies.onto_fbrowser as fbrowser
from Ontologies import graph
from keras.models import load_model
from keras.layers import Input


def to_int_onehot(arr):
    return np.argmax(arr)


def to_int_binary(bitlist):
    out = 0
    for bit in bitlist:
        bit = int(round(bit))
        out = (out << 1) | bit
    return out


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit numpy array"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def onehot_array(num, m):
    """Convert a positive integer num into one_hot numpy array"""
    arr = np.zeros(m)
    arr[num] = 1
    return arr


class StateTracker(object):
    encoder_size = 20
    node_size = 9
    edge_size = 7
    triplet_size = 2 * node_size + edge_size
    hidden_state = 1024

    def __init__(self, size, ontology, one_hot=True, lazy_encoding=True, data= None) -> None:
        """
        StateTracker constructor
        :param (int) size:
        :param (rdflib.Graph) ontology:
        """
        super().__init__()
        self.cursor = 0
        self.encoder = None
        self.encoder_type = one_hot
        self.encoder = self.load_encoder(one_hot)
        size = int(math.ceil(size / self.encoder_size))
        self.vectors = [np.zeros(self.hidden_state) for i in range(size)]
        self.ontology = rdflib.Graph()
        self.ontology += ontology
        self.graph = graph.Graph(ontology.triples((None, None, None)))
        self.recent_user_triplets = []
        self.recent_agent_triplets = []
        self.all_episode_triplets = []
        self.lazy_encoding = lazy_encoding
        self.state_map = {
            'ontology': self.ontology,
            'graph': self.graph,
            'recent_triplets': self.recent_user_triplets,
            'recent_agent_triplets': self.recent_agent_triplets,
            'last_user_action': None,
            'last_agent_action': None
        }
        self.triplets_to_transform = []

    def reset(self, size, ontology, one_hot=True, lazy_encoding=True, data=None):
        self.cursor = 0
        # encoder = self.load_encoder(one_hot)
        # if encoder is not None:
        #     self.encoder = encoder
        self.encoder_type = one_hot
        size = int(math.ceil(size / self.encoder_size))
        self.vectors = [np.zeros(self.hidden_state) for i in range(size)]
        self.ontology = rdflib.Graph()
        self.ontology += ontology
        self.graph = graph.Graph(ontology.triples((None, None, None)))
        self.recent_user_triplets = []
        self.recent_agent_triplets = []
        self.all_episode_triplets = []
        self.lazy_encoding = lazy_encoding
        self.state_map = {
            'ontology': self.ontology,
            'graph': self.graph,
            'recent_triplets': self.recent_user_triplets,
            'recent_agent_triplets': self.recent_agent_triplets,
            'last_user_action': None,
            'last_agent_action': None
        }
        self.triplets_to_transform = []

    def get_possible_actions(self):
        """
        gets the possible action given current state
        method to be redefined by children state trackers
        :return (np.array,list) : a numpy array that contains action's triples encodings
        and a list of tuples that contains action dict and its "ask" method
        """
        return np.array([]), []

    def get_action_size(self):
        return 1

    def get_state_size(self):
        return self.hidden_state * len(self.vectors)

    def rename_layers(self, model, one_hot):
        if one_hot: # ONE HOT
            model.get_layer('input_4').name = 'input_encoder'
            model.get_layer('input_5').name = 'input_decoder'
            model.get_layer('gru_3').name = 'gru_encoder'
            model.get_layer('gru_4').name = 'gru_decoder'
            model.get_layer('model_3').name = 'output_unit'
        else: # BINARY
            model.get_layer('input_19').name = 'input_encoder'
            model.get_layer('input_20').name = 'input_decoder'
            model.get_layer('gru_11').name = 'gru_encoder'
            model.get_layer('gru_12').name = 'gru_decoder'
            model.get_layer('model_12').name = 'output_unit'

    def create_encoder_model(self,model, one_hot):
        self.rename_layers(model, one_hot)
        input_encoder = model.get_layer('input_encoder').input
        input_encoder_state = Input(shape=(self.hidden_state,))
        gru_encoder = model.get_layer('gru_encoder')
        decoder_output, encoder_state = gru_encoder(input_encoder, initial_state=input_encoder_state)

        return Model([input_encoder, input_encoder_state], encoder_state)

    def load_encoder(self, one_hot=True):
        if self.encoder is not None and self.encoder_type == one_hot:
            return None
        if one_hot:
            self.int_to_vec = onehot_array
            self.vec_to_int = to_int_onehot
            path = 'models/model_onehot.h5'
            self.node_size = 2 ** self.node_size
            self.edge_size = 2 ** self.edge_size
        else:
            self.int_to_vec = bin_array
            self.vec_to_int = to_int_binary
            path = 'models/model_binary.h5'
        self.triplet_size = 2 * self.node_size + self.edge_size
        try:
            model = load_model(path)
            return self.create_encoder_model(model,one_hot)
        except Exception:
            return None

    def get_state(self, encoded=True):
        self.state_map['recent_triplets'] = self.recent_user_triplets
        if encoded:
            self.state_map['encoded'] = self.get_encoded_state()
        return self.state_map

    def get_encoded_state(self):
        self.encode_triplets([], False)
        return np.concatenate(self.vectors, axis=None)

    def triplet_encoding_shape(self, number_of_triplets):
        return number_of_triplets, self.triplet_size

    def nodes_encoding_shape(self, number_of_tuples, number_of_nodes):
        return number_of_tuples, number_of_nodes * self.node_size

    def encode_triplet(self, s, p, o):
        return np.concatenate((self.int_to_vec(s, self.node_size),
                               self.int_to_vec(p, self.edge_size),
                               self.int_to_vec(o, self.node_size)))

    def encode_nodes(self, nodes):
        return np.concatenate([self.int_to_vec(n, self.node_size) for n in nodes])

    def transform_nodes_rdf_to_encoding(self, nodes):

        transformed = np.zeros(self.nodes_encoding_shape(len(nodes), len(nodes[0])))
        for i, (ns) in enumerate(nodes):
            ns = self.graph.get_encoded_list_nodes(ns)
            encoded = self.encode_nodes(ns)
            transformed[i, :] = encoded
        return transformed

    def transform_triplets_rdf_to_encoding(self, triplets, encoded_triplets=False):
        """
        transforms rdflib.Graph to encoded graph
        :param (iterable) triplets: sub graph to transform
        :return (numpy.array): encoded triplets
        """

        transformed = np.zeros(self.triplet_encoding_shape(len(triplets)))
        for i, (s, p, o) in enumerate(triplets):
            if not encoded_triplets:
                s, p, o = self.graph.get_encoded_triplet((s, p, o))
            encoded = self.encode_triplet(s, p, o)
            transformed[i, :] = encoded
        return transformed

    def add_sub_graph(self, graph):
        """
        adds sub_graph to state's knowledge graph
        :param (rdflib.Graph) graph:
        :return:
        """
        self.add_triplets(graph.triples((None, None, None)))

    def add_triplets(self, triplets):
        # for s,p,o in triplets:
        #     print(s,p,o)
        self.graph.add_all(triplets)
        self.ontology += triplets
        self.update_inner_state(triplets)

    def get_data(self):
        return {}

    def update_inner_state(self, triplets):
        """
        method to be redefined by children classes to update inner variables for fast graph manipulation if needed
        :param (list) triplets: a list of triplets to be added to the knowledge graph
        :return:
        """
        pass

    def encode_triplets(self, triplets, lazy=True):
        """
        adds triplets to state's knowledge graph
        :param lazy: if lazy triplets are not encoded but added to a wait list
        :param (list) triplets: list of triplets to add to state graph
        :return:
        """
        self.triplets_to_transform += triplets
        if len(self.triplets_to_transform) == 0:
            return
        if not lazy:
            graph = self.transform_triplets_rdf_to_encoding(self.triplets_to_transform)
            i = self.cursor
            self.vectors[i] = self.encoder.predict([np.array([graph]), np.array([self.vectors[i]])])[0]
            self.cursor = (self.cursor + 1) % len(self.vectors)
            self.triplets_to_transform = []

    def get_triplets_from_action(self, user_action):
        """

        :param (dict) user_action:
        :return (list): list of triplets from user's action
        """
        return []

    def get_triplets_from_agent_action(self, agent_action):
        """

        :param (dict) agent_action:
        :return (list): list of triplets from user's action
        """
        return []

    def add_to_all_triplets(self, triplets):
        self.all_episode_triplets += self.graph.get_encoded_triplets(triplets)

    def update_state_user_action(self, user_action, update_encoding=True):
        self.state_map['last_user_action'] = user_action
        self.recent_user_triplets = self.get_triplets_from_action(user_action)
        self.add_triplets(self.recent_user_triplets)
        self.add_to_all_triplets(self.recent_user_triplets)
        if update_encoding:
            self.encode_triplets(self.recent_user_triplets)

    def update_state_agent_action(self, agent_action, update_encoding=True):
        self.state_map['last_agent_action'] = agent_action
        self.recent_agent_triplets = self.get_triplets_from_agent_action(agent_action)
        self.add_triplets(self.recent_agent_triplets)
        self.add_to_all_triplets(self.recent_agent_triplets)
        if update_encoding:
            self.encode_triplets(self.recent_agent_triplets)
        # self.refresh_triplets(agent_action)


    def get_new_triplets(self):
        new_triplets = self.recent_agent_triplets + self.recent_user_triplets
        return self.transform_triplets_rdf_to_encoding(new_triplets)

    def get_episode_triplets(self):
        return self.transform_triplets_rdf_to_encoding(self.all_episode_triplets, encoded_triplets=True)

    def refresh_triplets(self, agent_action):
        pass


if __name__ == '__main__':
    state = StateTracker(5, fbrowser.graph)
    # print(state.transform_triplets_rdf_to_encoding(fbrowser.graph.triples()).shape)

