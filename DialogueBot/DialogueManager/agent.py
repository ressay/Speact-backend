import random, copy
import re
import numpy as np
from keras.preprocessing.sequence import pad_sequences
import math

from DialogueManager.FileBrowserDM.errors import DialogueError
from DialogueManager.state_tracker import StateTracker
from keras.layers import Input, GRU, CuDNNGRU, Dense, Concatenate, TimeDistributed, RepeatVector, Lambda, Masking, \
    Conv1D, Flatten, Reshape, MaxPooling1D, LSTMCell
from keras.models import Model, load_model
import Ontologies.onto_fbrowser as fbrowser
import keras.backend as K


class Agent(object):
    def __init__(self, state_size, constants, train_by_batch=True,
                 use_graph_encoder=False, compress_state=False, one_hot=True, data=None) -> None:
        super().__init__()

        self.C = constants['agent']
        self.memory = []
        self.memory_index = 0
        self.memory_map = {}
        self.pair_count = {}
        self.memory_pairs = []
        self.index_map = []
        self.max_memory_size = self.C['max_mem_size']
        self.eps = self.C['epsilon_init']
        self.gamma = self.C['gamma']
        self.vanilla = self.C['vanilla']
        self.batch_size = self.C['batch_size']
        self.train_by_batch = train_by_batch
        self.samples_trained = 0
        self.avg_triplets_sample = 0
        self.use_graph_encoder = use_graph_encoder
        self.compress_state = compress_state
        self.maskv = -5
        self.one_hot = one_hot

        self.load_weights_file_path = self.C['load_weights_file_path']
        self.save_weights_file_path = self.C['save_weights_file_path']

        if self.max_memory_size < self.batch_size:
            raise ValueError('Max memory size must be at least as great as batch size!')

        self.state_size = state_size
        self.encoder_size = 200
        self.episodes_triplets = []
        # self.possible_actions = self.C['agent_actions']
        # self.num_actions = len(self.possible_actions)

        self.state_tracker = self.init_state_tracker(data)
        self.encoder_state_size = self.state_tracker.get_state_size()
        self.tar_model = self._build_model("_tar")
        self.beh_model = self._build_model("_beh")
        self._load_weights()
        self.copy()
        if not self.use_graph_encoder:
            self.get_state_output = self._build_state_model(self.beh_model)
            self.get_state_and_action = self._built_state_action_model(self.beh_model)

        self.graph_encoding = np.zeros(self.state_size)
        self.zeros = np.zeros(self.state_size)
        self.current_triplets_vectors = np.array([])
        self.current_actions_vector = np.array([])
        self.current_possible_actions = []

        def do_nothing(state):
            return state

        if not self.compress_state or self.use_graph_encoder:
            self.transform = do_nothing
        else:
            self.transform = self.uncompress_state

    def _build_state_model(self, model, name_pre="_beh"):
        encoder_inputs = model.get_layer('encoder_inputs' + name_pre)
        encoder_state_input = model.get_layer('encoder_state_input' + name_pre)
        gru_layer = model.get_layer('gru_layer' + name_pre)
        input1 = encoder_inputs.input
        input2 = encoder_state_input.input
        output = gru_layer.output[1]
        return K.function([input1, input2], [output])

    def _built_state_action_model(self, model, name_pre="_beh"):
        encoder_inputs = model.get_layer('encoder_inputs' + name_pre)
        encoder_state_input = model.get_layer('encoder_state_input' + name_pre)
        dqn_inputs = model.get_layer('dqn_inputs' + name_pre)
        gru_layer = model.get_layer('gru_layer' + name_pre)
        output_layer = model.get_layer('distributed' + name_pre)
        input1 = encoder_inputs.input
        input2 = encoder_state_input.input
        input3 = dqn_inputs.input
        output1 = gru_layer.output[1]
        output2 = output_layer.output
        return K.function([input1, input2, input3], [output1, output2])

    def _build_model(self, name_pre=''):
        triplet_size = self.state_tracker.triplet_size
        action_size = self.state_tracker.get_action_size()
        hidden_state = self.state_size
        output_dim = 2
        encoder_inputs = Input(shape=(None, triplet_size), name='encoder_inputs' + name_pre)
        encoder_state_input = Input(shape=(hidden_state,), name='encoder_state_input' + name_pre)
        # DQN_input = Input(shape=(triplet_size,))
        DQN_inputs = Input(shape=(None, action_size), name='dqn_inputs' + name_pre)
        _, encoder_state = GRU(hidden_state,
                               return_state=True,
                               return_sequences=False,
                               reset_after=True,
                               name='gru_layer' + name_pre)(encoder_inputs, initial_state=encoder_state_input)

        def DQN_unit(layers, hidden):
            DQN_input = Input(shape=(action_size + hidden,))
            output = Dense(layers[0], activation='relu')(DQN_input)
            # output = Reshape((layers[0], 1))(output)
            # output = Conv1D(16, 32, activation='relu')(output)
            # output = Conv1D(16, 16, activation='relu')(output)
            # output = MaxPooling1D(2)(output)
            # output = Conv1D(32, 8, activation='relu')(output)
            # output = Conv1D(32, 8, activation='relu')(output)
            # output = MaxPooling1D(2)(output)
            # output = Flatten()(output)
            for layer in layers[1:]:
                output = Dense(layer, activation='relu')(output)
            output = Dense(output_dim, activation='linear', name='output_layer')(output)
            model = Model(DQN_input, output, name='DQN_unit' + name_pre)
            # model.summary()
            return model

        def repeat_vector(args):
            layer_to_repeat = args[0]
            sequence_layer = args[1]
            return RepeatVector(K.shape(sequence_layer)[1])(layer_to_repeat)

        encoded_state_input = Input((self.encoder_state_size,), name='encoded_state_input' + name_pre)
        if not self.use_graph_encoder:
            state_repeated = Lambda(repeat_vector,
                                    output_shape=(None, hidden_state))([encoder_state, DQN_inputs])
            h = hidden_state
        # mask = Masking(mask_value=self.maskv)(DQN_inputs)
        else:
            state_repeated = Lambda(repeat_vector,
                                    output_shape=(None, self.encoder_state_size))([encoded_state_input, DQN_inputs])
            h = self.encoder_state_size
        concat = Concatenate()([state_repeated, DQN_inputs])

        outputs = TimeDistributed(DQN_unit([hidden_state * 3, hidden_state * 2, hidden_state], h),
                                  name='distributed' + name_pre)(
            concat)
        if self.use_graph_encoder:
            model = Model([encoded_state_input, DQN_inputs], outputs)
        else:
            model = Model([encoder_inputs, encoder_state_input, DQN_inputs], outputs)
        model.compile('sgd', loss='mse')
        # model.summary()
        return model

    def reset(self, user_action, data):
        """
        resets the agent with start user action
        :param user_action:
        :return:
        """
        self.graph_encoding = np.zeros(self.state_size)
        self.reinit_state_tracker(data)
        self.update_state_user_action(user_action)

    def reset_data(self, data):
        self.graph_encoding = np.zeros(self.state_size)
        self.reinit_state_tracker(data)

    def _get_state_compressed(self):
        return [self.state_tracker.all_episode_triplets.copy(), self.current_actions_vector.copy()]

    def uncompress_state(self, state):
        triplets, actions = state
        return [self.state_tracker.transform_triplets_rdf_to_encoding(triplets, True), self.zeros, actions]

    def get_state(self):
        """
        creates state for model prediction contains graph encoding, new triplets recently added to graph
        and current possible actions
        :return:
        """
        if self.compress_state and not self.use_graph_encoder:
            return self._get_state_compressed()
        if self.use_graph_encoder:
            return [self.state_tracker.get_encoded_state(), self.current_actions_vector.copy()]
        new_triplets = self.state_tracker.get_new_triplets()
        states = [new_triplets, self.graph_encoding.copy(), self.current_actions_vector.copy()]
        return states

    def update_state_user_action(self, user_action):
        """
        updates the state tracker: the knowledge graph, hence new action possibilities
        :param user_action:
        :return:
        """

        self.state_tracker.update_state_user_action(user_action, update_encoding=self.use_graph_encoder)
        self.current_actions_vector, self.current_possible_actions = self.state_tracker.get_possible_actions()

    def step_user_action(self, user_action):
        self.update_state_user_action(user_action)
        return self.step()

    def step(self):
        """
        given current state, which action the model will take
        :return:
        """
        states = self.get_state()
        # action_index, action = self.get_action(states)
        # self.graph_encoding = self._predict_state(states)
        if not self.use_graph_encoder:
            self.graph_encoding, action_index, action = self._predict_state_action(states)
        else:
            action_index, action = self.get_action(states)
            self.graph_encoding = self.state_tracker.get_encoded_state()
        try:
            self.state_tracker.update_state_agent_action(action, update_encoding=self.use_graph_encoder)
        except DialogueError as e:
            return action_index, {'intent': 'inform', 'error': e, 'required_reward': e.reward}
        return action_index, action

    def get_action(self, states=None):
        """
        Returns the action of the agent given a state.

        Gets the action of the agent given the current state. Either the rule-based policy or the neural networks are
        used to respond.

        Parameters:
            state (numpy.array): The database with format dict(long: dict)

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself

        """
        if states is None:
            states = self.get_state()
        if self.eps > random.random():
            index = random.randint(0, len(self.current_possible_actions) - 1)
            action = self.current_possible_actions[index]
            return index, action
        else:
            return self._dqn_action(states)

    def _dqn_action(self, states):
        """
        Returns a behavior model output given a state.

        Parameters:
            state (numpy.array)

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself
        """
        result = self._dqn_predict_one(states)
        index = np.argmax(result)
        action = self.current_possible_actions[index]
        return index, action

    def _dqn_predict_one(self, states, target=False):
        """
        Returns a model prediction given a state.

        Parameters:
            state (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """

        return self._dqn_predict_action(states, target=target).flatten()

    def _dqn_predict_action(self, states, target=False, one=True):
        """
        Returns a model prediction given an array of states.

        Parameters:
            states (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """
        if one:
            state = [np.array([s]) for s in states]
        else:
            state = [pad_sequences(np.array([s[S] for s in states]))
                     for S in range(len(states[0]))]
        if target:
            result = self.tar_model.predict(state)
        else:
            result = self.beh_model.predict(state)

        # flatten each sample of the batch
        result = np.array([sample.flatten() for sample in result])
        return result

    def _predict_state(self, states, target=False):
        """
        Returns a model prediction given an array of states.

        Parameters:
            states (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """
        new_triplets, state, possible_actions = [np.array([s]) for s in self.transform(states)]
        # model = self.beh_model if not target else self.tar_model
        # if target:
        #     return self.tar_model.predict([new_triplets,state,possible_actions])[1].flatten()
        # else:
        #     return self.beh_model.predict([new_triplets,state,possible_actions])[1].flatten()

        return self.get_state_output([new_triplets, state])[0][0]

    def _predict_state_action(self, states, random_gen=True):
        new_triplets, state, possible_actions = [np.array([s]) for s in self.transform(states)]
        new_state, action = self.get_state_and_action([new_triplets, state, possible_actions])
        new_state = new_state[0]
        if random_gen and self.eps > random.random():
            index = random.randint(0, len(self.current_possible_actions) - 1)
            # action = self.current_possible_actions[index]
        else:
            action = action[0].flatten()
            index = np.argmax(action)
        action = self.current_possible_actions[index]
        return new_state, index, action

    def add_experience(self, state, action, reward, next_state, done):
        """
        Adds an experience tuple made of the parameters to the memory.

        Parameters:
            state (list of numpy.array)
            action (int)
            reward (int)
            next_state (list of numpy.array)
            done (bool)

        """

        if not self.compress_state and not self.use_graph_encoder:
            tr, st, ac = state
            pair = len(tr), len(ac)
        elif not self.use_graph_encoder:
            tr, ac = state
            pair = len(tr)  # , len(ac)
        else:
            st, ac = state
            pair = len(ac)

        if pair not in self.memory_map:
            self.memory_map[pair] = {}
            self.pair_count[pair] = 0
            self.memory_pairs.append(pair)
        ind = self.pair_count[pair]
        self.pair_count[pair] += 1
        self.memory_map[pair][ind] = (state, action, reward, next_state, done)

        if len(self.memory) < self.max_memory_size:
            self.memory.append(None)
            self.index_map.append(None)
        else:
            p, i = self.index_map[self.memory_index]
            del self.memory_map[p][i]
            if len(self.memory_map[p]) == 0:
                del self.memory_map[p]
                self.memory_pairs.remove(p)
        self.index_map[self.memory_index] = pair, ind
        self.memory[self.memory_index] = (state, action, reward, next_state, done)
        self.memory_index = (self.memory_index + 1) % self.max_memory_size

    def empty_memory(self):
        """Empties the memory and resets the memory index."""
        self.memory = []
        self.memory_index = 0
        self.memory_map = {}
        self.pair_count = {}
        self.memory_pairs = []
        self.index_map = []

    def is_memory_full(self):
        """Returns true if the memory is full."""

        return len(self.memory) == self.max_memory_size

    def training_generator(self):
        self.samples_trained = 0
        while True:
            batch = random.sample(self.memory, 1)[0]

            states = batch[0]
            next_states = batch[3]

            # assert states.shape == (self.batch_size, self.state_size), 'States Shape: {}'.format(states.shape)
            # assert next_states.shape == states.shape

            beh_state_preds = self._dqn_predict_action(states)  # For leveling error
            if not self.vanilla:
                beh_next_states_preds = self._dqn_predict_action(next_states)  # For indexing for DDQN
            tar_next_state_preds = self._dqn_predict_action(next_states,
                                                            target=True)  # For target value for DQN (& DDQN)

            # inputs = np.zeros((self.batch_size, self.state_size))
            # targets = np.zeros((self.batch_size, self.num_actions))

            # for i, (s, a, r, s_, d) in enumerate(batch):
            (s, a, r, s_, d) = batch
            t = beh_state_preds[0]
            if not self.vanilla:
                t[a] = r + self.gamma * tar_next_state_preds[0][np.argmax(beh_next_states_preds[0])] * (not d)
            else:
                t[a] = r + self.gamma * np.amax(tar_next_state_preds[0]) * (not d)

            tr, st, ac = s
            input_tr = np.array([tr])
            input_st = np.array([st])
            # target_st = np.array([s_[0]])
            input_ac = np.array([ac])
            targets = np.array([np.array([np.array((t[i], t[i + 1])) for i in range(0, len(t), 2)])])
            self.samples_trained += len(targets)
            # print(input_tr.shape,input_st.shape,input_ac.shape)
            yield [input_tr, input_st, input_ac], targets

    def training_generator_by_batch(self, with_padding=False):
        self.samples_trained = 0

        def do_nothing(state):
            return state

        if not self.compress_state or self.use_graph_encoder:
            transform = do_nothing
        else:
            transform = self.uncompress_state
        while True:
            if with_padding:
                memory = self.memory
            else:
                memory = self.memory_map[random.choice(self.memory_pairs)]
            batch_size = self.batch_size if len(memory) >= self.batch_size else len(memory)
            batch = random.sample(list(memory.values()), batch_size)
            # if len(batch) != self.batch_size:
            #     print('problem in len batch: ', len(batch))

            states = [transform(sample[0]) for sample in batch]
            next_states = [transform(sample[3]) for sample in batch]

            # assert states.shape == (self.batch_size, self.state_size), 'States Shape: {}'.format(states.shape)
            # assert next_states.shape == states.shape

            beh_state_preds = self._dqn_predict_action(states, one=False)  # For leveling error
            if not self.vanilla:
                beh_next_states_preds = self._dqn_predict_action(next_states, one=False)  # For indexing for DDQN
            tar_next_state_preds = self._dqn_predict_action(next_states, target=True,
                                                            one=False)  # For target value for DQN (& DDQN)

            # inputs = np.zeros((self.batch_size, self.state_size))
            # targets = np.zeros((self.batch_size, self.num_actions))

            # for i, (s, a, r, s_, d) in enumerate(batch):
            input_tr = []
            input_st = []
            input_ac = []
            targets = []

            for i, sample in enumerate(batch):
                (s, a, r, s_, d) = sample
                t = beh_state_preds[i]
                if not self.vanilla:
                    t[a] = r + self.gamma * tar_next_state_preds[i][np.argmax(beh_next_states_preds[i])] * (not d)
                else:
                    t[a] = r + self.gamma * np.amax(tar_next_state_preds[i]) * (not d)
                if not self.use_graph_encoder:
                    tr, st, ac = transform(s)
                    input_tr.append(tr)
                else:
                    st, ac = transform(s)
                index = a if a % 2 == 0 else a - 1
                input_st.append(st)
                input_ac.append(np.array([ac[int(index / 2)]]))
                targets.append(np.array([np.array((t[index], t[index + 1]))]))
            # print(input_tr.shape,input_st.shape,input_ac.shape)
            input_st, input_ac, targets = [np.array(a)
                                           for a in [input_st, input_ac, targets]]
            if with_padding:
                input_tr = pad_sequences(input_tr)
                input_ac = pad_sequences(input_ac)
                targets = pad_sequences(targets)
            elif not self.use_graph_encoder:
                input_tr = np.array(input_tr)
                # input_ac = pad_sequences(input_ac, value=self.maskv)
                # targets = pad_sequences(targets, value=self.maskv)
            self.samples_trained += len(targets)

            if not self.use_graph_encoder:
                self.avg_triplets_sample += len(input_tr[0]) * len(targets)
                yield [input_tr, input_st, input_ac], targets
            else:
                yield [input_st, input_ac], targets

    def train(self):
        """
        Trains the agent by improving the behavior model given the memory tuples.

        Takes batches of memories from the memory pool and processing them. The processing takes the tuples and stacks
        them in the correct format for the neural network and calculates the Bellman equation for Q-Learning.

        """

        def mean(numbers):
            return float(sum(numbers)) / max(len(numbers), 1)

        # K.clear_session()
        # Calc. num of batches to run
        if self.train_by_batch:
            av = mean([len(v) for v in self.memory_map.values()])
            bs = self.batch_size if self.batch_size < av else av
            num_batches = len(self.memory) // bs
            num_batches *= 1.3
            train_gen = self.training_generator_by_batch
        else:
            num_batches = len(self.memory)
            train_gen = self.training_generator
        # self.beh_model._make_predict_function()
        if not self.use_graph_encoder:
            del self.get_state_output
            del self.get_state_and_action
        self.beh_model._make_predict_function()
        self.tar_model._make_predict_function()
        self.avg_triplets_sample = 0
        self.beh_model.fit_generator(train_gen(), epochs=1,
                                     verbose=1, steps_per_epoch=int(num_batches))
        self.avg_triplets_sample /= self.samples_trained
        if not self.use_graph_encoder:
            self.get_state_output = self._build_state_model(self.beh_model)
            self.get_state_and_action = self._built_state_action_model(self.beh_model)
        # K.clear_session()
        print('finished fitting on ', self.samples_trained, ' samples and avg triplet number: ',
              self.avg_triplets_sample)

    def copy(self):
        """Copies the behavior model's weights into the target model's weights."""

        self.tar_model.set_weights(self.beh_model.get_weights())

    def save_weights(self):
        """Saves the weights of both models in two h5 files."""

        if not self.save_weights_file_path:
            return
        beh_save_file_path = re.sub(r'\.h5', r'_beh.h5', self.save_weights_file_path)
        self.beh_model.save(beh_save_file_path)
        # tar_save_file_path = re.sub(r'\.h5', r'_tar.h5', self.save_weights_file_path)
        # self.tar_model.save_weights(tar_save_file_path)

    def _load_weights(self):
        """Loads the weights of both models from two h5 files."""

        if not self.load_weights_file_path:
            return
        beh_load_file_path = re.sub(r'\.h5', r'_beh.h5', self.load_weights_file_path)
        # self.beh_model = load_model(beh_load_file_path)
        self.beh_model.load_weights(beh_load_file_path)
        # tar_load_file_path = re.sub(r'\.h5', r'_tar.h5', self.load_weights_file_path)
        # self.tar_model.load_weights(tar_load_file_path)

    def init_state_tracker(self, data):
        state_tracker = StateTracker(self.encoder_size, fbrowser.graph, one_hot=self.one_hot, data=data)
        return state_tracker

    def reinit_state_tracker(self, data):
        self.state_tracker.reset(self.encoder_size, fbrowser.graph, one_hot=self.one_hot, data=data)


if __name__ == '__main__':
    g1 = list(fbrowser.graph.triples((None, None, None)))
