import rdflib
import sys
from rdflib import Literal, BNode

from DialogueManager.FileBrowserDM.errors import FileNameExistsError, RemoveCurrentDirError, MoveFileInsideItself, \
    PuttingFileUnderRegularFile, RenamingFolderInCurrentPath
from DialogueManager.FileBrowserDM.file_tree_sim import FileTreeSimulator
from DialogueManager.FileBrowserDM.intent_tracker import ActionTracker
from DialogueManager.state_tracker import StateTracker
import Ontologies.onto_fbrowser as fbrowser
import Ontologies.python_from_ontology as onto
from DialogueManager.FileBrowserDM.user_simulator import UserSimulatorFB as usim


class StateTrackerFB(StateTracker):
    default_action = {'intent': 'default', 'action_node': fbrowser.a_acted, 'file_node': fbrowser.a_acted}

    def __init__(self, size, ontology, one_hot=True, lazy_encoding=True, data=None, root_path='~') -> None:
        """
        StateTracker constructor
        :param (int) size:
        :param (rdflib.Graph) ontology:
        """
        super().__init__(size, ontology, one_hot, lazy_encoding, data)

        self.bricolage = Bricolage()
        # self.focused_file = None
        self.user_actions_map = {
            usim.Create_file_desire: self.agent_actions_desire_triplets_u,
            usim.Change_directory_desire: self.change_dir_triplets_u,
            usim.Delete_file_desire: self.agent_actions_desire_triplets_u,
            usim.Move_file_desire: self.copy_move_desire_triplets_u,
            usim.Copy_file_desire: self.copy_move_desire_triplets_u,
            usim.Open_file_desire: self.open_triplets_u,
            usim.Rename_file_desire: self.rename_triplets_u,
            usim.u_request: self.request_triplets_u,
            usim.u_inform: self.inform_triplets_u,
            usim.confirm: self.ask_triplets_u,
            usim.deny: self.ask_triplets_u,
            usim.default: self.default_u,
            usim.end: self.default_u,
            'unknown': self.default_u
        }
        self.inform_slots = {
            'file_name': fbrowser.File_name,
            'parent_directory': fbrowser.Parent_directory,
            'old_name': fbrowser.Old_name,
            'new_name': fbrowser.New_name,
            'directory': fbrowser.File_name,
            'dest': fbrowser.File_name,
            'origin': fbrowser.Parent_directory
        }
        self.children = {}
        self.parent = {}
        self.nodes_by_name = {}
        self.name_by_node = {}
        self.file_exists = set()
        self.file_type = {}
        self.special_actions = []
        self.special_nodes = {}
        self.root = None
        if root_path[-1] == '/':
            root_path = root_path[:-1]
        self.root_path = root_path
        self.current_path_node, self.current_path = None, None
        self.last_user_action = None
        if data is not None:
            self.add_known_files_to_graph(data['tree_sim'])
        else:
            self.add_known_files_to_graph()
        self.agent_actions_map = {
            "Create_file": self.create_file_triplets_a,
            "Delete_file": self.delete_file_triplets_a,
            "Change_directory": self.change_directory_triplets_a,
            "Move_file": self.move_copy_file_triplets_a,
            "Copy_file": self.move_copy_file_triplets_a,
            "Rename_file": self.rename_triplets_a,
            "Open_file": self.open_triplets_a,
            "inform": self.inform_triplets_a,
            "ask": self.ask_triplets_a,
            "request": self.request_triplets_a,
            'default': self.default_a
        }
        self.action_tracker = ActionTracker(self)

    def get_data(self):
        return {'current_tree_sim': self.create_tree_sim()}

    def reset(self, size, ontology, one_hot=True, lazy_encoding=True, data=None):
        super().reset(size, ontology, one_hot, lazy_encoding, data)
        # self.focused_file = None
        self.children = {}
        self.parent = {}
        self.nodes_by_name = {}
        self.name_by_node = {}
        self.special_actions = []
        self.special_nodes = {}
        self.file_exists = set()
        self.file_type = {}
        self.current_path_node, self.current_path = None, None
        if data is not None:
            self.add_known_files_to_graph(data['tree_sim'])
        else:
            self.add_known_files_to_graph()

    def get_possible_actions(self, encode_actions=True):
        actions = []

        def ask_action(act):
            return {'intent': 'ask', 'action': act, 'ask_nodes': act['ask_nodes']}

        if self.action_tracker.has_intent():
            actions = self.action_tracker.get_possible_actions()
        actions.append(self.default_action.copy())
        for a in actions:
            a['ask_nodes'] = (a['action_node'], a['file_node'])
        action_nodes = [(m['action_node'], m['file_node']) for m in actions]
        actions = sum([[act, ask_action(act)] for act in actions], [])
        if encode_actions:
            action_nodes = self.transform_nodes_rdf_to_encoding(action_nodes)
        else:
            action_nodes = [self.graph.get_encoded_list_nodes(a) for a in action_nodes]
        return action_nodes, actions

    def get_action_size(self):
        return self.node_size * 2

    def update_inner_state(self, triplets):
        super().update_inner_state(triplets)
        for n in self.parent:
            parent = self.parent[n]
            if parent not in self.parent and parent != self.root:
                print('error here start inner state')
        for s, p, o in triplets:
            if p == onto.rdf_type and o in (fbrowser.Directory, fbrowser.RegFile, fbrowser.File):
                self.set_file_in_inner_state(s)
                self.set_file_type(s, o)
            if p == fbrowser.contains_file:
                self.set_file_in_inner_state(o, s)
        for s, p, o in triplets:
            if p == fbrowser.has_name and s in self.parent:
                self.set_file_name(s, o)
        for n in self.parent:
            parent = self.parent[n]
            if parent not in self.parent and parent != self.root:
                print('indeed error here, thats weird')

    def remove_special_candidate(self, agent_action):
        self.special_nodes.clear()

    def add_to_special_candidate(self, node, desire, user_action):
        # if desire in (fbrowser.Copy_file, fbrowser.Move_file):
        dests = []
        if node in self.special_nodes:
            desire, dests = self.special_nodes[node]
        if 'dest' in user_action:
            dest = user_action['dest']
            dest_nodes = self.get_files_from_graph({'file_name': dest})
            if dest_nodes is not None:
                dest_nodes = [n for n in dest_nodes if self.file_type[n] == fbrowser.Directory]
                self.special_nodes[node] = (desire, list(dest_nodes))
        else:
            self.special_nodes[node] = (desire, dests)

    ############## USER ACTION TRIPLETS

    def get_triplets_from_action(self, user_action):
        """

        :param (dict) user_action:
        :return (list): list of triplets from user's action
        """
        self.last_user_action = user_action
        if 'dest' in user_action:
            self.bricolage.set_referenced_directory(user_action['dest'])
        user_action = self.bricolage.fix_references(user_action)
        return self.user_actions_map[user_action['intent']](user_action)

    def default_u(self, user_action):
        self.action_tracker.clear_current_action()
        return []

    def default_a(self, agent_action):
        return []

    # TODO OPTIONAL ADD_USER_ACTION NODES
    def request_triplets_u(self, user_action):
        triplets = []
        # when nlu does an error on intent, the slot value is generated automatically since we have just one slot
        # so for now, we enter directly if an error happens with user sim and slot is not in the dict
        if 'slot' not in user_action or user_action['slot'] == 'directory':
            self.action_tracker.set_current_action(user_action)
            triplets.append((fbrowser.User, fbrowser.has_desire, fbrowser.A_inform))
            for key in self.inform_slots:
                if key in user_action:
                    triplets.append((fbrowser.User, fbrowser.U_inform, self.inform_slots[key]))
                    triplets.append((self.inform_slots[key], fbrowser.has_parameter, Literal(user_action[key])))
            return triplets

    def rename_triplets_u(self, user_action):
        triplets = []
        self.action_tracker.set_current_action(user_action)
        triplets.append((fbrowser.User, fbrowser.has_desire, fbrowser.Rename_file))
        for key in self.inform_slots:
            if key in user_action:
                triplets.append((fbrowser.User, fbrowser.U_inform, self.inform_slots[key]))
                triplets.append((self.inform_slots[key], fbrowser.has_parameter, Literal(user_action[key])))
        return triplets

    def open_triplets_u(self, user_action):
        triplets = []
        self.action_tracker.set_current_action(user_action)
        triplets.append((fbrowser.User, fbrowser.has_desire, fbrowser.Open_file))
        for key in self.inform_slots:
            if key in user_action:
                triplets.append((fbrowser.User, fbrowser.U_inform, self.inform_slots[key]))
                triplets.append((self.inform_slots[key], fbrowser.has_parameter, Literal(user_action[key])))
        return triplets

    def change_dir_triplets_u(self, user_action):
        triplets = []
        self.action_tracker.set_current_action(user_action)
        triplets.append((fbrowser.User, fbrowser.has_desire, fbrowser.Change_directory))
        for key in self.inform_slots:
            if key in user_action:
                triplets.append((fbrowser.User, fbrowser.U_inform, self.inform_slots[key]))
                triplets.append((self.inform_slots[key], fbrowser.has_parameter, Literal(user_action[key])))
        return triplets

    def inform_triplets_u(self, user_action):

        # print('inform triplets:',user_action)
        triplets = []
        assert 'inform' == user_action['intent'], "intent is not inform inside inform_triplets method"
        prev = self.state_map['last_agent_action']
        if self.action_tracker.has_intent():
            self.action_tracker.add_inform_intent(user_action, prev)

            for key in self.inform_slots:
                if key in user_action:
                    triplets.append((fbrowser.User, fbrowser.U_inform, self.inform_slots[key]))
                    triplets.append((self.inform_slots[key], fbrowser.has_parameter, Literal(user_action[key])))
            # assert len(triplets) > 0, "no keys found in inform" + str(user_action)
            return triplets

        if 'slot' in prev and prev['slot'] == 'parent_directory':
            if 'file_name' in user_action:
                user_action['parent_directory'] = user_action['file_name']
                del user_action['file_name']
        if 'file_name' in user_action:
            # TODO fix case when inform comes after special action
            f = self.get_focused_file_node(True)

            if f in (fbrowser.Change_directory, fbrowser.Delete_file):
                file_nodes = self.get_files_from_graph(user_action)
                if file_nodes is None:
                    self.special_actions.append({'intent': 'request', 'slot': 'file_name',
                                                 'file_node': f, 'action_node': fbrowser.A_request,
                                                 'special': 'file_not_found',
                                                 'special_file_name': user_action['file_name']})
                else:
                    for file_node in file_nodes:
                        triplets.append((fbrowser.User, f, file_node))
            elif f in (fbrowser.Copy_file, fbrowser.Move_file):
                file_nodes = self.get_files_from_graph(user_action)
                if file_nodes is None:
                    self.special_actions.append({'intent': 'request', 'slot': 'file_name',
                                                 'file_node': f, 'action_node': fbrowser.A_request,
                                                 'special': 'file_not_found',
                                                 'special_file_name': user_action['file_name']})
                else:
                    for file_node in file_nodes:
                        triplets.append((fbrowser.User, f, file_node))
                        self.add_to_special_candidate(file_node, f, user_action)
            elif f is None:
                triplets.append((fbrowser.User, fbrowser.U_inform, Literal(user_action['file_name'])))
            else:
                if 'slot' in prev and prev['slot'] == 'dest':
                    user_action['dest'] = user_action['file_name']
                    self.add_to_special_candidate(f, None, user_action)
                    triplets.append((fbrowser.User, fbrowser.U_inform, Literal(user_action['file_name'])))
                else:
                    if f in self.parent:
                        triplets.append((f, fbrowser.has_name, Literal(user_action['file_name'])))

            return triplets
        if 'parent_directory' in user_action:
            t = self.get_files_from_graph({'file_name': user_action['parent_directory']})
            f = self.get_focused_file_node(True)

            if f is None:
                triplets.append((fbrowser.User, fbrowser.U_inform, Literal(user_action['parent_directory'])))
            # elif t is None:
            #     # t = self.ontology.triples((None,fbrowser.has_name,Literal(user_action['parent_directory'])))
            #     # if t is None:
            #     if f in self.parent:
            #         directory = BNode()
            #         triplets.append((directory, fbrowser.has_name, Literal(user_action['parent_directory'])))
            #         triplets.append((directory, onto.rdf_type, fbrowser.Directory))
            #         triplets.append((directory, fbrowser.contains_file, f))
            elif t is not None:
                if f in (fbrowser.Copy_file, fbrowser.Move_file):
                    new_special = dict((node, self.special_nodes[node]) for node in self.special_nodes
                                       if self.parent[node] in t)
                    self.special_nodes = new_special
                elif f in self.parent:
                    for directory in t:
                        if self.file_type[directory] == fbrowser.Directory:
                            triplets.append((directory, fbrowser.contains_file, f))
        return triplets

    def ask_triplets_u(self, user_action):
        triplets = []
        prev_aAction = self.state_map['last_agent_action']
        if prev_aAction is None:
            return []
        if prev_aAction['intent'] != 'ask':
            # print('confirming not "ask" intent', file=sys.stderr)
            return triplets
        node = prev_aAction['action_node']
        if user_action['intent'] == 'confirm':
            triplets.append((fbrowser.User, fbrowser.confirm, node))
        else:
            triplets.append((fbrowser.User, fbrowser.deny, node))
        return triplets

    def copy_move_desire_triplets_u(self, user_action):
        # self.action_tracker.clear_current_action()
        triplets = []
        self.action_tracker.set_current_action(user_action)
        desire = fbrowser.Move_file if user_action['intent'] == usim.Move_file_desire else fbrowser.Copy_file
        triplets.append((fbrowser.User, fbrowser.has_desire, desire))
        for key in self.inform_slots:
            if key in user_action:
                triplets.append((fbrowser.User, fbrowser.U_inform, self.inform_slots[key]))
                triplets.append((self.inform_slots[key], fbrowser.has_parameter, Literal(user_action[key])))
        return triplets

    def agent_actions_desire_triplets_u(self, user_action):
        triplets = []
        desires = {
            usim.Change_directory_desire: (fbrowser.Change_directory, fbrowser.Directory),
            usim.Delete_file_desire: (fbrowser.Delete_file, fbrowser.File),
            usim.Create_file_desire: (fbrowser.Create_file, None)
        }
        self.action_tracker.set_current_action(user_action)
        desire, file_type = desires[user_action['intent']]
        triplets.append((fbrowser.User, fbrowser.has_desire, desire))
        for key in self.inform_slots:
            if key in user_action:
                triplets.append((fbrowser.User, fbrowser.U_inform, self.inform_slots[key]))
                triplets.append((self.inform_slots[key], fbrowser.has_parameter, Literal(user_action[key])))
        return triplets

    ################ AGENT ACTION TRIPLETS

    def get_triplets_from_agent_action(self, agent_action):
        """

        :param (dict) agent_action:
        :return (list): list of triplets from user's action
        """
        triplets = self.agent_actions_map[agent_action['intent']](agent_action)
        if 'file_node' in agent_action:
            node = agent_action['file_node']
            if node in self.file_type:
                file_name = self.name_by_node[node]
                if agent_action['intent'] == 'Rename_file':
                    file_name = agent_action['new_name']
                if self.file_type[node] == fbrowser.Directory:
                    self.bricolage.set_referenced_directory(file_name)
                else:
                    self.bricolage.set_referenced_file(file_name)
        return triplets

    def ask_triplets_a(self, agent_action):
        triplets = []
        assert 'ask' == agent_action['intent'], "intent not ask in ask triplets agent method"
        agent_action['action_node'] = fbrowser.A_ask
        a_node, f_node = agent_action['ask_nodes']
        # triplets.append((ask_node, onto.rdf_type, fbrowser.A_ask))
        # TODO FIX ACTION_NODE
        triplets.append((a_node, fbrowser.A_ask, f_node))
        triplets.append((fbrowser.A_ask, fbrowser.has_parameter, agent_action['action']['action_node']))
        # triplets.append((fbrowser.Agent, fbrowser.a_acted, ask_node))
        return triplets

    def inform_triplets_a(self, agent_action):
        triplets = []
        triplets.append((fbrowser.Agent, fbrowser.a_acted, fbrowser.A_inform))
        return triplets

    def open_triplets_a(self, agent_action):
        triplets = []
        if self.file_type[agent_action['file_node']] == fbrowser.Directory:
            return self.change_directory_triplets_a(agent_action)
        triplets.append((fbrowser.Agent, fbrowser.a_acted, fbrowser.Open_file))
        return triplets

    def rename_triplets_a(self, agent_action):
        triplets = []
        node = agent_action['file_node']
        if self.has_ancestor(self.current_path_node, node):
            raise RenamingFolderInCurrentPath(self.name_by_node[node], self.get_path_with_real_root(node))
        triplets.append((fbrowser.Agent, fbrowser.a_acted, fbrowser.Rename_file))
        triplets.append((agent_action['file_node'], fbrowser.has_name, Literal(agent_action['new_name'])))
        return triplets

    def request_triplets_a(self, agent_action):
        triplets = []
        # req_node = BNode()
        # triplets.append((req_node, onto.rdf_type, fbrowser.A_request))
        # triplets.append((fbrowser.Agent, fbrowser.a_acted, req_node))
        # triplets.append((fbrowser.Agent, fbrowser.a_acted, fbrowser.A_request))
        return triplets

    def create_file_triplets_a(self, agent_action):
        triplets = []
        file_node = agent_action['file_node']
        t = fbrowser.RegFile if agent_action['is_file'] else fbrowser.Directory
        parent_node = agent_action['parent_node']
        # action_node = BNode()
        # triplets.append((action_node, onto.rdf_type, fbrowser.Create_file))
        triplets.append((file_node, onto.rdf_type, t))
        triplets.append((file_node, fbrowser.has_name, Literal(agent_action['file_name'])))
        triplets.append((parent_node, fbrowser.contains_file, file_node))
        self.add_triplets(triplets)
        self.add_to_all_triplets(triplets)
        triplets = [(fbrowser.Create_file, fbrowser.has_parameter, file_node)]
        # triplets.append((fbrowser.Agent, fbrowser.a_acted, action_node))

        #     triplets.append((parent_dir, onto.rdf_type, fbrowser.Directory))
        #     triplets.append((parent_dir, fbrowser.has_name, Literal(user_action['parent_directory'])))
        #     triplets.append((parent_dir, fbrowser.contains_file, file_node))
        # update inner state
        self.file_type[file_node] = t
        result = self.add_file_existence(file_node)
        if result is not None:
            f, c = result
            self.remove_file_existence(file_node, True)
            # print('file exists: ',f,' and ',c,' names: ',self.name_by_node[f], ' and ', self.name_by_node[c],
            #       ' paths: ', self.get_path_of_file_node(f), ' and ', self.get_path_of_file_node(c))
            raise FileNameExistsError(self.get_path_of_file_node(c), self.name_by_node[c],
                                      self.file_type[c])
        self.remove_special_candidate(agent_action)
        return triplets

    def delete_file_triplets_a(self, agent_action):
        triplets = []
        # action_node = BNode()
        # triplets.append((action_node, onto.rdf_type, fbrowser.Delete_file))
        triplets.append((fbrowser.Delete_file, fbrowser.has_parameter, agent_action['file_node']))
        # triplets.append((fbrowser.Agent, fbrowser.a_acted, action_node))

        # update inner state
        self.remove_file_existence(agent_action['file_node'], True)
        self.remove_special_candidate(agent_action)
        return triplets

    def move_copy_file_triplets_a(self, agent_action):
        triplets = []
        node = agent_action['file_node']
        desire = fbrowser.Copy_file if agent_action['intent'] == 'Copy_file' else fbrowser.Move_file
        triplets.append((desire, fbrowser.has_parameter, agent_action['file_node']))
        self.remove_special_candidate(agent_action)
        t = self.file_type[node]
        newNode = BNode()
        # self.set_file_in_inner_state(newNode, agent_action['dest_node'])
        # self.set_file_name(newNode, self.name_by_node[node])
        # self.set_file_type(newNode, t)
        dest = agent_action['dest_node']
        name = self.name_by_node[node]
        if self.has_ancestor(dest, node):
            path = self.get_path_with_real_root(node, False)
            dest_path = self.get_path_with_real_root(dest)
            raise MoveFileInsideItself(name, path, dest_path)

        triplets.append((newNode, onto.rdf_type, t))
        triplets.append((newNode, fbrowser.has_name, Literal(name)))
        triplets.append((dest, fbrowser.contains_file, newNode))
        self.add_triplets(triplets)
        self.add_to_all_triplets(triplets)
        result = self.add_file_existence(newNode)
        if result is not None:
            f, c = result
            self.remove_file_existence(newNode, True)
            # print('file exists: ',f,' and ',c,' names: ',self.name_by_node[f], ' and ', self.name_by_node[c],
            #       ' paths: ', self.get_path_of_file_node(f), ' and ', self.get_path_of_file_node(c))
            raise FileNameExistsError(self.get_path_of_file_node(c), self.name_by_node[c],
                                      self.file_type[c])

        if desire == fbrowser.Move_file:
            self.remove_file_existence(node, True)

        return []

    def change_directory_triplets_a(self, agent_action):
        triplets = []
        # action_node = BNode()
        # triplets.append((action_node, onto.rdf_type, fbrowser.Change_directory))
        triplets.append((fbrowser.Change_directory, fbrowser.change_dir_to, agent_action['file_node']))
        # triplets.append((fbrowser.Agent, fbrowser.a_acted, action_node))

        # update inner state
        self.change_directory_node(agent_action['file_node'])
        self.bricolage.set_parent_directory("" if agent_action['file_node'] not in self.parent
                                            else self.name_by_node[self.parent[agent_action['file_node']]])
        return triplets

    ############### FILE RELATED METHODS

    def get_focused_file_node(self, file_name_debug=False):
        prev = self.state_map['last_agent_action']
        if prev['intent'] == 'request':
            return prev['file_node']
        # if file_name_debug:
        #     print('problem should come after a request but didnt')
        return None

    def add_root_file(self, name='~'):
        """
        adds root directory
        :return: root's node and root's name
        """
        triplets = []
        root_uri = fbrowser.prefix1 + "root_directory"
        root = rdflib.URIRef(root_uri)
        self.root = root
        self.add_file_existence(root)
        self.file_type[root] = fbrowser.Directory
        triplets.append((root, onto.rdf_type, fbrowser.Directory))
        self.nodes_by_name[name.lower()] = [root]
        self.name_by_node[root] = name
        triplets.append((root, fbrowser.has_name, Literal(name)))
        self.current_path_node, self.current_path = root, name
        self.add_triplets(triplets)
        self.add_to_all_triplets(triplets)
        return self.root

    def add_tree_to_graph(self, tree_sim, root_node, add_existence=True):

        types = [fbrowser.Directory, fbrowser.RegFile]
        nodes = []
        for f, m in tree_sim.tree():
            n = BNode()
            triplets = [(n, onto.rdf_type, types[f]), (n, fbrowser.has_name, Literal(m['name'])),
                        (root_node, fbrowser.contains_file, n)]
            self.add_triplets(triplets)
            self.add_to_all_triplets(triplets)
            nodes.append(n)
            if not f:
                nodes += self.add_tree_to_graph(m['tree_sim'], n, False)

        if not add_existence:
            return nodes
        for n in nodes:
            self.add_file_existence(n)

    def get_triplets_from_tree(self):
        triplets = []
        for n in self.file_exists:
            if n != self.root:
                triplets += [(n, onto.rdf_type, self.file_type[n]),
                             (n, fbrowser.has_name, Literal(self.name_by_node[n])),
                             (self.parent[n], fbrowser.contains_file, n)]
        return triplets

    def refresh_triplets(self, agent_action):
        if agent_action['intent'] in ['Create_file', 'Delete_file', 'Move_file', 'Copy_file']:
            triplets = self.get_triplets_from_tree()
            self.all_episode_triplets.clear()
            self.add_to_all_triplets(triplets)

    def create_tree_sim(self, root_inf=None):
        if root_inf is None:
            root = self.root
            sim = FileTreeSimulator([], name=self.name_by_node[root])
        else:
            root, sim = root_inf
        if root not in self.children:
            return sim
        for n in self.children[root]:
            if n not in self.file_exists:
                continue
            t = 0 if self.file_type[n] == fbrowser.Directory else 1
            _, m = sim.add_file(self.name_by_node[n], t)
            if not t:
                self.create_tree_sim((n, m['tree_sim']))
        return sim

    def add_known_files_to_graph(self, tree_sim=None):
        """
        adds root directory for now
        :return: root's node and root's name
        """
        if tree_sim is None:
            tree_sim = FileTreeSimulator()
        root = self.add_root_file(tree_sim.name)
        self.add_tree_to_graph(tree_sim, root)

    def add_file_existence(self, file_node, first=True):
        if file_node != self.root:  # in self.parent:
            p = self.parent[file_node]
            for c in self.children[p]:
                if c not in self.name_by_node:
                    continue
                if c != file_node and self.name_by_node[c] == self.name_by_node[file_node] \
                        and c in self.file_exists:  # and self.file_type[file_node] == self.file_type[c]:
                    if first:
                        return file_node, c
                    else:
                        return [c]

            result = self.add_file_existence(p, False)
            if result is not None:
                if len(result) == 2:
                    return result
                elif not self.set_file_in_inner_state(file_node, result[0]):
                    return file_node, result[0]
        self.file_exists.add(file_node)
        return None

    def remove_file_existence(self, file_node, from_all=False, rm_from_parent=True):
        if self.has_ancestor(self.current_path_node, file_node):
            raise RemoveCurrentDirError(self.name_by_node[file_node],
                                        self.get_path_with_real_root(self.current_path_node),
                                        file_node != self.current_path_node)
        if file_node in self.file_exists:
            self.file_exists.remove(file_node)
        if file_node in self.children:
            for c in self.children[file_node]:
                if c in self.file_exists:
                    self.remove_file_existence(c, from_all, False)
                elif from_all:
                    del self.parent[c]
                    del self.file_type[c]
                    if c in self.children:
                        del self.children[c]
                    if c in self.name_by_node:
                        name = self.name_by_node[c]
                        del self.name_by_node[c]
                        self.nodes_by_name[name.lower()].remove(c)
                    if c in self.special_nodes:
                        del self.special_nodes[c]
            self.children[file_node].clear()

        if from_all:
            if rm_from_parent:
                p = self.parent[file_node]
                self.children[p].remove(file_node)
            del self.parent[file_node]
            del self.file_type[file_node]
            if file_node in self.children:
                del self.children[file_node]
            name = self.name_by_node[file_node]
            del self.name_by_node[file_node]
            self.nodes_by_name[name.lower()].remove(file_node)
            if file_node in self.special_nodes:
                del self.special_nodes[file_node]
            for node in self.special_nodes:
                _, dests = self.special_nodes[node]
                if file_node in dests:
                    dests.remove(file_node)

    def get_path_of_file(self, file_name):
        """
        gets file's ancestors
        :param file_name:
        :return:
        """
        assert file_name.lower() in self.nodes_by_name, str(file_name) + " not in inner state of state tracker"
        node = self.choose_suitable_node(self.nodes_by_name[file_name.lower()])
        return self.get_path_of_file_node(node)

    def get_path_with_real_root(self, node, add_self=True):
        path = self.get_path_of_file_node(node, add_self)
        dirs = path.split('/')
        dirs[0] = self.root_path
        return '/'.join(dirs)

    def get_path_of_file_node(self, node, add_self=True):
        """
        gets file's ancestors
        :param add_self:
        :param node:
        :return:
        """
        if node == self.root:
            return self.name_by_node[self.root]
        assert node in self.parent, "PATH FROM NODE ERROR: node has no parent directory node: " + str(node) + \
                                    ' name: ' + str(self.name_by_node[node])
        # # TODO fix error that is happening here
        assert self.file_type[self.parent[node]] == fbrowser.Directory, 'TYPE OF PARENT IS NOT DIRECTORY BUT: ' + \
                                                                        str(self.file_type[self.parent[node]]) + \
                                                                        ' file name is ' + \
                                                                        str(self.name_by_node[self.parent[node]]) + \
                                                                        ' path is ' + \
                                                                        self.get_path_of_file_node(self.parent[node])
        file_name = self.name_by_node[node]
        path = file_name if add_self else ""
        node = self.parent[node]
        while True:
            path = self.name_by_node[node] + '/' + path
            if node not in self.parent:
                break
            node = self.parent[node]
        return path

    def choose_suitable_node(self, nodes):
        # TODO implement LCA to find most plausible node
        return nodes[0]

    def has_ancestor(self, s, p):
        """
        checks if p is ancestor of s
        :param s:
        :param p:
        :return:
        """
        while True:
            if s == p:
                return True
            if s == self.root:
                return False
            s = self.parent[s]

    def set_file_in_inner_state(self, s, parent=None):
        if parent is None:
            parent = self.current_path_node
        # if parent == s:  # case root do nothing
        #     return False
        if self.has_ancestor(parent, s):
            # print('ancestor problem!!')
            return False
        if self.file_type[parent] != fbrowser.Directory:
            name = None
            if s in self.name_by_node:
                name = self.name_by_node[s]
            raise PuttingFileUnderRegularFile(name, self.get_path_with_real_root(parent))

        if s in self.parent:
            assert self.parent[s] in self.children, str(s) + " has parent " + str(self.parent[s]) + \
                                                    " but is not in parent's children"
            self.children[self.parent[s]].remove(s)

        self.parent[s] = parent
        if parent not in self.children:
            self.children[parent] = set()
        self.children[parent].add(s)
        return True

    def set_file_name(self, s, o):
        o = str(o)
        if s in self.name_by_node:
            name = self.name_by_node[s]
            self.nodes_by_name[name.lower()].remove(s)
        if o.lower() not in self.nodes_by_name:
            self.nodes_by_name[o.lower()] = []
        if s not in self.nodes_by_name[o.lower()]:
            self.nodes_by_name[o.lower()].append(s)
        self.name_by_node[s] = o
        nodes = self.nodes_by_name[o.lower()]
        for node in nodes:
            if node not in self.parent and node != self.root:
                print('something is wrong here set file name')

    def set_file_type(self, s, o):
        self.file_type[s] = o

    def change_directory(self, path):
        dirs = path.split('/')
        if dirs[-1] == "":
            del dirs[-1]
        node = self.choose_suitable_node(self.nodes_by_name[dirs[0].lower()])
        # TODO FIX ROOT THAT CAN BE /HOME/USER/ AFTER SPLIT ERRORS SHOULD HAPPEN
        for dir in dirs[1:]:
            children = self.children[node]
            found = False
            for child in children:
                if self.name_by_node[child] == dir:
                    node = child
                    found = True
            if not found:
                raise Exception('incorrect path exception')
        self.change_directory_node(node)

    def change_directory_node(self, node):
        self.current_path, self.current_path_node = self.get_path_of_file_node(node), node

    def get_files_from_graph(self, file_info, exists=True):
        if 'file_name' not in file_info:
            return None
        file_name = file_info['file_name']
        if file_name.lower() not in self.nodes_by_name:
            return None
        # nodes = self.nodes_by_name[file_name]
        nodes = [n for n in self.nodes_by_name[file_name.lower()] if not exists or n in self.file_exists]
        for node in nodes:
            if node not in self.file_type:
                print('THIS IS IMPOSSIBLE NODE NOT IN FILETYPE BUT EXISTS')
        if 'parent_directory' in file_info:
            parent = file_info['parent_directory']
            nodes = [n for n in nodes
                     if n in self.parent and self.name_by_node[self.parent[n]] == parent
                     ]
            if not len(nodes):
                return None
        return set(nodes)

    def get_current_path_files(self):
        files = []
        type_map = {
            fbrowser.Directory: 'directory',
            fbrowser.RegFile: 'file',
            fbrowser.File: 'file'
        }
        if self.current_path_node in self.children:
            for child in self.children[self.current_path_node]:
                files.append((self.name_by_node[child], type_map[self.file_type[child]]))
        return files

    def get_file_from_graph(self, file_info):
        """
        gets file's node from graph, None if it does not exist
        :param file_info:
        :return:
        """
        nodes = self.get_files_from_graph(file_info)
        if nodes is None:
            return None
        return self.choose_suitable_node(nodes)

    def print_tree(self, root=None, pref=''):
        if root is None:
            root = self.root
        # if root not in self.file_exists:
        #     return
        name = '<unknown>' if root not in self.name_by_node else self.name_by_node[root]
        types = {fbrowser.RegFile: '(rfile)', fbrowser.Directory: '(dir)', fbrowser.File: '(file)'}
        t = types[self.file_type[root]]
        print(pref + '>' + str(name) + ' ' + t)
        if root not in self.children:
            return
        for n in self.children[root]:
            self.print_tree(n, pref + '--')


class Bricolage(object):
    def __init__(self) -> None:
        super().__init__()
        self.last_file_refs = ['it', 'that', 'the file', 'that file']
        self.file_slots = ['file_name', 'old_name']
        self.last_directory_refs = ['there', 'it', 'that', 'that directory', 'the directory',
                                    'that folder', 'the folder']
        self.dir_slots = ['directory', 'dest', 'parent_directory', 'origin', 'file_name']
        self.prev_directory_refs = ['back', 'previous folder', 'previous directory', 'previous',
                                    'the parent folder', 'parent folder', 'parent',
                                    'the parent directory', 'parent directory'
                                    ]
        self.prev_slots = ['directory', 'dest', 'parent_directory', 'origin', 'file_name']
        self.last_file = ""
        self.last_directory = ''
        self.parent_directory = ""
        

    def fix_references(self, user_action):
        done = {}
        for slot in self.file_slots:
            if slot in user_action and user_action[slot] in self.last_file_refs and len(self.last_file):
                user_action[slot] = self.last_file
                done[slot] = True

        for slot in self.dir_slots:
            if slot in user_action and user_action[slot] in self.last_directory_refs and slot not in done\
                    and len(self.last_directory):
                user_action[slot] = self.last_directory
                done[slot] = True

        for slot in self.prev_slots:
            if slot in user_action and user_action[slot] in self.prev_directory_refs and slot not in done\
                    and len(self.parent_directory):
                user_action[slot] = self.parent_directory
        return user_action

    def set_parent_directory(self, parent_directory):
        self.parent_directory = parent_directory

    def set_referenced_file(self, referenced_file):
        self.last_file = referenced_file

    def set_referenced_directory(self, referenced_directory):
        self.last_directory = referenced_directory

if __name__ == '__main__':
    state_tracker = StateTrackerFB(1, fbrowser.graph)
    state_tracker.print_tree()
    state_tracker.create_tree_sim().print_tree()
