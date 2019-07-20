from rdflib import BNode

from DialogueManager.FileBrowserDM.user_simulator import UserSimulatorFB
import Ontologies.onto_fbrowser as fbrowser

class IntentTracker(object):

    open_file_requirements = {
        'file_name': (True, []),
        'parent_directory': (False, ['file_name'])
    }
    rename_file_requirements = {
        'old_name': (True, []),
        'new_name': (True, ['old_name']),
        'parent_directory': (False, ['old_name'])
    }
    search_file_requirements = {
        'file_name': (True, [])
    }
    change_directory_requirements = {
        'directory': (True, [])
    }
    delete_file_requirements = {
        'file_name': (True, []),
        'parent_directory': (False, ['file_name'])
    }
    create_file_requirements = {
        'file_name': (True, []),
        'parent_directory': (False, ['file_name']),
        'is_file': (True, ['file_name'])
    }
    move_file_requirements = {
        'file_name': (True, []),
        'dest': (True, ['file_name']),
        'origin': (False, ['file_name'])
    }
    copy_file_requirements = {
        'file_name': (True, []),
        'dest': (True, ['file_name']),
        'origin': (False, ['file_name'])
    }
    intents_requirements = {
        UserSimulatorFB.Open_file_desire: open_file_requirements,
        UserSimulatorFB.Rename_file_desire: rename_file_requirements,
        UserSimulatorFB.u_request: search_file_requirements,
        UserSimulatorFB.Change_directory_desire: change_directory_requirements,
        UserSimulatorFB.Create_file_desire: create_file_requirements,
        UserSimulatorFB.Delete_file_desire: delete_file_requirements,
        UserSimulatorFB.Move_file_desire: move_file_requirements,
        UserSimulatorFB.Copy_file_desire: copy_file_requirements
    }
    slot_converter = {
        'directory': ['directory', 'new_directory', 'file_name'],
        'old_name': ['old_name', 'file_name', 'directory'],
        'new_name': ['new_name', 'file_name', 'directory'],
        'parent_directory': ['parent_directory', 'file_name', 'directory'],
        'dest': ['dest', 'file_name', 'parent_directory', 'directory'],
        'origin': ['origin', 'file_name', 'parent_directory', 'directory']
    }

    def __init__(self) -> None:
        super().__init__()
        self.current_intent_info = {
            'name': None
        }
        self.current_intent_requirements = {

        }

    def set_current_intent(self, user_action):
        self.current_intent_requirements = self.intents_requirements[user_action['intent']]
        self.current_intent_info = {
            'name': user_action['intent']
        }
        if user_action['intent'] == UserSimulatorFB.Change_directory_desire and 'file_name' in user_action:
            user_action['directory'] = user_action['file_name']
        for key in self.current_intent_requirements:
            if key in user_action:
                self.current_intent_info[key] = user_action[key]

        # for key in user_action:
        #     if key in self.current_intent_requirements and key not in self.current_intent_info:
        #

    def add_inform_intent(self, user_action, prev_user_action):
        if prev_user_action['intent'] == 'request':
            slot = prev_user_action['slot']
            cslots = self.slot_converter[slot] if slot in self.slot_converter else [slot]
            for s in cslots:
                if s in user_action:
                    value = user_action[s]
                    del user_action[s]
                    user_action[slot] = value
                    break

        for key in self.current_intent_requirements:
            if key in user_action:
                self.current_intent_info[key] = user_action[key]

    def clear_current_intent(self):
        self.current_intent_info = {
            'name': None
        }
        self.current_intent_requirements = {

        }

    def all_required_slots_filled(self):
        for key in self.current_intent_requirements:
            r, _ = self.current_intent_requirements[key]
            if r and key not in self.current_intent_info:
                return False
        return True
    def has_intent(self):
        return self.current_intent_info['name'] is not None

    def is_intent_supported(self, user_action):
        return user_action['intent'] in self.intents_requirements

    def get_request_key_needs(self, key):
        assert key in self.current_intent_requirements, "key is not in current intent possible slots" + key
        _, needed = self.current_intent_requirements[key]
        result = {}
        for slot in needed:
            if slot not in self.current_intent_info:
                return None
            result[slot] = self.current_intent_info[slot]
        return result


class ActionTracker(object):
    file_node_slot_filler = {
        'file_name': 'desire',
        'directory': 'desire',
        'old_name': 'desire'
    }

    user_action_to_onto_node = {
        UserSimulatorFB.Open_file_desire: fbrowser.Open_file,
        UserSimulatorFB.Rename_file_desire: fbrowser.Rename_file,
        UserSimulatorFB.u_request: fbrowser.A_inform,
        UserSimulatorFB.Change_directory_desire: fbrowser.Change_directory,
        UserSimulatorFB.Copy_file_desire: fbrowser.Copy_file,
        UserSimulatorFB.Move_file_desire: fbrowser.Move_file,
        UserSimulatorFB.Delete_file_desire: fbrowser.Delete_file,
        UserSimulatorFB.Create_file_desire: fbrowser.Create_file
    }

    def __init__(self, state_tracker) -> None:
        """

        :param StateTrackerFB state_tracker:
        """
        super().__init__()
        self.state_tracker = state_tracker
        self.nodes_updater = {
            UserSimulatorFB.Open_file_desire: self.get_open_file_nodes,
            UserSimulatorFB.Rename_file_desire: self.get_rename_file_nodes,
            UserSimulatorFB.u_request: self.get_search_file_nodes,
            UserSimulatorFB.Change_directory_desire: self.get_change_directory_nodes,
            UserSimulatorFB.Create_file_desire: self.get_create_file_nodes,
            UserSimulatorFB.Delete_file_desire: self.get_delete_file_nodes,
            UserSimulatorFB.Move_file_desire: self.get_move_file_nodes,
            UserSimulatorFB.Copy_file_desire: self.get_copy_file_nodes
        }
        self.possible_actions = {
            UserSimulatorFB.Open_file_desire: self.possible_actions_open,
            UserSimulatorFB.Rename_file_desire: self.possible_actions_rename,
            UserSimulatorFB.u_request: self.possible_actions_search,
            UserSimulatorFB.Change_directory_desire: self.possible_actions_change_dir,
            UserSimulatorFB.Create_file_desire: self.possible_actions_create,
            UserSimulatorFB.Delete_file_desire: self.possible_actions_delete,
            UserSimulatorFB.Move_file_desire: self.possible_actions_move,
            UserSimulatorFB.Copy_file_desire: self.possible_actions_copy
        }
        self.current_action_info = {
            'desire': None,
            'nodes_info': None
        }

        self.intent_tracker = IntentTracker()

    def update_action_info(self):
        intent = self.intent_tracker.current_intent_info['name']
        self.current_action_info['desire'] = self.user_action_to_onto_node[intent]
        self.update_files_nodes()

    def set_current_action(self, user_action):
        self.clear_current_action()
        self.intent_tracker.set_current_intent(user_action)
        self.update_action_info()

    def add_inform_intent(self, user_action, prev_user_action):
        self.intent_tracker.add_inform_intent(user_action, prev_user_action)
        self.update_action_info()

    def clear_current_action(self):
        self.intent_tracker.clear_current_intent()
        self.current_action_info = {
            'desire': None,
            'nodes_info': None
        }

    def has_intent(self):
        return self.intent_tracker.current_intent_info['name'] is not None

    def is_intent_supported(self, user_action):
        return user_action['intent'] in self.intent_tracker.intents_requirements

    def update_files_nodes(self):
        self.current_action_info['nodes_info'] = self.nodes_updater[self.intent_tracker.current_intent_info['name']]()
        # print('nodes:', self.current_action_info['nodes_info'])

    def get_possible_actions(self):
        self.update_files_nodes()
        actions = []
        for key in self.intent_tracker.current_intent_requirements:
            # print('start', key)
            if key == 'name':
                # print('no name')
                continue
            needs = self.intent_tracker.get_request_key_needs(key)
            if needs is None:
                # print('needs is None')
                continue
            if key in self.file_node_slot_filler: # if it's file_name, old_name, directory
                if key in self.intent_tracker.current_intent_info:
                    candidate_nodes = self.current_action_info['nodes_info']['candidate_nodes']
                    intent_info = self.intent_tracker.current_intent_info
                    needs['fname'] = intent_info[key]
                    if len(candidate_nodes) > 1:
                        paths = [self.state_tracker.get_path_with_real_root(node) for node in candidate_nodes]
                        needs['nlg'] = {'multiple_files_found': paths}
                    elif len(candidate_nodes) == 0:
                        value = None
                        if 'parent_directory' in intent_info:
                            value = intent_info['parent_directory']
                        if 'origin' in intent_info:
                            value = intent_info['origin']
                        needs['nlg'] = {'no_file_found': value}
                needs['file_node'] = self.current_action_info[self.file_node_slot_filler[key]]
                actions.append(self.create_request_action(key, needs))
            else:
                for node in self.current_action_info['nodes_info']['candidate_nodes']:
                    needs['file_node'] = node
                    actions.append(self.create_request_action(key, needs))
        if self.intent_tracker.all_required_slots_filled():
            # print('slot filled')
            actions += self.possible_actions[self.intent_tracker.current_intent_info['name']]()
        # print(self.intent_tracker.current_intent_info)
        return actions

    def possible_actions_open(self):
        candidates = self.current_action_info['nodes_info']['candidate_nodes']
        if candidates is None or len(candidates) == 0:
            return []
        actions = []
        for node in candidates:
            actions.append({'intent': 'Open_file', 'file_name': self.intent_tracker.current_intent_info['file_name'],
                            'path': self.state_tracker.get_path_with_real_root(node, False),
                            'action_node': fbrowser.Open_file, 'file_node': node})
        return actions

    def possible_actions_rename(self):
        candidates = self.current_action_info['nodes_info']['candidate_nodes']
        if candidates is None or len(candidates) == 0:
            return []
        actions = []
        for node in candidates:
            actions.append({'intent': 'Rename_file', 'old_name': self.intent_tracker.current_intent_info['old_name'],
                            'new_name': self.intent_tracker.current_intent_info['new_name'],
                            'path': self.state_tracker.get_path_with_real_root(node, False),
                            'action_node': fbrowser.Rename_file, 'file_node': node})
        return actions

    def possible_actions_change_dir(self):
        candidates = self.current_action_info['nodes_info']['candidate_nodes']
        if candidates is None or len(candidates) == 0:
            # print('out here because candidates is ', candidates)
            return []
        actions = []
        for node in candidates:
            actions.append({'intent': 'Change_directory',
                            'new_directory': self.state_tracker.get_path_with_real_root(node),
                            'action_node': fbrowser.Change_directory, 'file_node': node})
        return actions

    def possible_actions_search(self):
        candidates = self.current_action_info['nodes_info']['candidate_nodes']
        if candidates is None or len(candidates) == 0:
            if 'file_name' in self.intent_tracker.current_intent_info:
                return [{
                    'intent': 'inform',
                    'file_name': self.intent_tracker.current_intent_info['file_name'],
                    'paths': [],
                    'action_node': fbrowser.U_inform,
                    'file_node': fbrowser.U_inform
                }]
        actions = [{'intent': 'inform', 'file_name': self.intent_tracker.current_intent_info['file_name'],
                    'paths': [self.state_tracker.get_path_with_real_root(node, False) for node in candidates],
                    'action_node': fbrowser.U_inform, 'file_node': list(candidates)[0]}]
        return actions

    def possible_actions_create(self):
        candidates = self.current_action_info['nodes_info']['candidate_nodes']
        if candidates is None or len(candidates) == 0:
            return []
        node = candidates[0]
        parent_nodes = self.current_action_info['nodes_info']['parent_nodes']
        if parent_nodes is None or len(parent_nodes) == 0:
            return []
        actions = []
        infos = self.intent_tracker.current_intent_info
        file_type = 'file' if infos['is_file'] else 'directory'
        for p in parent_nodes:
            actions.append({'intent': 'Create_file', 'file_name': infos['file_name'], 'is_file': infos['is_file'],
                            'path': self.state_tracker.get_path_with_real_root(p), 'file_node': node,
                            'action_node': fbrowser.Create_file, 'file_type': file_type, 'parent_node': p})
        return actions

    def possible_actions_delete(self):
        candidates = self.current_action_info['nodes_info']['candidate_nodes']
        if candidates is None or len(candidates) == 0:
            return []
        actions = []
        infos = self.intent_tracker.current_intent_info
        for node in candidates:
            try:
                file_type = 'directory' if self.state_tracker.file_type[node] == fbrowser.Directory else 'file'
                actions.append({'intent': 'Delete_file', 'file_name': infos['file_name'], 'file_type': file_type,
                                'path': self.state_tracker.get_path_with_real_root(node, False),
                                'action_node': fbrowser.Delete_file, 'file_node': node})
            except KeyError as e:
                print(e)
                if node not in self.state_tracker.file_exists:
                    print('freaking it does not exist, how come!')
                else:
                    print('well it exists')
                    print(self.state_tracker.get_path_with_real_root(node))
                print(candidates)
                raise e
        return actions

    def possible_actions_move(self):
        candidates = self.current_action_info['nodes_info']['candidate_nodes']
        if candidates is None or len(candidates) == 0:
            return []
        dest_nodes = self.current_action_info['nodes_info']['dest_nodes']
        if dest_nodes is None or len(dest_nodes) == 0:
            return []

        actions = []
        infos = self.intent_tracker.current_intent_info
        for node in candidates:
            for dest in dest_nodes:
                actions.append({'intent': 'Move_file', 'file_name': infos['file_name'],
                                'origin': self.state_tracker.get_path_with_real_root(node, False),
                                'dest': self.state_tracker.get_path_with_real_root(dest), 'action_node': dest,
                                'file_node': node, 'dest_node': dest})
        return actions

    def possible_actions_copy(self):
        candidates = self.current_action_info['nodes_info']['candidate_nodes']
        if candidates is None or len(candidates) == 0:
            # print('out candidates')
            return []
        dest_nodes = self.current_action_info['nodes_info']['dest_nodes']
        if dest_nodes is None or len(dest_nodes) == 0:
            # print('out dest')
            return []

        actions = []
        infos = self.intent_tracker.current_intent_info
        for node in candidates:
            for dest in dest_nodes:
                actions.append({'intent': 'Copy_file', 'file_name': infos['file_name'],
                                'origin': self.state_tracker.get_path_with_real_root(node, False),
                                'dest': self.state_tracker.get_path_with_real_root(dest), 'action_node': dest,
                                'file_node': node, 'dest_node': dest})
        return actions

    """
    FILE NODES UPDATE METHODS
    """

    def get_open_file_nodes(self):
        candidates = self.state_tracker.get_files_from_graph(self.intent_tracker.current_intent_info)
        return {'candidate_nodes': candidates if candidates is not None else []}

    def get_rename_file_nodes(self):
        file_infos = self.intent_tracker.current_intent_info.copy()
        if 'old_name' in file_infos:
            file_infos['file_name'] = file_infos['old_name']
        candidates = self.state_tracker.get_files_from_graph(file_infos)
        return {'candidate_nodes': candidates if candidates is not None else []}

    def get_search_file_nodes(self):
        candidates = self.state_tracker.get_files_from_graph(self.intent_tracker.current_intent_info)
        return {'candidate_nodes': candidates if candidates is not None else []}

    def get_change_directory_nodes(self):
        file_infos = {}
        if 'directory' in self.intent_tracker.current_intent_info:
            file_infos['file_name'] = self.intent_tracker.current_intent_info['directory']
        candidates = self.state_tracker.get_files_from_graph(file_infos)
        return {'candidate_nodes': candidates if candidates is not None else []}

    def get_create_file_nodes(self):
        file_infos = {}
        if 'parent_directory' in self.intent_tracker.current_intent_info:
            file_infos['file_name'] = self.intent_tracker.current_intent_info['parent_directory']
        candidates = self.state_tracker.get_files_from_graph(file_infos)
        if 'create_node' not in self.current_action_info:
            self.current_action_info['create_node'] = BNode()
            self.state_tracker.graph._add_node(self.current_action_info['create_node'])
        node = self.current_action_info['create_node']
        return {
            'candidate_nodes': [node],
            'parent_nodes': candidates if candidates is not None else [self.state_tracker.current_path_node]
        }

    def get_delete_file_nodes(self):
        candidates = self.state_tracker.get_files_from_graph(self.intent_tracker.current_intent_info)
        return {'candidate_nodes': candidates if candidates is not None else []}

    def get_move_file_nodes(self):
        file_infos = self.intent_tracker.current_intent_info.copy()
        if 'origin' in self.intent_tracker.current_intent_info:
            file_infos['parent_directory'] = self.intent_tracker.current_intent_info['origin']
        candidates = self.state_tracker.get_files_from_graph(file_infos)
        file_infos = {}
        if 'dest' in self.intent_tracker.current_intent_info:
            file_infos['file_name'] = self.intent_tracker.current_intent_info['dest']
        dest_nodes = self.state_tracker.get_files_from_graph(file_infos)
        return {
            'candidate_nodes': candidates if candidates is not None else [],
            'dest_nodes': dest_nodes if dest_nodes is not None else []
        }

    def get_copy_file_nodes(self):
        file_infos = self.intent_tracker.current_intent_info.copy()
        if 'origin' in self.intent_tracker.current_intent_info:
            file_infos['parent_directory'] = self.intent_tracker.current_intent_info['origin']
        candidates = self.state_tracker.get_files_from_graph(file_infos)
        file_infos = {}
        if 'dest' in self.intent_tracker.current_intent_info:
            file_infos['file_name'] = self.intent_tracker.current_intent_info['dest']
        dest_nodes = self.state_tracker.get_files_from_graph(file_infos)
        return {
            'candidate_nodes': candidates if candidates is not None else [],
            'dest_nodes': dest_nodes if dest_nodes is not None else []
        }

    """
    STATIC METHODS
    """

    @staticmethod
    def create_request_action(key, needed):
        action = {'intent': 'request', 'slot': key, 'action_node': fbrowser.A_request}
        for slot in needed:
            action[slot] = needed[slot]
        return action


