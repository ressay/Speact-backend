import json
import random, copy
from DialogueManager.dialogue_config import FAIL, SUCCESS, NO_OUTCOME
from DialogueManager.user_simulator import UserSimulator
from rdflib import Graph
from DialogueManager.FileBrowserDM.file_tree_sim import FileTreeSimulator


class UserSimulatorFB(UserSimulator):
    """Simulates a real user, to train the agent with reinforcement learning."""
    """
    Debugging mask
    """
    CURRENT_TREE = 1
    GOAL_TREE = 2
    SUB_GOALS = 4
    GOAL_DIR = 8
    SIMILARITY = 16
    """
    User possible actions
    """
    Change_directory_desire = "Change_directory_desire"
    Delete_file_desire = 'Delete_file_desire'
    Create_file_desire = 'Create_file_desire'
    Copy_file_desire = 'Copy_file_desire'
    Move_file_desire = 'Move_file_desire'
    Open_file_desire = 'Open_file_desire'
    Rename_file_desire = 'Rename_file_desire'
    # Create_directory_desire = 'Create_directory_desire'
    u_inform = 'inform'
    u_request = 'request'
    confirm = 'confirm'
    deny = 'deny'

    usersim_intents = [
        Change_directory_desire,
        Delete_file_desire,
        Create_file_desire,
        Copy_file_desire,
        Move_file_desire,
        Open_file_desire,
        Rename_file_desire,
        u_inform,
        u_request,
        confirm,
        deny
    ]

    def __init__(self, constants, ontology, rewards=None, probas=None):
        """
        The constructor for UserSimulator. Sets dialogue config variables.

        Parameters:
            constants (dict): Dict of constants loaded from file
            ontology (rdflib.Graph): ontology graph
        """
        super().__init__(constants, ontology)
        self.agent_tree_actions = ["Create_file", "Delete_file", "Copy_file", "Move_file", "Rename_file"]
        for a in self.agent_tree_actions:
            self.user_responses[a] = self._build_response
        self.user_responses['Change_directory'] = self._build_response
        self.user_responses['Open_file'] = self._build_response
        self.user_responses['inform'] = self._inform_response
        self.user_responses['ask'] = self._ask_response
        self.user_responses['request'] = self._req_response
        self.user_responses['default'] = self._build_response
        self.debug = 0
        self.subgoal_reward = None
        self.rewards = {}
        if rewards is not None:
            self.rewards = rewards
        self.probas = {}
        if probas is not None:
            self.probas = probas

    def generate_goal(self):
        goal_tree = self.state['current_file_tree'].copy()
        # as 30% of actions contain errors of infuser, we multiply by 1.3
        self.max_round = int(goal_tree.random_modifications() * 4 * 2)
        self.goal = {'goal_tree': goal_tree, 'end_directory': goal_tree.get_random_directory().path(), 'sub_goal': []}
        # print("new goal:")
        # goal_tree.print_tree()
        return self.generate_next_focused_file() is not None

    def reset(self, data):
        """
        Resets the user sim. by emptying the state and returning the initial action.

        Returns:
            dict: The initial action of an episode
        """
        fileTreeSimulator = data['current_tree_sim'].copy()
        self.state = {
            # empty file tree
            'current_file_tree': fileTreeSimulator,
            'focused_file': {'file': -1, 'map': {}, 'delete': -1},
            'current_directory': '~/',
            'previous_directory': '~/',
            'previous_similarity': [0, 1],
            'current_similarity': [0, 1],
            'previous_uAction': None,
            'current_uAction': None
        }
        self.round = 0
        while True:
            t = self.generate_goal()
            if t:
                break
            self.print_debug()
        goal_sim = self.goal['goal_tree']
        found, total = fileTreeSimulator.tree_similarity(goal_sim)
        self.state['current_similarity'] = [found, total]
        self.state['previous_similarity'] = [found, total]
        # self.print_debug()
        # self.max_round = self.goal['goal_tree'].r_size() * 4
        return self._return_init_action()

    def _return_init_action(self):
        """
        Returns the initial action of the episode.

        The initial action has an intent of request, required init. inform slots and a single request slot.

        Returns:
            dict: Initial user response
        """
        self.add_random_sub_goal()
        user_response = self.generate_tree_desire_intent()
        self.state['previous_intent'] = user_response
        return user_response

    def step(self, agent_action):
        """
        Return the response of the user sim. to the agent by using rules that simulate a user.

        Given the agent action craft a response by using deterministic rules that simulate (to some extent) a user.
        Some parts of the rules are stochastic. Check if the agent has succeeded or lost or still going.

        Parameters:
            agent_action (dict): The agent action that the user sim. responds to

        Returns:
            dict: User sim. response
            int: Reward
            bool: Done flag
            int: Success: -1, 0 or 1 for loss, neither win nor loss, win
        """
        self.state['previous_uAction'] = self.state['current_uAction']
        done = False
        self.round += 1
        # First check round num, if equal to max then fail
        if self.round == self.max_round:
            done = True
            success = FAIL
            user_response = self._end_response()
        else:
            # try:
            success = self.update_state(agent_action)
            if success:
                done = True
                user_response = self._end_response()
            else:
                agent_intent = agent_action['intent']
                assert agent_intent in self.user_responses, 'Not acceptable agent action'
                user_response = self.user_responses[agent_intent](agent_action)
                # except Exception as e:
                #     print('ERROR HAPPENED AND IGNORING IT: ', e)
                #     return self._default_response(), -5, False, 0
        self.state['current_uAction'] = user_response
        reward = self.reward_function(agent_action, success)
        self.print_debug()
        return user_response, reward, done, 1 if success == 1 else 0

    def reward_function(self, agent_action, success):
        if success:
            if 'success' in self.rewards:
                return self.rewards['success']
            return 2
        if 'required_reward' in agent_action:
            return agent_action['required_reward']
        # if self.sub_goal_exists():  # if there are pending sub_goals
        #     reward = self.get_sub_goal_reward(self.goal['sub_goal'][0])
        #     if reward:
        #         return reward
        if self.subgoal_reward is not None:
            reward = self.subgoal_reward
            self.subgoal_reward = None
            return reward
        f, t = self.state['current_similarity']
        pf, pt = self.state['previous_similarity']
        if t != 0 and f / t > pf / pt:  # tree similarity got better
            if 'improved' in self.rewards:
                return self.rewards['improved']
            return 2
        elif t == 0 or f / t < pf / pt:
            if f < pf:
                if 'nimprovedf' in self.rewards:
                    return self.rewards['nimprovedf']
                return -3
            if 'nimproved' in self.rewards:
                return self.rewards['nimproved']
            return -3
        # if confirming an action for agent, reward is neutral
        if self.state['current_uAction']['intent'] == self.confirm:
            if 'confirm' in self.rewards:
                return self.rewards['confirm']
            return 0.25
        if self.state['current_uAction']['intent'] == self.deny:
            if 'deny' in self.rewards:
                return self.rewards['deny']
            return -0.25

        if 'other' in self.rewards:
            return self.rewards['other']
        return -1

    def apply_agent_tree_action(self, agent_action, f_sim):
        """

        :param (dict) agent_action: dictionary that contains agent action
        :param (FileTreeSimulator) f_sim: tree simulator to apply the agent action on
        :return:
        """
        intent = agent_action['intent']
        assert intent in self.agent_tree_actions, "trying to apply action that doesn't exist in agent_tree_actions"
        if intent == 'Create_file':
            f_sim.add_file(agent_action['file_name'], agent_action['is_file'], agent_action['path'], True)
        elif intent == 'Delete_file':
            f_sim.remove_file(agent_action['file_name'], agent_action['path'])
        elif intent == 'Move_file':
            f_sim.move_file(agent_action['file_name'], agent_action['origin'], agent_action['dest'])
        elif intent == 'Copy_file':
            f_sim.copy_file(agent_action['file_name'], agent_action['origin'], agent_action['dest'])
        elif intent == 'Rename_file':
            f_sim.rename_file(agent_action['old_name'], agent_action['new_name'], agent_action['path'])

    def update_sub_goals(self, agent_action):
        for sub_goal in self.goal['sub_goal']:
            if sub_goal['name'] == 'Change_directory':
                last_dir = sub_goal['dirs'][-1]
                current_dir = self.state['current_directory']
                if current_dir[-1] == '/': current_dir = current_dir[:-1]
                if last_dir[-1] == '/': last_dir = last_dir[:-1]
                if current_dir == last_dir:
                    self.goal['sub_goal'].remove(sub_goal)
                    self.subgoal_reward = len(sub_goal['dirs'])
                elif current_dir in sub_goal['dirs']:
                    self.subgoal_reward = sub_goal['dirs'].index(current_dir) + 1
                    del sub_goal['dirs'][:self.subgoal_reward]
            if sub_goal['name'] == 'Search_file':
                if agent_action['intent'] == 'inform' and 'paths' in agent_action:
                    self.subgoal_reward = 0
                    paths = agent_action['paths']
                    r = self.state['current_file_tree'].lookup_file_name(sub_goal['file'])
                    self.subgoal_reward = -2
                    if r is not None:
                        f, m = r
                        for path in paths:
                            if FileTreeSimulator.equal_paths(m['tree_sim'].path(True), path):
                                self.subgoal_reward = 2
                                break

                    self.goal['sub_goal'].remove(sub_goal)
            if sub_goal['name'] == 'Rename_file':
                if agent_action['intent'] == 'Rename_file':
                    p_dir = FileTreeSimulator.last_dir_in_path(agent_action['path'])
                    agent_action['parent_directory'] = p_dir
                    keys = ['old_name', 'new_name', 'parent_directory']
                    self.subgoal_reward = 2
                    for key in keys:
                        if agent_action[key] != sub_goal[key]:
                            self.subgoal_reward = -3

                    self.goal['sub_goal'].remove(sub_goal)
            if sub_goal['name'] == 'Open_file':
                if agent_action['intent'] == sub_goal['name']:
                    p_dir = FileTreeSimulator.last_dir_in_path(agent_action['path'])
                    if p_dir != sub_goal['parent_directory'] or agent_action['file_name'] != sub_goal['file']:
                        self.subgoal_reward = -3
                    else:
                        self.subgoal_reward = 2
                    self.goal['sub_goal'].remove(sub_goal)
            if sub_goal['name'] == 'Move_file' or sub_goal['name'] == 'Copy_file':
                if agent_action['intent'] == sub_goal['name']:
                    keys = ['origin', 'dest']
                    agent_action['file'] = agent_action['file_name']
                    self.subgoal_reward = 2
                    for key in keys:
                        if not FileTreeSimulator.equal_paths(sub_goal[key], agent_action[key]):
                            self.subgoal_reward = -3
                            break
                    if agent_action['file'] != sub_goal['file']:
                        self.subgoal_reward = -3
                    self.goal['sub_goal'].remove(sub_goal)
                    break

    def _ask_sub_goal(self, agent_action):
        if not self.sub_goal_exists():
            return None
        asked_action = agent_action['action']
        sub_goal = self.next_sub_goal()
        if asked_action['intent'] == 'Change_directory' and sub_goal['name'] == 'Change_directory' \
                and asked_action['new_directory'] in sub_goal['dirs']:
            return {'intent': self.confirm}
        if asked_action['intent'] in ('Move_file', 'Copy_file') and sub_goal['name'] == asked_action['intent']:
            keys = ['origin', 'dest']
            asked_action['file'] = asked_action['file_name']
            response = {'intent': self.confirm}
            for key in keys:
                if not FileTreeSimulator.equal_paths(sub_goal[key], asked_action[key]):
                    return {'intent': self.deny}
            if asked_action['file'] != sub_goal['file']:
                return {'intent': self.deny}
            return response
        if asked_action['intent'] == 'Open_file' and sub_goal['name'] == asked_action['intent']:
            p_dir = FileTreeSimulator.last_dir_in_path(asked_action['path'])
            if p_dir == sub_goal['parent_directory'] and asked_action['file_name'] == sub_goal['file']:
                return {'intent': self.confirm}
            else:
                return {'intent': self.deny}
        if sub_goal['name'] == 'Rename_file' and asked_action['intent'] == sub_goal['name']:
            p_dir = FileTreeSimulator.last_dir_in_path(asked_action['path'])
            asked_action['parent_directory'] = p_dir
            keys = ['old_name', 'new_name', 'parent_directory']
            response = {'intent': self.confirm}
            for key in keys:
                if asked_action[key] != sub_goal[key]:
                    return {'intent': self.deny}
            return response

        return None

    def sub_goal_exists(self):
        return len(self.goal['sub_goal']) > 0

    def next_sub_goal(self):
        return self.goal['sub_goal'][0]

    def add_change_directory_sub_goal(self, dirs):
        self.goal['sub_goal'].append({'name': 'Change_directory',
                                      'dirs': [(d[:-1] if d[-1] == '/' else d) for d in dirs]})

    def add_copy_sub_goal(self, origin, dest, file):
        self.goal['sub_goal'].append({'name': 'Copy_file',
                                      'origin': origin,
                                      'dest': dest,
                                      'file': file})

    def add_move_sub_goal(self, origin, dest, file):
        self.goal['sub_goal'].append({'name': 'Move_file',
                                      'origin': origin,
                                      'dest': dest,
                                      'file': file})

    def add_search_sub_goal(self, file):
        self.goal['sub_goal'].append({'name': 'Search_file',
                                      'file': file})

    def add_open_sub_goal(self, file, p_dir):
        self.goal['sub_goal'].append({'name': 'Open_file',
                                      'file': file,
                                      'parent_directory': p_dir})

    def add_rename_sub_goal(self, old_file, new_file, p_dir):
        self.goal['sub_goal'].append({'name': 'Rename_file',
                                      'old_name': old_file,
                                      'new_name': new_file,
                                      'parent_directory': p_dir})

    def get_slot_from_sub_goal(self, slot):
        if not self.sub_goal_exists():
            return None
        sub_goal = self.next_sub_goal()
        if sub_goal['name'] in ('Copy_file', 'Move_file', 'Search_file', 'Open_file'):
            if slot == 'file_name':
                return sub_goal['file']
            if slot == 'parent_directory' or slot == 'origin':
                if 'parent_directory' in sub_goal:
                    return sub_goal['parent_directory']
                if 'origin' in sub_goal:
                    return FileTreeSimulator.last_dir_in_path(sub_goal['origin'])
            if slot == 'dest' and 'dest' in sub_goal:
                return FileTreeSimulator.last_dir_in_path(sub_goal['dest'])
        elif sub_goal['name'] == 'Rename_file':
            if slot == 'old_name':
                return sub_goal['old_name']
            if slot == 'new_name':
                return sub_goal['new_name']
            if slot == 'parent_directory':
                return sub_goal['parent_directory']
        elif sub_goal['name'] == 'Change_directory':
            if slot == 'directory':
                return FileTreeSimulator.last_dir_in_path(sub_goal['dirs'][0])
        return None

    def generate_sub_goal_intent(self):
        sub_goal = self.next_sub_goal()
        pF = 0.5
        if sub_goal['name'] == 'Change_directory':
            assert self.state['current_directory'] != sub_goal['dirs'][-1], 'sub goal already reached'
            dirs = sub_goal['dirs']
            index = int(random.uniform(0, len(dirs) - 0.01))
            directory = dirs[index].split('/')[-1]
            return self.create_change_dir_desire(directory)
        elif sub_goal['name'] == 'Search_file':
            if random.random() < pF:
                return {'intent': self.u_request, 'slot': 'directory', 'file_name': sub_goal['file']}
            return {'intent': self.u_request, 'slot': 'directory'}
        elif sub_goal['name'] == 'Open_file':
            action = {'intent': self.Open_file_desire}
            if random.random() < pF:
                action['file_name'] = sub_goal['file']
            if random.random() < pF:
                action['parent_directory'] = sub_goal['parent_directory']
            return action
        elif sub_goal['name'] == 'Rename_file':
            action = {'intent': self.Rename_file_desire}
            if random.random() < pF:
                action['old_name'] = sub_goal['old_name']
            if random.random() < pF:
                action['new_name'] = sub_goal['new_name']
            if random.random() < pF:
                action['parent_directory'] = sub_goal['parent_directory']
            return action
        elif sub_goal['name'] == 'Move_file' or sub_goal['name'] == 'Copy_file':
            pO, pD, pF = 0.2, 0.4, 0.8
            action = {'intent': (self.Move_file_desire if sub_goal['name'] == 'Move_file' else self.Copy_file_desire)}
            if random.uniform(0, 1) < pO:
                action['origin'] = FileTreeSimulator.last_dir_in_path(sub_goal['origin'])
            if random.uniform(0, 1) < pD:
                action['dest'] = FileTreeSimulator.last_dir_in_path(sub_goal['dest'])
            if random.uniform(0, 1) < pF:
                action['file_name'] = sub_goal['file']
            return action
        return None

    def get_sub_goal_reward(self, sub_goal):
        reward = 0
        if sub_goal['name'] == 'Change_directory':
            current_dir = self.state['current_directory']
            if current_dir[-1] == '/': current_dir = current_dir[:-1]
            if current_dir in sub_goal['dirs']:
                reward = sub_goal['dirs'].index(current_dir) + 1
                del sub_goal['dirs'][:reward]
                if not len(sub_goal['dirs']):  # finished sub_goal
                    self.goal['sub_goal'].remove(sub_goal)
        return reward

    def add_random_sub_goal(self):
        # probabilities of copy/move, search file
        pCM = 0.1
        pS = 0.05
        pRF = 0.05
        pOF = 0.05

        if random.uniform(0, 1) <= pCM + pS + pOF + pRF:
            if random.uniform(0, pCM + pS + pOF + pRF) <= pRF:
                r = self.state['current_file_tree'].get_random_file()
                if r is not None:
                    f, m = r
                    chars = [chr(i) for i in range(ord('a'), ord('z') + 1)]
                    random_name = ''.join([chars[random.randint(0, len(chars) - 1)] for i in range(4)])
                    self.add_rename_sub_goal(m['name'], random_name, m['tree_sim'].parent.name)
                    self.max_round += 6
            elif random.uniform(0, pCM + pS + pOF) <= pOF:
                r = self.state['current_file_tree'].get_random_file()
                if r is not None:
                    f, m = r
                    self.add_open_sub_goal(m['name'], m['tree_sim'].parent.name)
                    self.max_round += 2
            elif random.uniform(0, pCM + pS) <= pS:
                r = self.state['current_file_tree'].get_random_file()
                if r is not None:
                    f, m = r
                    self.add_search_sub_goal(m['name'])
                    self.max_round += 2
            else:
                if random.uniform(0, 1) < 0.5:
                    modif = self.state['current_file_tree'].random_copy_modif()
                    if modif is not None and modif['origin'] != modif['dest']:
                        self.add_copy_sub_goal(modif['origin'], modif['dest'], modif['file_name'])
                        self.max_round += 4
                else:
                    modif = self.state['current_file_tree'].random_move_modif()
                    if modif is not None and modif['origin'] != modif['dest']:
                        self.add_move_sub_goal(modif['origin'], modif['dest'], modif['file_name'])
                        self.max_round += 8

    def update_state(self, agent_action):
        """
        :param (dict) agent_action: action of the dialogue manager
        :return (int) : 1 if success reached, 0 else wise
        """
        intent = agent_action['intent']
        f_sim = self.state['current_file_tree']
        goal_sim = self.goal['goal_tree']
        self.state['previous_directory'] = self.state['current_directory']
        if intent in self.agent_tree_actions:
            self.apply_agent_tree_action(agent_action, f_sim)

        elif intent == 'Change_directory':
            self.state['current_directory'] = agent_action['new_directory']
        self.generate_next_focused_file()
        self.update_sub_goals(agent_action)
        if len(self.goal['sub_goal']) > 1:
            print("seems like it does surpass 1 :o :o")

        if intent in self.agent_tree_actions and not self.sub_goal_exists():
            self.add_random_sub_goal()

        found, total = f_sim.tree_similarity(goal_sim)
        self.state['previous_similarity'] = self.state['current_similarity']
        self.state['current_similarity'] = [found, total]

        return 1 if found == total else 0

    def generate_next_focused_file(self):
        f_sim = self.state['current_file_tree']
        result = f_sim.get_first_dissimilarity(self.goal['goal_tree'])
        # print('next was called, result is: ', result)
        if result is not None:
            f, m, d = result
            self.state['focused_file'] = {
                'file': f == 1,
                'map': m,
                'delete': not d  # found in current_file_tree but not in goal_tree
            }
        return result

    """
    File browser user intents:
    default: when user sim doesn't know how to respond
    end: user sim ends conversation
    inform: user simulator informs the agent about:
        file_name
        file_parent
        desire
    """

    def debug_bitmask(self, bitmask):
        self.debug = bitmask

    def debug_add(self, bitmask):
        self.debug |= bitmask

    def print_debug(self):
        if self.debug & self.CURRENT_TREE:
            print('DEBUG: Current file tree:')
            self.state['current_file_tree'].print_tree()
        if self.debug & self.GOAL_TREE:
            print('DEBUG: Goal tree:')
            self.goal['goal_tree'].print_tree()
        if self.debug & self.SUB_GOALS:
            print('DEBUG: Sub goals:')
            for goal in self.goal['sub_goal']:
                print(goal)
        if self.debug & self.GOAL_DIR:
            print('DEBUG: End directory:')
            print(self.goal['end_directory'])
        if self.debug & self.SIMILARITY:
            print('DEBUG: tree similarity')
            print(self.state['current_similarity'])

    def create_change_dir_desire(self, directory):
        pD = 0.5
        action = {'intent': self.Change_directory_desire}
        if random.uniform(0, 1) < pD:
            action['directory'] = directory
        return action

    def generate_tree_desire_intent(self):
        """
        generate an action related to tree creation
        :return (dict): the generated action
        """
        proba_file = 0.8 if 'proba_file' not in self.probas else self.probas['proba_file']
        proba_parent = 0.6 if 'proba_parent' not in self.probas else self.probas['proba_parent']
        proba_change_dir = 0.4 if 'proba_change_dir' not in self.probas else self.probas['proba_change_dir']

        def next_dir(origin, destination):
            """
            :param (str) origin: path origin
            :param (str) destination: path destination
            :return (str,list) : next directory to take and list of directories to get to destination in order
            """

            def paths(dirs):
                return ["/".join(dirs[:i + 1]) for i in range(len(dirs))]

            if origin[-1] == '/':
                origin = origin[:-1]
            if destination[-1] == '/':
                destination = destination[:-1]

            if origin == destination:
                return None
            origin = origin.split('/')
            destination = destination.split('/')

            if len(origin) > len(destination):
                return origin[-2], paths(origin)[len(destination)::-1]

            i = 0
            for o, d in zip(origin, destination[:len(origin)]):
                if o != d:
                    return origin[-2], paths(origin)[i::-1]
                i += 1

            if random.uniform(0, 1) < 0.5:  # returns next directory
                return destination[len(origin)], paths(destination)[len(origin):]
            # generate any index randomly between length origin and length destination (directories to get to
            # destination from origin)
            index = int(random.uniform(len(origin), len(destination) - 0.01))
            return destination[index], paths(destination)[len(origin):]

        # if tree already finished
        if self.state['current_similarity'][0] == self.state['current_similarity'][1]:
            result = next_dir(self.state['current_directory'], self.goal['end_directory'])
            if result is not None:
                directory, dirs = result
                return self.create_change_dir_desire(directory)

        if self.sub_goal_exists():
            return self.generate_sub_goal_intent()

        focused_file = self.state['focused_file']
        file_map = focused_file['map']

        if random.uniform(0, 1) < proba_change_dir:
            result = next_dir(self.state['current_directory'], file_map['tree_sim'].parent.path())

            if result is not None:
                directory, dirs = result
                self.add_change_directory_sub_goal(dirs)
                return self.create_change_dir_desire(directory)

        is_file = 0
        if focused_file['delete']:
            intent = self.Delete_file_desire
        else:
            intent = self.Create_file_desire
        if focused_file['file']:
            is_file = 1

        response = {'intent': intent}
        name = focused_file['map']['name']
        parent_name = focused_file['map']['tree_sim'].parent.name
        params = {'file_name': name, 'parent_directory': parent_name, 'is_file': is_file}

        if random.uniform(0, 1) > proba_file:
            del params['file_name']
        if random.uniform(0, 1) > proba_parent:
            del params['parent_directory']

        for k in params:
            response[k] = params[k]
        return response

    def _build_response(self, agent_action):
        response = self.generate_tree_desire_intent()
        return response



    def _inform_response(self, agent_action):
        if 'error' in agent_action:
            if random.random() < 0.4:
                return self.create_change_dir_desire(self.state['current_file_tree'].path())
        return self._build_response(agent_action)

    def _ask_response(self, agent_action):
        assert agent_action['intent'] == 'ask', 'intent is not "ask" in ask_response'
        asked_action = agent_action['action']
        response = self._ask_sub_goal(agent_action)
        if response is not None:
            return response
        if asked_action['intent'] in self.agent_tree_actions:
            file_sim_copy = self.state['current_file_tree'].copy()
            self.apply_agent_tree_action(asked_action, file_sim_copy)
            f, t = file_sim_copy.tree_similarity(self.goal['goal_tree'])
            pf, pt = self.state['current_similarity']
            # file_sim_copy.print_tree()
            # print('DEBUG ASK: COPY:',f,t,' CURRENT SIM: ',pf,pt)
            if t != 0 and f / t > pf / pt:
                return {'intent': self.confirm}
        # elif asked_action['intent'] == 'Change_directory':
        #     if not self.sub_goal_exists():
        #         return {'intent': self.deny}
        #     sub_goal = self.next_sub_goal()
        #     if sub_goal['name'] == 'Change_directory' \
        #             and asked_action['new_directory'] in sub_goal['dirs']:
        #         return {'intent': self.confirm}
        return {'intent': self.deny}

    def _req_response(self, agent_action):
        response = {'intent': self.u_inform}
        requested = agent_action['slot']
        focused_file = self.state['focused_file']['map']
        file_name = self.get_slot_from_sub_goal(requested)
        if file_name is not None:
            response['file_name'] = file_name
            return response
        if requested == 'file_name':
            response['file_name'] = focused_file['name']
        elif requested == 'parent_directory':
            if 'file_name' not in agent_action \
                    or focused_file['name'] == agent_action['file_name']:
                parent = focused_file['tree_sim'].parent
            else:
                name = agent_action['file_name']
                file = self.goal['goal_tree'].lookup_file_name(name)
                if file is None:
                    return self._default_response()
                f, m = file
                parent = m['tree_sim'].parent
            response['parent_directory'] = parent.name
        else:
            return self._default_response()
        return response


if __name__ == '__main__':
    chars = [chr(i) for i in range(ord('a'), ord('z') + 1)]
    random_name = ''.join([chars[random.randint(0, len(chars) - 1)] for i in range(4)])
    print(random_name)
