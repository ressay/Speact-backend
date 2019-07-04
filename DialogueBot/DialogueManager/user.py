import json

from rdflib import Graph

from DialogueManager.FileBrowserDM.agent import AgentFB
from DialogueManager.FileBrowserDM.file_tree_sim import FileTreeSimulator
from DialogueManager.FileBrowserDM.nlg import Nlg_system
from DialogueManager.FileBrowserDM.state_tracker import StateTrackerFB
from DialogueManager.FileBrowserDM.user_simulator import UserSimulatorFB
import Ontologies.onto_fbrowser as fbrowser

class UserAgent:

    def __init__(self,user_sim=None,state_tracker=StateTrackerFB(1,fbrowser.graph)):
        """
        The constructor for User.
        :param (DialogueManager.user_simulator.UserSimulator) user_sim:
        """
        self.user_sim = user_sim
        self.state_tracker = state_tracker

    def return_response(self,user_action):
        """
        Asks user in console for response then receives a response as input.

        Format must be like this:

        Create_directory, file_name:dir1; path:~/
        Delete_file, file_name:dir1; path:~/
        Create_file, file_name:file1; path:~/; is_file:1
        request, slot:parent_directory
        ask, intent:Create_directory; file_name: dir1; path:~/
        ask, intent:Create_RegFile; file_name: file1; path:~/
        Change_directory, new_directory:~/dir1

        intents, informs keys and values, and request keys and values cannot contain / , :

        Returns:
            dict: The response of the user
        """
        self.state_tracker.update_state_user_action(user_action,False)
        print("possible actions:")
        nodes,actions = self.state_tracker.get_possible_actions()
        # actions = actions[::2]
        for i, a in enumerate(actions):
            print(i+1,': '),
            print(a)
        # a,an = self.state_tracker.get_possible_actions()

        # print(a)
        input_string = input('Response: ')
        response = actions[int(input_string)-1]
        # input_string = input_string.replace(' ','')
        # intent,input_string = input_string.split(',')
        # arguments = input_string.split(';')
        # response = {'intent':intent}
        # if intent == 'ask':
        #     r = {}
        #     for arg in arguments:
        #         key,val = arg.split(':')
        #         r[key] = val
        #     response['action'] = r
        #     return response
        # for arg in arguments:
        #     key,val = arg.split(':')
        #     response[key] = val
        self.state_tracker.update_state_agent_action(response,False)
        return response

    def start_conversation(self):
        assert self.user_sim, 'User simulator not defined'
        mask = UserSimulatorFB.CURRENT_TREE | UserSimulatorFB.SIMILARITY | UserSimulatorFB.GOAL_DIR \
        | UserSimulatorFB.SUB_GOALS
        self.user_sim.debug_add(mask)
        response = self.user_sim.reset(self.state_tracker.get_data())
        print(response)
        while True:
            response, reward, done, success = self.user_sim.step(self.return_response(response))
            print(response)
            print('reward is:',reward)
            print("************state tracker tree*************")
            self.state_tracker.print_tree()

            if done:
                print('end of conversation with'),
                if success:
                    print('success')
                else:
                    print('failure')
                print('******')
                print('starting new conversation')
                self.state_tracker = StateTrackerFB(1, fbrowser.graph)
                response = self.user_sim.reset(self.state_tracker.get_data())

                print(response)

class UserHuman(object):
    """Connects a real user to the conversation through the console."""

    def __init__(self, directory='/home/ressay/workspace/PFEM2/DialogueBot/Simulation'):
        """
        The constructor for User.
        :param (DialogueManager.user_simulator.UserSimulator) user_sim:
        """

        CONSTANTS_FILE_PATH = 'FileBrowserDM/constants.json'
        constants_file = CONSTANTS_FILE_PATH

        with open(constants_file) as f:
            constants = json.load(f)
            constants['agent']['load_weights_file_path'] = 'FileBrowserDM/my_weights/m50.h5'
        compress = True
        train_batch = True
        use_encoder = False
        one_hot = True
        self.dqn_agent = AgentFB(50, constants,train_batch, use_encoder, compress, one_hot)
        self.directory = directory

    def return_response(self,agent_action=None):
        """
        Asks user in console for response then receives a response as input.

        Format must be like this:

        Delete_file_desire, file_name:dir1; parent_directory:dir1; is_file:1
        Create_file_desire, file_name:file1; parent_directory:dir1; is_file:1
        request, slot:parent_directory
        Change_directory_desire, directory:dir1
        request, slot:directory; file_name:dddd
        intents, informs keys and values, and request keys and values cannot contain / , :

        Returns:
            dict: The response of the user
        """
        input_string = input('Response: ')
        input_string = input_string.replace(' ','')
        intent,input_string = input_string.split(',')
        arguments = input_string.split(';')
        response = {'intent': intent}
        for arg in arguments:
            key,val = arg.split(':')
            response[key] = val
        return response

    def reset(self,first_action):
        tree = FileTreeSimulator.read_existing_dirs(directory=self.directory)
        data = {'current_tree_sim': tree, 'tree_sim': tree}
        self.dqn_agent.eps = 0
        self.dqn_agent.reset(first_action,data)


    def start_conversation(self):
        nlg_sys = Nlg_system()
        response = self.return_response()
        self.reset(response)
        while True:
            self.dqn_agent.state_tracker.print_tree()
            print('TRIPLETS:')
            for t in self.dqn_agent.state_tracker.all_episode_triplets:
                print(t, self.dqn_agent.state_tracker.graph.get_decoded_triplet(t))
            _, agent_action = self.dqn_agent.step()
            print(nlg_sys.get_sentence(agent_action))
            response = self.return_response(agent_action)
            self.dqn_agent.update_state_user_action(response)

if __name__ == '__main__':

    c = json.load(open('FileBrowserDM/constants.json', 'r'))
    # user_sim = UserSimulatorFB(c,Graph())
    # agent = UserAgent(user_sim)
    # agent.start_conversation()
    user = UserHuman()
    user.start_conversation()


