import json
from DialogueBot.DialogueManager.FileBrowserDM.file_tree_sim import FileTreeSimulator
from DialogueBot.DialogueManager.FileBrowserDM.agent import AgentFB

CONSTANTS_FILE_PATH = 'resources/constants.json'
constants_file = CONSTANTS_FILE_PATH

with open(constants_file) as f:
    constants = json.load(f)
    constants['agent']['load_weights_file_path'] = 'models/m50.h5'
compress = True
train_batch = True
use_encoder = False
one_hot = True
print('loading agent...')
dqn_agent = AgentFB(50, constants,train_batch, use_encoder, compress, one_hot)
print('agent loaded!')
first = True

def reset_agent(directory=None):
    if directory is None:
        directory = '/home/ressay/workspace/PFEM2/DialogueBot/Simulation'
    tree = FileTreeSimulator.read_existing_dirs(directory=directory)
    tree.print_tree()
    data = {'current_tree_sim': tree, 'tree_sim': tree}
    # user_action = {'intent': 'Open_file_desire', 'file_name': 'second'}
    dqn_agent.reset_data(data)
    dqn_agent.eps = 0
    # print(dqn_agent.step_user_action(user_action))

def step_agent(user_action):
    print('user action before step')
    print(user_action)
    _, agent_action = dqn_agent.step_user_action(user_action)
    return agent_action

def try_agent():
    try_ag = AgentFB(50, constants, train_batch, use_encoder, compress, one_hot)
    directory = '/home/ressay/workspace/PFEM2/DialogueBot/Simulation'
    tree = FileTreeSimulator.read_existing_dirs(directory=directory)
    data = {'current_tree_sim': tree, 'tree_sim': tree}
    try_ag.dqn_agent.eps = 0
    first_action = {'intent': 'Open_file_desire'}
    try_ag.reset(first_action, data)


reset_agent()



