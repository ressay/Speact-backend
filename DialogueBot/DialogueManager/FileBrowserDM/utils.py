import inspect
import json
import os

root = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
constants = json.load(open(root+'/constants.json', 'r'))
agent = constants['agent']
agent_actions = agent['agent_actions']
run = constants['run']
emc = constants['emc']