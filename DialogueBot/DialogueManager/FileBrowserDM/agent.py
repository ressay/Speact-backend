import json
from DialogueManager.FileBrowserDM.state_tracker import StateTrackerFB
from DialogueManager.agent import Agent
import Ontologies.onto_fbrowser as fbrowser


class AgentFB(Agent):
    def __init__(self, state_size, constants, train_by_batch=True,
                 use_graph_encoder=True,compress_state=False, one_hot=True,data=None) -> None:
        super().__init__(state_size, constants, train_by_batch,
                         use_graph_encoder, compress_state, one_hot,data)

    def init_state_tracker(self,data):
        return StateTrackerFB(self.encoder_size, fbrowser.graph, one_hot=self.one_hot,data=data)


if __name__ == '__main__':
    c = json.load(open('constants.json', 'r'))
    agent = AgentFB(10, c)
    # agent._build_model()
