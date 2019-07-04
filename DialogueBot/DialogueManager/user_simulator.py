import random, copy
from DialogueManager.dialogue_config import FAIL, SUCCESS, NO_OUTCOME

class UserSimulator:
    """Simulates a real user, to train the agent with reinforcement learning."""
    end = 'end'
    default = 'default'
    def __init__(self, constants, ontology):
        """
        The constructor for UserSimulator. Sets dialogue config variables.

        Parameters:
            constants (dict): Dict of constants loaded from file
            ontology (rdflib.Graph): ontology graph
        """

        self.state = {}
        self.max_round = constants['run']['max_round_num']
        self.agent_possible_intents = constants['agent']['agent_actions']
        self.ontology = ontology
        # - start by setting all responses to default
        self.user_responses = dict((a,self._default_response) for a in self.agent_possible_intents)
        self.goal = None
        self.round = 0

    def generate_goal(self):
        return {}

    def reset(self, data):
        """
        Resets the user sim. by emptying the state and returning the initial action.

        Returns:
            dict: The initial action of an episode
        """
        #TODO reset state
        self.generate_goal()
        self.round = 0
        return self._return_init_action()

    def _return_init_action(self):
        """
        Returns the initial action of the episode.

        The initial action has an intent of request, required init. inform slots and a single request slot.

        Returns:
            dict: Initial user response
        """
        return self._default_response()

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

        done = False
        self.round += 1
        # First check round num, if equal to max then fail
        if self.round == self.max_round:
            done = True
            success = FAIL
            user_response = self._end_response()
        else:
            try:
                success = self.update_state(agent_action)
                if success:
                    user_response = self._end_response()
                else:
                    agent_intent = agent_action['intent']
                    assert agent_intent in self.user_responses, 'Not acceptable agent action'
                    user_response = self.user_responses[agent_intent](agent_action)
            except Exception:
                return self._default_response(),-5,False,False

        reward = self.reward_function(agent_action, success)

        return user_response, reward, done, True if success is 1 else False

    def reward_function(self,agent_action, success):
        return -1

    def update_state(self,agent_action):
        """
        :param (dict) agent_action: action of the dialogue manager
        :return (int) : 1 if success reached, 0 else wise
        """
        return NO_OUTCOME

    def _default_response(self, agent_action=None):
        response = {'intent':self.default}
        return response

    def _end_response(self, agent_action=None):
        response = {'intent':self.end}
        return response

