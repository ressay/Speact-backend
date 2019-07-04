import random
from DialogueManager.FileBrowserDM.user_simulator import UserSimulatorFB
from DialogueManager.FileBrowserDM.intent_tracker import IntentTracker


class ErrorModelController:
    """Adds error to the user action."""

    def __init__(self, constants):
        """
        The constructor for ErrorModelController.

        Saves items in constants, etc.

        Parameters:
            constants (dict): Loaded constants in dict
        """

        self.slot_error_prob = constants['emc']['slot_error_prob']
        self.slot_error_mode = constants['emc']['slot_error_mode']  # [0, 3]
        self.intent_error_prob = constants['emc']['intent_error_prob']
        self.intents = UserSimulatorFB.usersim_intents
        # self.intents.remove(UserSimulatorFB.u_request)
        requirements = IntentTracker.intents_requirements
        self.intent_slots_map = dict((key, [k for k in requirements[key]])
                                     for key in requirements)
        self.untouchable_keys = ['is_file', 'slot']

    def infuse_error(self, frame):
        """
        Takes a semantic frame/action as a dict and adds 'error'.

        Given a dict/frame it adds error based on specifications in constants. It can either replace slot values,
        replace slot and its values, delete a slot or do all three. It can also randomize the intent.

        Parameters:
            frame (dict): format dict('intent': '', 'inform_slots': {}, 'request_slots': {}, 'round': int,
                          'speaker': 'User')
        """

        keys = [key for key in frame if key != 'intent' and key not in self.untouchable_keys]
        for key in keys:
            if random.random() < self.slot_error_prob:
                # if self.slot_error_mode == 0:  # replace the slot_value only
                #     self._slot_value_noise(key, frame)
                # elif self.slot_error_mode == 1:  # replace slot and its values
                #     self._slot_noise(key, frame)
                # elif self.slot_error_mode == 2:  # delete the slot
                #     self._slot_remove(key, frame)
                # else:  # Combine all three
                rand_choice = random.random()
                if rand_choice <= 0.33:
                    self._slot_value_noise(key, frame)
                elif 0.33 < rand_choice <= 0.66:
                    self._slot_noise(key, frame)
                else:
                    self._slot_remove(key, frame)
        if random.random() < self.intent_error_prob:  # add noise for intent level
            frame['intent'] = random.choice(self.intents)

    def _slot_value_noise(self, key, frame):
        """
        Selects a new value for the slot given a key and the dict to change.

        Parameters:
            key (string)
            frame (dict)
        """
        chars = [chr(i) for i in range(ord('a'), ord('z') + 1)]
        random_name = ''.join([chars[random.randint(0, len(chars) - 1)] for i in range(4)])
        frame[key] = random_name

    def _slot_noise(self, key, frame):
        """
        Replaces current slot given a key in the informs dict with a new slot and selects a random value for this new slot.

        Parameters:
            key (string)
            frame (dict)
        """
        value = frame[key]
        frame.pop(key)
        intent = frame['intent']
        if intent in self.intent_slots_map:
            random_slot = random.choice(self.intent_slots_map[intent])
        else:
            r_key = random.choice(list(self.intent_slots_map.keys()))
            random_slot = random.choice(self.intent_slots_map[r_key])
        if random_slot not in self.untouchable_keys:
            frame[random_slot] = value

    def _slot_remove(self, key, frame):
        """
        Removes the slot given the key from the informs dict.

        Parameters:
            key (string)
            frame (dict)
        """

        frame.pop(key)
