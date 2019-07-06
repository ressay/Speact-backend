import random
import re

class Nlg_system(object):
    def __init__(self) -> None:
        super().__init__()
        self.models = {
            'ask': [
                "Should I <action>?",
                "Do you want me to <action>"
            ],
            'request': {
                'file_not_found': [
                    "I'm sorry, I couldn't find <special_file_name>, maybe you meant something else?",
                    "Sorry, Could you please repeat? I could not find <special_file_name>",
                    "I couldn't find <special_file_name>, did you mean something else? sorry for the inconvenience"
                ],
                'file_name': [
                    "What's the file's name?",
                    "Please give me the file's name",
                    "Can you give me the file's name please?",
                    "What is the name of the file?"
                ],
                'old_name': [
                    "What's the file's name?",
                    "Please, give me the file's name",
                    "Can you give me the file's name please?",
                    "What is the name of the file?"
                ],
                'new_name': [
                    "What should I change the name to?",
                    "Please, tell me what's <old_name>'s new name?",
                    "Can you give me <old_name>'s new name?",
                    "Please, what should I change <old_name> to?"
                ],
                'directory': [
                    "What's the directory's name?",
                    "Please give me the directory's name",
                    "Can you give me the directory's name, please?",
                    "What is the name of the directory?"
                ],
                'parent_directory': [
                    "What's <file_name>'s parent directory?",
                    "Please give me <file_name>'s parent directory",
                    "Can you give me the parent directory of <file_name> please?",
                    "What is the directory of <file_name>?",
                    "Where is <file_name> located?"
                ],
                'origin': [
                    "What's <file_name>'s parent directory?",
                    "Please give me <file_name>'s parent directory",
                    "Can you give me the parent directory of <file_name> please?",
                    "What is the directory of <file_name>?",
                    "Where is <file_name> located?"
                ],
                'multiple_file_found': [
                    "I found many files named <file_name>, could you please tell me what's its parent directory?",
                    "I found many files '<file_name>', Please give me its parent directory?",
                    "Sorry I can't tell which '<file_name>' you meant, could you tell me where is it located?"
                ],
                'dest': [
                    "Where do you want me to put <file_name>?",
                    "Please give me <file_name>'s destination",
                    "Can you please provide <file_name>'s destination?",
                    "To which destination?",
                    "Where should I put <file_name>?",
                    "Where to?"
                ]
            },
            'Create_file': [
                '<file_type> <file_name> has been created!',
                'I created <file_type> <file_name> in <path>',
                '<path> now contains <file_name>!',
                '<file_type> <file_name> has been created under <path>',
                'I created a <file_type> <file_name>',
                'I created <file_type> <file_name> in <path>'
            ],
            'Delete_file': [
                '<file_type> <file_name> has been removed!',
                'I deleted <file_type> <file_name> from <path>',
                '<file_type> <file_name> has been deleted from <path>',
                'I removed <file_name>',
                'I removed <file_type> <file_name> from <path>'
            ],
            'Move_file': [
                'I moved <file_name> from <origin> to <dest>',
                '<file_name> has been moved from <origin> to <dest>',
                'I moved <file_name> to <dest>',
            ],
            'Copy_file': [
                'I copied <file_name> from <origin> to <dest>',
                '<file_name> has been copied from <origin> to <dest>',
                'I copied <file_name> to <dest>',
            ],
            'Change_directory': [
                'Changed directory to <new_directory>',
                'I moved to <new_directory>',
                "We're in <new_directory> now!",
                "I moved to path <new_directory>"
            ],
            'inform': {
                'paths': [
                    'I did find it under several paths: <paths>'
                ],
                'path': [
                    'I found it under <path>'
                ],
                'nopath': [
                    "Sorry I couldn't find <file_name>"
                ],
                'error': {
                    'file_name_exists': [

                    ],
                    'file_does_not_exist': [

                    ],
                    'remove_current_dir': [

                    ],
                    'move_file_inside_itself': [

                    ],
                    'putting_file_under_regfile': [

                    ],
                    'renaming_in_current_dir': [

                    ]
                }
            },
            'Open_file': [
                '<file_name> has been opened!',
                'I opened <file_name>',
                'I opened <file_name>',
                'Here is <file_name>!'
            ],
            'Rename_file': [
                "<old_name> has been changed to <new_name>!",
                "I changed <old_name> to <new_name>",
                "<old_name> is now <new_name>!",
                "the file's name has been changed to <new_name>",
                "I changed it to <new_name>"
            ],
            'default': [
                "Sorry, can you repeat, I did not understand",
                "Hmmm, I failed to understand",
                "Sorry?",
                "Could you please repeat? I did not understand"
            ]
        }
        self.expression_gen = {
            'file_name': [self.value_replacer],
            'new_directory': [self.value_replacer],
            'dest': [self.value_replacer],
            'origin': [self.value_replacer],
            'file_type': [self.value_replacer],
            'path': [self.value_replacer],
            'paths': [self.paths_expression],
            'special_file_name': [self.value_replacer],
            'old_name': [self.value_replacer],
            'new_name': [self.value_replacer],
        }
        self.actions = {
            'Create_file': [
                'create <file_type> <file_name> under <path>'
            ],
            'Delete_file': [
                'delete <file_type> <file_name> from <path>'

            ],
            'Move_file': [
                'move <file_name> from <origin> to <dest>'
            ],
            'Copy_file': [
                'copy <file_name> from <origin> to <dest>'
            ],
            'Change_directory': [
                'change directory to <new_directory>'
            ]
        }

    def value_replacer(self, value):
        return value

    def paths_expression(self, value):
        result = value[0]
        for val in value[1:]:
            result += " and " + val
        return result

    def action_expression(self, action):
        models = self.actions[action['intent']]
        result = []
        for model in models:
            for param in re.findall('<(.+?)>', model):
                expressions = self.get_expressions(param, action[param])
                e = self.choose_random(expressions)
                model = model.replace('<'+param+'>', e)
            result.append(model)
        return result

    def get_expressions(self, key, value):
        if key == 'action':
            return self.action_expression(value)
        tab = self.expression_gen[key]
        return [func(value) for func in tab]

    def get_models(self, agent_action):
        intent = agent_action['intent']
        if intent == 'request':
            if 'special' in agent_action:
                return self.models[intent][agent_action['special']]
            return self.models[intent][agent_action['slot']]
        if intent == 'inform':
            if 'paths' in agent_action:
                paths = agent_action['paths']
                if len(paths) > 1:
                    return self.models[intent]['paths']
                if len(paths) == 1:
                    agent_action['path'] = paths[0]
                    return self.models[intent]['path']
                return self.models[intent]['nopath']

        return self.models[intent]

    def choose_random(self,tab):
        return tab[random.randint(0,len(tab)-1)]

    def fix_slots(self, agent_action):
        if 'file_name' not in agent_action and 'old_name' in agent_action:
            agent_action['file_name'] = agent_action['old_name']

    def get_sentence(self, agent_action):
        if agent_action['intent'] == 'inform' and 'error' in agent_action:
            err = agent_action['error']
            return err.turn_to_text()
        if agent_action['intent'] == 'ask' and agent_action['action']['intent'] in ('inform', 'request','default'):
            agent_action['intent'] = 'default'
        self.fix_slots(agent_action)
        models = self.get_models(agent_action)
        # print(models)
        model = self.choose_random(models)
        for param in re.findall('<(.+?)>',model):
            expressions = self.get_expressions(param,agent_action[param])
            e = self.choose_random(expressions)
            model = model.replace('<'+param+'>', e)
        return model


if __name__ == '__main__':
    nlg = Nlg_system()
    print(nlg.get_sentence({'intent': 'inform', 'file_name': 'khobzish',
                            'paths': ['home/bla','home/blo']}))
    print(nlg.get_sentence({'intent': 'Change_directory',
                            'new_directory': 'esta/lavida/baby'}))
    print(nlg.get_sentence({'intent': 'ask', 'action': {'intent': 'Create_file', 'file_name': 'khobz',
                            'path': 'eso/es', 'file_type': 'file'}}))

