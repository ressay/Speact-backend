import random

import re


class DialogueError(Exception):
    def __init__(self, *args) -> None:
        super().__init__(*args)
        self.reward = -1

    def get_nlg_models(self):
        return []

    def turn_to_dict(self):
        return dict()

    def turn_to_text(self):
        models = self.get_nlg_models()
        model = random.choice(models)
        slots = self.turn_to_dict()
        for param in re.findall('<(.+?)>', model):
            model = model.replace('<'+param+'>', slots[param])
        return model


class FileNameExistsError(DialogueError):
    def __init__(self, path, name, type_of_existing_file, *args) -> None:
        super().__init__(*args)
        self.path = path
        self.name = name
        self.type_of_file = type_of_existing_file
        self.reward = -0.1

    def get_nlg_models(self):
        possible_outputs = [
            'file <name> already exists in <path>',
            'path <path> already contains <name>'
        ]
        return possible_outputs

    def turn_to_dict(self):
        return {
            'path': self.path,
            'name': self.name
        }


class FileDoesNotExist(DialogueError):
    def __init__(self, file_name, *args) -> None:
        super().__init__(*args)
        self.file_name = file_name

    def get_nlg_models(self):
        possible_outputs = [
            'file <name> does not exist!',
            'sorry, I could not find <name>'
        ]
        return possible_outputs

    def turn_to_dict(self):
        return {
            'name': self.file_name
        }


class RemoveCurrentDirError(DialogueError):
    def __init__(self, file_name, path, is_ancestor_of_current_dir, *args) -> None:
        super().__init__(*args)
        self.file_name = file_name
        self.is_ancestor = is_ancestor_of_current_dir
        self.current_path = path
        self.reward = -0.1

    def get_nlg_models(self):
        possible_outputs = [
            'could not perform the action because it would remove the current directory',
            'sorry but removing <name> would remove the current path',
            'I can nor remove <name> because it would remove <path>'
        ]
        return possible_outputs

    def turn_to_dict(self):
        return {
            'path': self.current_path,
            'name': self.file_name
        }


class MoveFileInsideItself(DialogueError):
    def __init__(self, file_name, path, dest_path, *args) -> None:
        super().__init__(*args)
        self.file_name = file_name
        self.path = path
        self.dest_path = dest_path
        self.reward = -0.1

    def get_nlg_models(self):
        possible_outputs = [
            'I cannot move <path> inside <dest>',
            '<path> is parent of <dest>, I can not move it inside <dest>',
            'the destination <dest> is inside <path>'
        ]
        return possible_outputs

    def turn_to_dict(self):
        return {
            'path': self.path,
            'name': self.file_name,
            'dest': self.dest_path
        }


class PuttingFileUnderRegularFile(DialogueError):
    def __init__(self, file_name, parent_path, *args) -> None:
        super().__init__(*args)
        self.file_name = file_name
        self.parent_path = parent_path
        self.reward = -0.1

    def get_nlg_models(self):
        possible_outputs = [
            '<path> is not a folder, I cannot put <name> under it',
            '<path> is a regular file, I could not put <name> in it'
        ]
        return possible_outputs

    def turn_to_dict(self):
        return {
            'path': self.parent_path,
            'name': self.file_name
        }


class RenamingFolderInCurrentPath(DialogueError):
    def __init__(self, file_name, path, *args) -> None:
        super().__init__(*args)
        self.file_name = file_name
        self.path = path
        self.reward = 2

    def get_nlg_models(self):
        possible_outputs = [
            "it's kind of confusing to rename <name> as it is in current path",
            'current path <path> contains <name>, I cannot rename it',
            'sorry, I cannot rename <name> since it is in current path'
        ]
        return possible_outputs

    def turn_to_dict(self):
        return {
            'path': self.path,
            'name': self.file_name
        }


if __name__ == '__main__':
    err = RenamingFolderInCurrentPath('khobz','~/khobz/bla')
    print(err.turn_to_text())
