intent_maps = {
    'create_file_desire': 'Create_file_desire',
    'create_directory_desire': 'Create_file_desire',
    'delete_file_desire': 'Delete_file_desire',
    'delete_directory_desire': 'Delete_file_desire',
    "open_file_desire": 'Open_file_desire',
    "close_file_desire": 'unknown',
    "copy_file_desire": 'Copy_file_desire',
    "move_file_desire": 'Move_file_desire',
    "rename_file_desire": 'Rename_file_desire',
    "change_directory_desire": 'Change_directory_desire',
}
slot_map = {
    'new_directory': 'directory',
    'directory_name': 'file_name'
}
# None means remove for all slots
# remove_prefix_only = {
#     'create_file_desire': None,
#     'create_directory_desire': None,
#     'delete_file_desire': None,
#     'delete_directory_desire': None,
#     "open_file_desire": None,
#     "copy_file_desire": ["ALTER.file_name"],
#     "move_file_desire": ["ALTER.file_name"],
#     "inform": None,
#     "request": None,
# }
add_slot_from_intent = {
    'create_file_desire': {'is_file': 1},
    'create_directory_desire': {'is_file': 0},
    'request': {'slot': 'directory'}
}


def transform_to_action(intent, slots, text):
    action = {}
    intent = intent[0]
    action['intent'] = intent if intent not in intent_maps else intent_maps[intent]
    slot_set = set(slots)
    for slot in slot_set:
        if slot == 'NUL':
            continue
        slot_value = ' '.join([t for s, t in zip(slots, text) if s == slot])
        if '.' in slot:
            slot = slot.split('.')[1]  # remove prefix
        if slot in slot_map:
            slot = slot_map[slot]
        action[slot] = slot_value
    if intent in add_slot_from_intent:
        for key in add_slot_from_intent[intent]:
            action[key] = add_slot_from_intent[intent][key]
    return action
