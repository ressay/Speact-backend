import os
import sys

import shutil

sys.path.insert(0, "./DialogueBot")

from DialogueBot.DialogueManager.FileBrowserDM.nlg import Nlg_system
from action_transformer import transform_to_action

from flask import Flask, render_template, request
from flask_cors import CORS
import random
from flask import jsonify
from models_loader import dqn_agent, step_agent, SIM_DIR
from real_encoder_decoder_training import intent_tags_predict

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/req", methods=['GET'])
def get_bot_response():
    intent, tags, text = intent_tags_predict(str(request.args.get('text')))
    user_action = transform_to_action(intent, tags, text)
    print('user action')
    print(user_action)
    action = step_agent(user_action)
    print('agent action')
    print(action)
    nlg_sys = Nlg_system()
    output_text = 'well, nlg error'
    apply_action(action)
    # try:
    output_text = nlg_sys.get_sentence(action)
    # except Exception as e:
    #     print('error nlg222')
    #     print(e)
    files = dqn_agent.state_tracker.get_current_path_files()
    reg_files = [f[0] for f in files if f[1] == 'file']
    dirs = [d[0] for d in files if d[1] == 'directory']
    print(files)
    print('files: ', reg_files)
    print('directories: ', dirs)
    dqn_agent.state_tracker.print_tree()
    return jsonify(
        text=output_text,
        files=reg_files+dirs,
        parsed={
            'intent': intent,
            'slots': tags,
            # 'action_agent': action
        }
    )


@app.route("/api/asr", methods=['GET'])
def asr():
    return str(request.args.get('text') + ' for the watch')


def apply_action(agent_action):
    intent = agent_action['intent']
    if intent == 'Create_file':
        file_name = agent_action['file_name']
        path = agent_action['path'][2:]
        path = path+'/' if len(path) != 0 and path[-1] != '/' else path
        path = SIM_DIR + path
        if agent_action['is_file']:
            open(path+file_name, 'w')
        else:
            os.mkdir(path+file_name)
    if intent == 'Delete_file':
        file_name = agent_action['file_name']
        path = agent_action['path'][2:]
        path = path + '/' if len(path) != 0 and path[-1] != '/' else path
        path = SIM_DIR + path
        if os.path.isdir(path+file_name):
            shutil.rmtree(path+file_name)
        else:
            os.remove(path+file_name)
    if intent == 'Open_file':
        file_name = agent_action['file_name']
        path = agent_action['path'][2:]
        path = path + '/' if len(path) != 0 and path[-1] != '/' else path
        path = SIM_DIR + path
        filepath = path + file_name
        if not os.path.isdir(filepath):
            os.system('xdg-open ' + filepath)

    if intent == 'Move_file' or intent == 'Copy_file':
        file_name = agent_action['file_name']
        path = agent_action['origin'][2:]
        dest = agent_action['dest'][2:]
        path = path + '/' if len(path) != 0 and path[-1] != '/' else path
        dest = dest + '/' if len(dest) != 0 and dest[-1] != '/' else dest
        path = SIM_DIR + path
        filepath = path + file_name
        new_filepath = SIM_DIR + dest + file_name
        if intent == 'Move_file':
            os.system("mv " + filepath + ' ' + new_filepath)
        else:
            if os.path.isdir(filepath):
                os.system('cp -a ' + filepath + ' ' + new_filepath)
            else:
                os.system('cp ' + filepath + ' ' + new_filepath)

    if intent == 'Rename_file':
        file_name = agent_action['old_name']
        new_name = agent_action['new_name']
        path = agent_action['path'][2:]
        path = path + '/' if len(path) != 0 and path[-1] != '/' else path

        path = SIM_DIR + path
        filepath = path + file_name
        new_filepath = path + new_name
        os.system('mv ' + filepath + ' ' + new_filepath)

if __name__ == "__main__":
    app.run()
