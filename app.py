import sys
sys.path.insert(0, "./DialogueBot")

from DialogueBot.DialogueManager.FileBrowserDM.nlg import Nlg_system
from action_transformer import transform_to_action

from flask import Flask, render_template, request
from flask_cors import CORS
import random
from flask import jsonify
from models_loader import dqn_agent, step_agent
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

    try:
        output_text = nlg_sys.get_sentence(action)
    except:
        print('error nlg')
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
    pass

if __name__ == "__main__":
    app.run()
