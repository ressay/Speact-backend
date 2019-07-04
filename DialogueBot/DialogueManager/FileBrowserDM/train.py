from DialogueManager.FileBrowserDM.agent import AgentFB
from DialogueManager.FileBrowserDM.user_simulator import UserSimulatorFB
import Ontologies.onto_fbrowser as fbrowser
import argparse, json



if __name__ == "__main__":
    # Can provide constants file path in args OR run it as is and change 'CONSTANTS_FILE_PATH' below
    # 1) In terminal: python train.py --constants_path "constants.json"
    # 2) Run this file as is
    parser = argparse.ArgumentParser()
    parser.add_argument('--constants_path', dest='constants_path', type=str, default='')
    args = parser.parse_args()
    params = vars(args)

    # Load constants json into dict
    CONSTANTS_FILE_PATH = 'constants.json'
    if len(params['constants_path']) > 0:
        constants_file = params['constants_path']
    else:
        constants_file = CONSTANTS_FILE_PATH

    with open(constants_file) as f:
        constants = json.load(f)


    # Load run constants
    run_dict = constants['run']
    USE_USERSIM = run_dict['usersim']
    WARMUP_MEM = run_dict['warmup_mem']
    NUM_EP_TRAIN = run_dict['num_ep_run']
    TRAIN_FREQ = run_dict['train_freq']
    MAX_ROUND_NUM = run_dict['max_round_num']
    SUCCESS_RATE_THRESHOLD = run_dict['success_rate_threshold']
    compress = True
    train_batch = True
    use_encoder = False
    one_hot = True

    # Init. Objects
    user = UserSimulatorFB(constants, fbrowser.graph)

    dqn_agent = AgentFB(1024, constants, train_batch, use_encoder, compress, one_hot)


def run_round():
    # 1) Agent takes action given state tracker's representation of dialogue (state)
    state = dqn_agent.get_state()

    agent_action_index, agent_action = dqn_agent.step()
    user_action, reward, done, success = user.step(agent_action)
    # if not done:
        # 4) Infuse error into semantic frame level of user action
        # emc.infuse_error(user_action)
    # 5) Update state tracker with user action
    dqn_agent.update_state_user_action(user_action)
    # state_tracker.update_state_user(user_action)
    # 6) Get next state and add experience
    next_state = dqn_agent.get_state()

    # next_state = state_tracker.get_state(done)
    dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)

    return reward, done, success


def train_run():
    """
    Runs the loop that trains the agent.
    Trains the agent on the goal-oriented chatbot task. Training of the agent's neural network occurs every episode that
    TRAIN_FREQ is a multiple of. Terminates when the episode reaches NUM_EP_TRAIN.
    """
    print('Training Started...')
    episode = 0
    period_reward_total = 0
    period_success_total = 0
    success_rate_best = 0.0
    while episode < NUM_EP_TRAIN:
        episode_reset()
        episode += 1
        # print('running episode:',episode)
        done = False
        # state = state_tracker.get_state()
        while not done:
            reward, done, success = run_round()
            period_reward_total += reward

        # print('success is: ',success)
        period_success_total += success

        # Train
        if episode % TRAIN_FREQ == 0:

            # Check success rate
            success_rate = period_success_total / TRAIN_FREQ
            avg_reward = period_reward_total / TRAIN_FREQ
            print('training after getting success_rate:', success_rate, " and avg_reward: ",avg_reward)

            # Update current best success rate
            if success_rate > success_rate_best:
                # print('Episode: {} NEW BEST SUCCESS RATE: {} Avg Reward: {}' .format(episode, success_rate, avg_reward))
                success_rate_best = success_rate
                dqn_agent.save_weights()
            period_success_total = 0
            period_reward_total = 0
            # Copy
            dqn_agent.copy()
            # Train
            dqn_agent.train()
            # Flush
            # if success_rate >= success_rate_best and success_rate >= SUCCESS_RATE_THRESHOLD:
            #     dqn_agent.empty_memory()
    print('...Training Ended')


def episode_reset():
    """
    Resets the episode/conversation in the warmup and training loops.
    Called in warmup and train to reset the state tracker, user and agent. Also get's the initial user action.
    """
    user_action = user.reset(dqn_agent.state_tracker.get_data())
    dqn_agent.reset(user_action)


# warmup_run()
train_run()

def simulate():
    done = False
    user_action = user.reset(dqn_agent.state_tracker.get_data())
    print('user: ', user_action)
    dqn_agent.reset(user_action)
    while not done:
        agent_action_index, agent_action = dqn_agent.step()
        print('agent: ',agent_action)
        user_action, reward, done, success = user.step(agent_action)
        print('user: ', user_action)
        dqn_agent.update_state_user_action(user_action)
