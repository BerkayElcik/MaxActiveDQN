#import gym
import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

import numpy as np
from double_dqn_agent import DoubleDQNAgent
from utils import plot_learning_curve
#from gym import wrappers
from gymnasium import wrappers
from terahertz_drone_environment import thz_drone_env
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #env = make_env('CUSTOM_ENV(not_ready)')
    #env = gym.make('CUSTOM_ENV(not_ready)')

    no_of_channels=3000
    no_of_actions=700

    env=thz_drone_env(n_channels=no_of_channels, P_T=1, freq_of_movement=0.1, no_of_actions=no_of_actions)

    env = FlattenObservation(env)


    best_score = -np.inf
    load_checkpoint = False
    #n_games = 55
    #n_games=1000
    n_games=40000
    #n_games=100 #for testing and debugging

    """
    agent = DQNAgent(gamma=0.99, epsilon=1, lr=0.0001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.1,
                     batch_size=32, replace=1000, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DQNAgent',
                     env_name='PongNoFrameskip-v4')
    """

    #num_actions = np.prod(env.action_space.nvec)
    num_actions = env.action_space.n



    #print(np.random.choice(env.action_space))
    #print(env.action_space.sample())
    #print(env.observation_space.shape)
    #print(env.action_space.shape)
    print(env.observation_space.shape[0])
    print(env.observation_space.shape)
    print(env.observation_space)
    #print(env.action_space.nvec)
    print(env.action_space.n)
    print(env.action_space)
    print(num_actions)
    #print(env.action_space.sample())
    #print(env.observation_space.sample())
    #print(env.observation_space.sample().size)

    starting_eps=1
    ending_eps=0.01
    #no_activity=4500
    #no_activity=200
    no_activity=10
    #no_activity=10 #for testing and debugging
    n_training_steps=n_games*no_activity
    time_taken_to_decrease_epsilon=0.6

    dec_step_of_epsilon=(starting_eps-ending_eps)/(n_training_steps*time_taken_to_decrease_epsilon)

    agent = DoubleDQNAgent(input_dims=env.observation_space.shape[0],
                     n_actions=num_actions,
                     epsilon=starting_eps,
                     mem_size=2000,
                     eps_min=ending_eps,
                     eps_dec=dec_step_of_epsilon,
                     chkpt_dir='models/', algo='DoubleDQNAgent',
                     env_name='THz_channel_selection')



    if load_checkpoint:
        agent.load_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    # if you want to record video of your agent playing, do a mkdir tmp && mkdir tmp/dqn-video
    # and uncomment the following 2 lines.
    #env = wrappers.Monitor(env, "tmp/dqn-video",
    #                    video_callable=lambda episode_id: True, force=True)
    n_steps = 0

    scores, eps_history, steps_array = [], [], []
    Capacity_graph = []
    no_of_channels_graph = []
    action_index_graph=[]
    snr_threshold_graph=[]


    distance_1_graph_scores, distance_2_graph_scores, distance_3_graph_scores, distance_4_graph_scores, distance_5_graph_scores, \
    distance_6_graph_scores, distance_7_graph_scores, distance_8_graph_scores, distance_9_graph_scores, distance_10_graph_scores, \
    distance_11_graph_scores = [], [], [], [], [], [], [], [], [], [], []


    distance_1_graph_capacity, distance_2_graph_capacity, distance_3_graph_capacity, distance_4_graph_capacity, distance_5_graph_capacity, \
    distance_6_graph_capacity, distance_7_graph_capacity, distance_8_graph_capacity, distance_9_graph_capacity, distance_10_graph_capacity, \
    distance_11_graph_capacity = [], [], [], [], [], [], [], [], [], [], []


    distance_1_graph_no_of_channels, distance_2_graph_no_of_channels, distance_3_graph_no_of_channels, distance_4_graph_no_of_channels, \
    distance_5_graph_no_of_channels, distance_6_graph_no_of_channels, distance_7_graph_no_of_channels, distance_8_graph_no_of_channels, \
    distance_9_graph_no_of_channels, distance_10_graph_no_of_channels, distance_11_graph_no_of_channels = [], [], [], [], [], [], [], [], [], [], []

    distance_1_graph_action_index, distance_2_graph_action_index, distance_3_graph_action_index, distance_4_graph_action_index, \
    distance_5_graph_action_index, distance_6_graph_action_index, distance_7_graph_action_index, distance_8_graph_action_index, \
    distance_9_graph_action_index, distance_10_graph_action_index, distance_11_graph_action_index = [], [], [], [], [], [], [], [], [], [], []

    distance_1_graph_snr_threshold, distance_2_graph_snr_threshold, distance_3_graph_snr_threshold, distance_4_graph_snr_threshold, \
    distance_5_graph_snr_threshold, distance_6_graph_snr_threshold, distance_7_graph_snr_threshold, distance_8_graph_snr_threshold, \
    distance_9_graph_snr_threshold, distance_10_graph_snr_threshold, distance_11_graph_snr_threshold = [], [], [], [], [], [], [], [], [], [], []

    print("Started Training")
    n_activity=no_activity
    num_training_steps=n_games*n_activity
    progress_bar = tqdm(range(num_training_steps))

    for i in range(n_games):
        done = False
        observation, info = env.reset()


        #print(observation[no_of_channels]) #distance value

        """
        print("obsmain")
        print(observation)
        print(len(observation))
        print(info)
        """

        Score = 0
        #while not done:
        for i in range(n_activity):
            action, action_index = agent.choose_action(observation)
            observation_, reward, terminated, truncated, info = env.step(action)



            """
            print("---------------------OBSERVATION---------------------")
            print(observation)
            print(observation.shape)
            """

            """
            print("---------------------ACTION---------------------")
            print(action)
            """

            """
            print("---------------------NEXT OBSERVATION---------------------")
            print(observation_)
            """

            """
            print("---------------------REWARD---------------------")
            print(reward)
            """


            """
            print("---------------------INFO---------------------")
            print(info)
            """

            Score += reward
            """
            print("---------------------SCORE---------------------")
            print(Score)
            """



            if not load_checkpoint:
                agent.store_transition(observation, action, action_index,
                                     reward, observation_, done)
                agent.learn()
            observation = observation_
            n_steps += 1
            progress_bar.update(1)



        scores.append(Score)
        steps_array.append(n_steps)
        Capacity_graph.append(info["capacity"])
        no_of_channels_graph.append(info["no_of_channels"])
        action_index_graph.append(info["action_index"])
        snr_threshold_graph.append(info["snr_threshold"])

        if info["distance"] == 1:
            distance_1_graph_scores.append(Score)
            distance_1_graph_capacity.append(info["capacity"])
            distance_1_graph_no_of_channels.append(info["no_of_channels"])
            distance_1_graph_action_index.append(info["action_index"])
            distance_1_graph_snr_threshold.append(info["snr_threshold"])
        elif info["distance"] == 2:
            distance_2_graph_scores.append(Score)
            distance_2_graph_capacity.append(info["capacity"])
            distance_2_graph_no_of_channels.append(info["no_of_channels"])
            distance_2_graph_action_index.append(info["action_index"])
            distance_2_graph_snr_threshold.append(info["snr_threshold"])
        elif info["distance"] == 3:
            distance_3_graph_scores.append(Score)
            distance_3_graph_capacity.append(info["capacity"])
            distance_3_graph_no_of_channels.append(info["no_of_channels"])
            distance_3_graph_action_index.append(info["action_index"])
            distance_3_graph_snr_threshold.append(info["snr_threshold"])
        elif info["distance"] == 4:
            distance_4_graph_scores.append(Score)
            distance_4_graph_capacity.append(info["capacity"])
            distance_4_graph_no_of_channels.append(info["no_of_channels"])
            distance_4_graph_action_index.append(info["action_index"])
            distance_4_graph_snr_threshold.append(info["snr_threshold"])
        elif info["distance"] == 5:
            distance_5_graph_scores.append(Score)
            distance_5_graph_capacity.append(info["capacity"])
            distance_5_graph_no_of_channels.append(info["no_of_channels"])
            distance_5_graph_action_index.append(info["action_index"])
            distance_5_graph_snr_threshold.append(info["snr_threshold"])
        elif info["distance"] == 6:
            distance_6_graph_scores.append(Score)
            distance_6_graph_capacity.append(info["capacity"])
            distance_6_graph_no_of_channels.append(info["no_of_channels"])
            distance_6_graph_action_index.append(info["action_index"])
            distance_6_graph_snr_threshold.append(info["snr_threshold"])
        elif info["distance"] == 7:
            distance_7_graph_scores.append(Score)
            distance_7_graph_capacity.append(info["capacity"])
            distance_7_graph_no_of_channels.append(info["no_of_channels"])
            distance_7_graph_action_index.append(info["action_index"])
            distance_7_graph_snr_threshold.append(info["snr_threshold"])
        elif info["distance"] == 8:
            distance_8_graph_scores.append(Score)
            distance_8_graph_capacity.append(info["capacity"])
            distance_8_graph_no_of_channels.append(info["no_of_channels"])
            distance_8_graph_action_index.append(info["action_index"])
            distance_8_graph_snr_threshold.append(info["snr_threshold"])
        elif info["distance"] == 9:
            distance_9_graph_scores.append(Score)
            distance_9_graph_capacity.append(info["capacity"])
            distance_9_graph_no_of_channels.append(info["no_of_channels"])
            distance_9_graph_action_index.append(info["action_index"])
            distance_9_graph_snr_threshold.append(info["snr_threshold"])
        elif info["distance"] == 10:
            distance_10_graph_scores.append(Score)
            distance_10_graph_capacity.append(info["capacity"])
            distance_10_graph_no_of_channels.append(info["no_of_channels"])
            distance_10_graph_action_index.append(info["action_index"])
            distance_10_graph_snr_threshold.append(info["snr_threshold"])
        elif info["distance"] == 11:
            distance_11_graph_scores.append(Score)
            distance_11_graph_capacity.append(info["capacity"])
            distance_11_graph_no_of_channels.append(info["no_of_channels"])
            distance_11_graph_action_index.append(info["action_index"])
            distance_11_graph_snr_threshold.append(info["snr_threshold"])

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,'Capacity: ', info["capacity"], 'Score: ', Score,
             ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
            'epsilon %.2f' % agent.epsilon, 'steps', n_steps, "no of channels:", info['no_of_channels'])


        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score


        eps_history.append(agent.epsilon)

    if not load_checkpoint:
        agent.save_models()

    """

    # Plot for distance_1_graph_capacity
    figure_file = 'plots/' + "distance_1_graph_capacity" + '.png'
    fig1_capacity, ax1_capacity = plt.subplots()
    abscissa_len_1_capacity = len(distance_1_graph_capacity)
    abscissa_1_capacity = np.arange(abscissa_len_1_capacity)
    ax1_capacity.scatter(abscissa_1_capacity, distance_1_graph_capacity, color="C1")
    ax1_capacity.axes.get_xaxis().set_visible(False)
    ax1_capacity.yaxis.tick_right()
    ax1_capacity.set_ylabel("Capacity", color="C1")
    ax1_capacity.yaxis.set_label_position('left')
    ax1_capacity.tick_params(axis='y', colors="C1")
    fig1_capacity.savefig(figure_file)
    plt.close()



    # Plot for distance_1_graph_capacity_running_average
    figure_file = 'plots/' + "distance_1_graph_capacity_running_average" + '.png'
    fig1_capacity_running_average, ax1_capacity_running_average = plt.subplots()
    abscissa_len_1_capacity_running_average = len(distance_1_graph_capacity)
    abscissa_1_capacity_running_average = np.arange(abscissa_len_1_capacity_running_average)
    running_avg_1_capacity = np.empty(abscissa_len_1_capacity_running_average)
    for t in range(abscissa_len_1_capacity_running_average):
        running_avg_1_capacity[t] = np.mean(distance_1_graph_capacity[max(0, t - 20):(t + 1)])
    ax1_capacity_running_average.scatter(abscissa_1_capacity_running_average, running_avg_1_capacity, color="C1")
    ax1_capacity_running_average.axes.get_xaxis().set_visible(False)
    ax1_capacity_running_average.yaxis.tick_right()
    ax1_capacity_running_average.set_ylabel("Capacity", color="C1")
    ax1_capacity_running_average.yaxis.set_label_position('left')
    ax1_capacity_running_average.tick_params(axis='y', colors="C1")
    fig1_capacity_running_average.savefig(figure_file)
    plt.close()


    # Plot for distance_1_graph_scores
    figure_file = 'plots/' + "distance_1_graph_scores" + '.png'
    fig1_scores, ax1_scores = plt.subplots()
    abscissa_len_1_scores = len(distance_1_graph_scores)
    abscissa_1_scores = np.arange(abscissa_len_1_scores)
    ax1_scores.scatter(abscissa_1_scores, distance_1_graph_scores, color="C1")
    ax1_scores.axes.get_xaxis().set_visible(False)
    ax1_scores.yaxis.tick_right()
    ax1_scores.set_ylabel("Score", color="C1")
    ax1_scores.yaxis.set_label_position('left')
    ax1_scores.tick_params(axis='y', colors="C1")
    fig1_scores.savefig(figure_file)
    plt.close()



    # Plot for distance_1_graph_scores_running_average
    figure_file = 'plots/' + "distance_1_graph_scores_running_average" + '.png'
    fig1_scores_running_average, ax1_scores_running_average = plt.subplots()
    abscissa_len_1_scores_running_average = len(distance_1_graph_scores)
    abscissa_1_scores_running_average = np.arange(abscissa_len_1_scores_running_average)
    running_avg_1_scores = np.empty(abscissa_len_1_scores_running_average)
    for t in range(abscissa_len_1_scores_running_average):
        running_avg_1_scores[t] = np.mean(distance_1_graph_scores[max(0, t - 20):(t + 1)])
    ax1_scores_running_average.scatter(abscissa_1_scores_running_average, running_avg_1_scores, color="C1")
    ax1_scores_running_average.axes.get_xaxis().set_visible(False)
    ax1_scores_running_average.yaxis.tick_right()
    ax1_scores_running_average.set_ylabel("Scores", color="C1")
    ax1_scores_running_average.yaxis.set_label_position('left')
    ax1_scores_running_average.tick_params(axis='y', colors="C1")
    fig1_scores_running_average.savefig(figure_file)
    plt.close()




    # Repeat for distance_2_graph_capacity
    figure_file = 'plots/' + "distance_2_graph_capacity" + '.png'
    abscissa_len = len(distance_2_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_2_graph_capacity, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_2_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_2_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_2_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_2_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_2_graph_scores
    figure_file = 'plots/' + "distance_2_graph_scores" + '.png'
    abscissa_len = len(distance_2_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_2_graph_scores, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_2_graph_scores_moving_average
    figure_file = 'plots/' + "distance_2_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_2_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_2_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Repeat for distance_3_graph_capacity
    figure_file = 'plots/' + "distance_3_graph_capacity" + '.png'
    abscissa_len = len(distance_3_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_3_graph_capacity, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_3_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_3_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_3_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_3_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_3_graph_scores
    figure_file = 'plots/' + "distance_3_graph_scores" + '.png'
    abscissa_len = len(distance_3_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_3_graph_scores, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_3_graph_scores_moving_average
    figure_file = 'plots/' + "distance_3_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_3_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_3_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Repeat for distance_4_graph_capacity
    figure_file = 'plots/' + "distance_4_graph_capacity" + '.png'
    abscissa_len = len(distance_4_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_4_graph_capacity, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_4_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_4_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_4_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_4_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_4_graph_scores
    figure_file = 'plots/' + "distance_4_graph_scores" + '.png'
    abscissa_len = len(distance_4_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_4_graph_scores, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_4_graph_scores_moving_average
    figure_file = 'plots/' + "distance_4_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_4_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_4_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Repeat for distance_5_graph_capacity
    figure_file = 'plots/' + "distance_5_graph_capacity" + '.png'
    abscissa_len = len(distance_5_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_5_graph_capacity, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_5_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_5_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_5_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_5_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_5_graph_scores
    figure_file = 'plots/' + "distance_5_graph_scores" + '.png'
    abscissa_len = len(distance_5_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_5_graph_scores, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_5_graph_scores_moving_average
    figure_file = 'plots/' + "distance_5_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_5_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_5_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Repeat for distance_6_graph_capacity
    figure_file = 'plots/' + "distance_6_graph_capacity" + '.png'
    abscissa_len = len(distance_6_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_6_graph_capacity, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_6_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_6_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_6_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_6_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_6_graph_scores
    figure_file = 'plots/' + "distance_6_graph_scores" + '.png'
    abscissa_len = len(distance_6_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_6_graph_scores, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_6_graph_scores_moving_average
    figure_file = 'plots/' + "distance_6_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_6_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_6_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Repeat for distance_7_graph_capacity
    figure_file = 'plots/' + "distance_7_graph_capacity" + '.png'
    abscissa_len = len(distance_7_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_7_graph_capacity, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_7_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_7_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_7_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_7_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_7_graph_scores
    figure_file = 'plots/' + "distance_7_graph_scores" + '.png'
    abscissa_len = len(distance_7_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_7_graph_scores, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_7_graph_scores_moving_average
    figure_file = 'plots/' + "distance_7_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_7_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_7_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Repeat for distance_8_graph_capacity
    figure_file = 'plots/' + "distance_8_graph_capacity" + '.png'
    abscissa_len = len(distance_8_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_8_graph_capacity, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_8_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_8_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_8_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_8_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_8_graph_scores
    figure_file = 'plots/' + "distance_8_graph_scores" + '.png'
    abscissa_len = len(distance_8_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_8_graph_scores, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_8_graph_scores_moving_average
    figure_file = 'plots/' + "distance_8_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_8_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_8_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Repeat for distance_9_graph_capacity
    figure_file = 'plots/' + "distance_9_graph_capacity" + '.png'
    abscissa_len = len(distance_9_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_9_graph_capacity, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_9_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_9_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_9_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_9_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_9_graph_scores
    figure_file = 'plots/' + "distance_9_graph_scores" + '.png'
    abscissa_len = len(distance_9_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_9_graph_scores, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_9_graph_scores_moving_average
    figure_file = 'plots/' + "distance_9_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_9_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_9_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Repeat for distance_10_graph_capacity
    figure_file = 'plots/' + "distance_10_graph_capacity" + '.png'
    abscissa_len = len(distance_10_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_10_graph_capacity, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_10_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_10_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_10_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_10_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_10_graph_scores
    figure_file = 'plots/' + "distance_10_graph_scores" + '.png'
    abscissa_len = len(distance_10_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_10_graph_scores, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_10_graph_scores_moving_average
    figure_file = 'plots/' + "distance_10_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_10_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_10_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Repeat for distance_11_graph_capacity
    figure_file = 'plots/' + "distance_11_graph_capacity" + '.png'
    abscissa_len = len(distance_11_graph_capacity)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_11_graph_capacity, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_11_graph_capacity_moving_average
    figure_file = 'plots/' + "distance_11_graph_capacity_moving_average" + '.png'
    abscissa_len = len(distance_11_graph_capacity)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_11_graph_capacity[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_11_graph_scores
    figure_file = 'plots/' + "distance_11_graph_scores" + '.png'
    abscissa_len = len(distance_11_graph_scores)
    abscissa = np.arange(abscissa_len)
    plt.scatter(abscissa, distance_11_graph_scores, color="C1")
    plt.savefig(figure_file)


    # Plot for distance_11_graph_scores_moving_average
    figure_file = 'plots/' + "distance_11_graph_scores_moving_average" + '.png'
    abscissa_len = len(distance_11_graph_scores)
    abscissa = np.arange(abscissa_len)
    running_avg = np.empty(abscissa_len)
    for t in range(abscissa_len):
        running_avg[t] = np.mean(distance_11_graph_scores[max(0, t - 20):(t + 1)])
    plt.scatter(abscissa, running_avg, color="C1")
    plt.savefig(figure_file)

    """

    for i in range(1, 12):
        # Plot for capacity
        figure_file = f'plots/distance_{i}_graph_capacity.png'
        fig_capacity, ax_capacity = plt.subplots()
        abscissa_len_capacity = len(globals()[f'distance_{i}_graph_capacity'])
        abscissa_capacity = np.arange(abscissa_len_capacity)
        ax_capacity.scatter(abscissa_capacity, globals()[f'distance_{i}_graph_capacity'], color="C1")
        ax_capacity.axes.get_xaxis().set_visible(False)
        ax_capacity.yaxis.tick_right()
        ax_capacity.set_xlabel("Occurence")
        ax_capacity.set_ylabel("Capacity", color="C1")
        ax_capacity.yaxis.set_label_position('left')
        ax_capacity.tick_params(axis='y', colors="C1")
        fig_capacity.savefig(figure_file)
        plt.close()

        # Plot for capacity running average
        figure_file = f'plots/distance_{i}_graph_capacity_running_average.png'
        fig_capacity_running_avg, ax_capacity_running_avg = plt.subplots()
        abscissa_len_capacity_running_avg = len(globals()[f'distance_{i}_graph_capacity'])
        abscissa_capacity_running_avg = np.arange(abscissa_len_capacity_running_avg)
        running_avg_capacity = np.empty(abscissa_len_capacity_running_avg)
        for t in range(abscissa_len_capacity_running_avg):
            running_avg_capacity[t] = np.mean(globals()[f'distance_{i}_graph_capacity'][max(0, t - 20):(t + 1)])
        ax_capacity_running_avg.scatter(abscissa_capacity_running_avg, running_avg_capacity, color="C1")
        ax_capacity_running_avg.axes.get_xaxis().set_visible(False)
        ax_capacity_running_avg.yaxis.tick_right()
        ax_capacity_running_avg.set_xlabel("Occurence")
        ax_capacity_running_avg.set_ylabel("Capacity", color="C1")
        ax_capacity_running_avg.yaxis.set_label_position('left')
        ax_capacity_running_avg.tick_params(axis='y', colors="C1")
        fig_capacity_running_avg.savefig(figure_file)
        plt.close()

        # Plot for scores
        figure_file = f'plots/distance_{i}_graph_scores.png'
        fig_scores, ax_scores = plt.subplots()
        abscissa_len_scores = len(globals()[f'distance_{i}_graph_scores'])
        abscissa_scores = np.arange(abscissa_len_scores)
        ax_scores.scatter(abscissa_scores, globals()[f'distance_{i}_graph_scores'], color="C1")
        ax_scores.axes.get_xaxis().set_visible(False)
        ax_scores.yaxis.tick_right()
        ax_scores.set_xlabel("Occurence")
        ax_scores.set_ylabel("Score", color="C1")
        ax_scores.yaxis.set_label_position('left')
        ax_scores.tick_params(axis='y', colors="C1")
        fig_scores.savefig(figure_file)
        plt.close()

        # Plot for scores running average
        figure_file = f'plots/distance_{i}_graph_scores_running_average.png'
        fig_scores_running_avg, ax_scores_running_avg = plt.subplots()
        abscissa_len_scores_running_avg = len(globals()[f'distance_{i}_graph_scores'])
        abscissa_scores_running_avg = np.arange(abscissa_len_scores_running_avg)
        running_avg_scores = np.empty(abscissa_len_scores_running_avg)
        for t in range(abscissa_len_scores_running_avg):
            running_avg_scores[t] = np.mean(globals()[f'distance_{i}_graph_scores'][max(0, t - 20):(t + 1)])
        ax_scores_running_avg.scatter(abscissa_scores_running_avg, running_avg_scores, color="C1")
        ax_scores_running_avg.axes.get_xaxis().set_visible(False)
        ax_scores_running_avg.yaxis.tick_right()
        ax_scores_running_avg.set_xlabel("Occurence")
        ax_scores_running_avg.set_ylabel("Scores", color="C1")
        ax_scores_running_avg.yaxis.set_label_position('left')
        ax_scores_running_avg.tick_params(axis='y', colors="C1")
        fig_scores_running_avg.savefig(figure_file)
        plt.close()

    """
    no of channels graph
    action index graph
    snr threshold graph
    """

    for i in range(1, 12):
        # Plot for no_of_channels
        figure_file = f'plots/distance_{i}_graph_no_of_channels.png'
        fig, ax = plt.subplots()
        abscissa_len = len(globals()[f'distance_{i}_graph_no_of_channels'])
        abscissa = np.arange(abscissa_len)
        ax.scatter(abscissa, globals()[f'distance_{i}_graph_no_of_channels'], color="C1")
        ax.axes.get_xaxis().set_visible(False)
        ax.yaxis.tick_right()
        ax.set_xlabel("Occurence")
        ax.set_ylabel("No of Channels", color="C1")
        ax.yaxis.set_label_position('left')
        ax.tick_params(axis='y', colors="C1")
        fig.savefig(figure_file)
        plt.close()

        # Plot for action_index
        figure_file = f'plots/distance_{i}_graph_action_index.png'
        fig, ax = plt.subplots()
        abscissa_len = len(globals()[f'distance_{i}_graph_action_index'])
        abscissa = np.arange(abscissa_len)
        ax.scatter(abscissa, globals()[f'distance_{i}_graph_action_index'], color="C1")
        ax.axes.get_xaxis().set_visible(False)
        ax.yaxis.tick_right()
        ax.set_xlabel("Occurence")
        ax.set_ylabel("Action Index", color="C1")
        ax.yaxis.set_label_position('left')
        ax.tick_params(axis='y', colors="C1")
        fig.savefig(figure_file)
        plt.close()

        # Plot for snr_threshold
        figure_file = f'plots/distance_{i}_graph_snr_threshold.png'
        fig, ax = plt.subplots()
        abscissa_len = len(globals()[f'distance_{i}_graph_snr_threshold'])
        abscissa = np.arange(abscissa_len)
        ax.scatter(abscissa, globals()[f'distance_{i}_graph_snr_threshold'], color="C1")
        ax.axes.get_xaxis().set_visible(False)
        ax.yaxis.tick_right()
        ax.set_xlabel("Occurence")
        ax.set_ylabel("SNR Threshold", color="C1")
        ax.yaxis.set_label_position('left')
        ax.tick_params(axis='y', colors="C1")
        fig.savefig(figure_file)
        plt.close()


    #CALCULATE MAX VALUE
    max_distance_1_score = np.max(distance_1_graph_scores)
    max_distance_1_capacity = np.max(distance_1_graph_capacity)

    max_distance_2_score = np.max(distance_2_graph_scores)
    max_distance_2_capacity = np.max(distance_2_graph_capacity)

    max_distance_3_score = np.max(distance_3_graph_scores)
    max_distance_3_capacity = np.max(distance_3_graph_capacity)

    max_distance_4_score = np.max(distance_4_graph_scores)
    max_distance_4_capacity = np.max(distance_4_graph_capacity)

    max_distance_5_score = np.max(distance_5_graph_scores)
    max_distance_5_capacity = np.max(distance_5_graph_capacity)

    max_distance_6_score = np.max(distance_6_graph_scores)
    max_distance_6_capacity = np.max(distance_6_graph_capacity)

    max_distance_7_score = np.max(distance_7_graph_scores)
    max_distance_7_capacity = np.max(distance_7_graph_capacity)

    max_distance_8_score = np.max(distance_8_graph_scores)
    max_distance_8_capacity = np.max(distance_8_graph_capacity)

    max_distance_9_score = np.max(distance_9_graph_scores)
    max_distance_9_capacity = np.max(distance_9_graph_capacity)

    max_distance_10_score = np.max(distance_10_graph_scores)
    max_distance_10_capacity = np.max(distance_10_graph_capacity)

    max_distance_11_score = np.max(distance_11_graph_scores)
    max_distance_11_capacity = np.max(distance_11_graph_capacity)

    distance_to_score_graph=[max_distance_1_score, max_distance_2_score, max_distance_3_score, max_distance_4_score, max_distance_5_score, max_distance_6_score, max_distance_7_score, max_distance_8_score, max_distance_9_score, max_distance_10_score, max_distance_11_score]

    distance_to_capacity_graph = [max_distance_1_capacity, max_distance_2_capacity, max_distance_3_capacity, max_distance_4_capacity,
                               max_distance_5_capacity, max_distance_6_capacity, max_distance_7_capacity, max_distance_8_capacity,
                               max_distance_9_capacity, max_distance_10_capacity, max_distance_11_capacity]

    print(max_distance_1_capacity)
    print(max_distance_2_capacity)
    print(max_distance_3_capacity)
    print(distance_to_capacity_graph)
    print(distance_to_score_graph)



    figure_file = 'plots/' + "distance_to_score_graph" + '.png'
    fig, ax = plt.subplots()
    abscissa_len_distance_score = len(distance_to_score_graph)
    abscissa = np.arange(abscissa_len_distance_score)
    print(abscissa_len_distance_score)
    print(abscissa)
    ax.scatter(abscissa, distance_to_score_graph, color="C1")
    ax.axes.get_xaxis().set_visible(False)
    ax.yaxis.tick_right()
    ax.set_xlabel("Distance")
    ax.set_ylabel("Score", color="C1")
    ax.yaxis.set_label_position('left')
    ax.tick_params(axis='y', colors="C1")
    fig.savefig(figure_file)
    plt.close()






    figure_file = 'plots/' + "distance_to_capacity_graph" + '.png'
    fig, ax = plt.subplots()
    abscissa_len_distance_capacity = len(distance_to_capacity_graph)
    abscissa = np.arange(abscissa_len_distance_capacity)
    print(abscissa_len_distance_capacity)
    print(abscissa)
    ax.scatter(abscissa, abscissa, distance_to_capacity_graph, color="C1")
    ax.axes.get_xaxis().set_visible(False)
    ax.yaxis.tick_right()
    ax.set_xlabel("Distance")
    ax.set_ylabel("Capacity", color="C1")
    ax.yaxis.set_label_position('left')
    ax.tick_params(axis='y', colors="C1")
    fig.savefig(figure_file)
    plt.close()







    x = [i + 1 for i in range(len(scores))]
    figure_file = 'plots/' + "Scores_over_time_moving_average" + '.png'
    plot_learning_curve(steps_array, scores, eps_history, figure_file)

    figure_file = 'plots/' + "Capacity_over_time_moving_average" + '.png'
    plot_learning_curve(steps_array, Capacity_graph, eps_history, figure_file)



