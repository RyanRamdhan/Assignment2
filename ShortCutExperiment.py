# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from ShortCutEnvironment import ShortcutEnvironment

def print_greedy_actions(Q):
    greedy_actions = np.argmax(Q, 1).reshape((12,12))
    print_string = np.zeros((12, 12), dtype=str)
    print_string[greedy_actions==0] = '^'
    print_string[greedy_actions==1] = 'v'
    print_string[greedy_actions==2] = '<'
    print_string[greedy_actions==3] = '>'
    print_string[np.max(Q, 1).reshape((12, 12))==0] = '0'
    line_breaks = np.zeros((12,1), dtype=str)
    line_breaks[:] = '\n'
    print_string = np.hstack((print_string, line_breaks))
    print(print_string.tobytes().decode('utf-8')) 
    
def run_repetitions(agent_type, n_episodes = 1000, n_repititions=100):
    reward_array = np.empty(n_repititions)
    if agent_type == 'qlearning':
        policy = QLearningAgent(4, ShortcutEnvironment().state_size(), epsilon=0.1, alpha=0.1)
    elif agent_type == 'sarsa':
        policy = SARSAAgent(4, ShortcutEnvironment().state_size(), epsilon=0.1, alpha=0.1)
    elif agent_type == 'expected_sarsa':
        policy = ExpectedSARSAAgent(4, ShortcutEnvironment().state_size(), epsilon=0.1, alpha=0.1)
        
        
    for repitition in range(n_repititions):
        cum_reward = 0
        for episode in range(n_episodes):
            env = ShortcutEnvironment()
            state = env.state()
            finished = False
            while not finished:
                action = policy.select_action(state)
                print(action)
                reward = env.step(action)
                cum_reward += reward
                next_state = env.state()
                policy.update(state, action, reward, next_state, env.done())           
                finished = env.done()
                state = next_state
        cum_reward /= 1000
        reward_array[repitition] = cum_reward
        print(policy.Q_sa)
        print_greedy_actions(policy.Q_sa)
    
    create_plot(n_repititions, reward_array)
    
def run_repetitions_sarsa(n_episodes = 1000, n_repititions=100):
    reward_array = np.empty(n_repititions)
    sarsa = SARSAAgent(4, ShortcutEnvironment().state_size(), epsilon=0.1, alpha=0.1)
    for repitition in range(n_repititions):
        cum_reward = 0
        for episode in range(n_episodes):
            env = ShortcutEnvironment()
            state = env.state()
            finished = False
            while not finished:
                action = sarsa.select_action(state)
                print(action)
                reward = env.step(action)
                cum_reward += reward
                next_state = env.state()
                sarsa.update(state,reward, action, next_state, env.done())          
                finished = env.done()
                
                state = next_state
        cum_reward /= 1000
        reward_array[repitition] = cum_reward
        
        print_greedy_actions(sarsa.Q_sa)
    
    create_plot(n_repititions, reward_array)
    
    
def create_plot(n_repitions, reward_array):
    smoothed = smooth(reward_array, 11)
    plot = plt.plot(np.arange(n_repitions), smoothed)
    plt.xlabel('repititions')
    plt.ylabel('average reward')
    plt.title('cum plot')
    plt.show()
    
def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)       

     
run_repetitions('expected_sarsa')
        

        
    
