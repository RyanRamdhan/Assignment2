# Write your experiments in here! You can use the plotting helper functions from the previous assignment if you want.
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from ShortCutAgents import QLearningAgent, SARSAAgent, ExpectedSARSAAgent
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment

class CreatePlot:
    def __init__(self,n_episodes, title=None):
        self.fig,self.ax = plt.subplots()
        self.ax.set_xlabel('repetitions')
        self.ax.set_ylabel('Average reward') 
        self.ax.set_yticks(np.arange(-5000, 1, 1000//10))
        self.ax.set_xlim(-50, n_episodes)
        self.ax.set_xticks(np.arange(0, n_episodes+1, n_episodes//5))
        if title is not None:
            self.ax.set_title(title)
        
        
    def add_curve(self,x,y,label=None):
        ''' x: vector of parameter values
        y: vector of associated mean reward for the parameter values in x 
        label: string to appear as label in plot legend '''
        if label is not None:
            self.ax.plot(x,y,label=label)
        else:
            self.ax.plot(x,y)
        
    def save(self,name='test.png'):
        ''' name: string for filename of saved figure '''
        self.ax.set_yticks(np.arange(0, 1.1, 0.1))
        self.ax.legend()
        self.fig.savefig(name,dpi=300)

def smooth(y, window, poly=1):
    '''
    y: vector to be smoothed 
    window: size of the smoothing window '''
    return savgol_filter(y,window,poly)

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
    
def run_repetitions(agent_type, alpha, n_episodes, n_repititions):
    reward_array = np.empty(n_episodes)
    epsilon = 0.1
    if agent_type == 'qlearning':
        policy = QLearningAgent(4, ShortcutEnvironment().state_size(), epsilon, alpha)
    elif agent_type == 'sarsa':
        policy = SARSAAgent(4, ShortcutEnvironment().state_size(), epsilon, alpha)
    elif agent_type == 'expected_sarsa':
        policy = ExpectedSARSAAgent(4, ShortcutEnvironment().state_size(), epsilon, alpha)

    for episode in range(n_episodes):
        cum_reward = 0
        for repitition in range(n_repititions):
            env = ShortcutEnvironment()
            state = env.state()
            finished = False
            while not finished:
                action = policy.select_action(state)
                reward = env.step(action)
                cum_reward += reward
                next_state = env.state()
                policy.update(state, action, reward, next_state, env.done())           
                finished = env.done()
                state = next_state
        cum_reward /= n_repititions
        reward_array[episode] = cum_reward
    print_greedy_actions(policy.Q_sa)
    return reward_array

def run_windy_repetitions(agent_type, n_episodes):
    print(agent_type)
    env = WindyShortcutEnvironment()
    epsilon = 0.1
    alpha = 0.1
    
    if agent_type == 'qlearning':
        policy = QLearningAgent(4, WindyShortcutEnvironment().state_size(), epsilon, alpha)
    elif agent_type == 'sarsa':
        policy = SARSAAgent(4, WindyShortcutEnvironment().state_size(), epsilon, alpha)

    for episode in range(n_episodes):
        state = env.state()
        finished = False
        
        while not finished:
            action = policy.select_action(state)
            
            reward = env.step(action)
            next_state = env.state()
            policy.update(state, action, reward, next_state, env.done())           
            finished = env.done()
            state = next_state
        env.reset()
              
    Q_sa = policy.Q_sa
    
    print_greedy_actions(Q_sa)
    return Q_sa


#...............................................................................................
def create_plots(n_episodes, n_repitions):
     agent_types = ['qlearning', 'sarsa', 'expected_sarsa']
     #agent_types = ['expected_sarsa']
     for agent_type in agent_types:
         print(agent_type)
         plot = CreatePlot(n_episodes, title='cumulative reward')
         alphas = [0.01, 0.1, 0.5, 0.9]
         for alpha in alphas:
             print('alpha: '+str(alpha))
             reward_array = run_repetitions(agent_type, alpha, n_episodes, n_repitions)
             smoothed = smooth(reward_array, 31)
             plot.add_curve(x=np.arange(n_episodes), y=smoothed, label="alpha: "+str(alpha))
         plot.save('smoothed_'+agent_type+'.png')

def create_comparison(n_episodes, n_repititions):
    agent_types = ['qlearning', 'sarsa', 'expected_sarsa']
    plot = CreatePlot(n_episodes, title='comparison plot')
    alphas = [0.9, 0.1, 0.1]
    i = 0
    for agent_type in agent_types:
        reward_array = run_repetitions(agent_type, alphas[i], n_episodes, n_repititions)
        smoothed = smooth(reward_array, 31)
        plot.add_curve(x=np.arange(n_episodes), y=smoothed, label=agent_type)
        
        i += 1
    plot.save('comparison.png')

def create_windy_plots(n_episodes):
    agent_types = ['qlearning', 'sarsa']
    state_size = WindyShortcutEnvironment().state_size()
    actions = [0,1,2,3]
    best_actions = np.empty(state_size)
    for agent_type in agent_types:
        Q_sa = run_windy_repetitions(agent_type, n_episodes)
        
        for state in np.arange(state_size):
            best_actions[state] = np.argmax(Q_sa[state])
                    
        plot = plt.plot(np.arange(state_size), best_actions, 'o',label=agent_type)
        plt.show()
         

create_plots(1000, 100)
#create_windy_plots(10000)
#run_repetitions('qlearning', 0.5, 1000, 100)
#create_comparison(1000, 100)

        
    
