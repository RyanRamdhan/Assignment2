import random
import numpy as np


class QLearningAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha=0.1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q_sa = np.zeros((n_states, n_actions))
        
        
        
    def select_action(self, state):
        # TO DO: Add own code
        #Select action bases on e-greedy strategy
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.argmax(self.Q_sa[state])
        
        return action
        
    def update(self, state, action, reward, next_state, done):
        # TO DO: Add own code
        max_next_Q = np.max(self.Q_sa[next_state])
        self.Q_sa[state, action] += self.alpha * (reward + max_next_Q - self.Q_sa[state, action])

class SARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha=0.1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q_sa = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # TO DO: Add own code
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.argmax(self.Q_sa[state])
        
        return action
    
    
    def update(self, state, action, reward, next_state, done):
        # TO DO: Add own code
        if not done:
            next_action = self.select_action(next_state)
            next_reward = reward + self.Q_sa[next_state, next_action]

        else:
            next_reward = reward

        self.Q_sa[state, action] += self.alpha * (next_reward - self.Q_sa[state, action])

class ExpectedSARSAAgent(object):

    def __init__(self, n_actions, n_states, epsilon, alpha=0.1):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        self.alpha = alpha
        self.Q_sa = np.zeros((n_states, n_actions))
        
    def select_action(self, state):
        # TO DO: Add own code
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else:
            action = np.argmax(self.Q_sa[state])
        
        return action
        
    def update(self, state, action, reward, next_state, done):
        # TO DO: Add own code
        probabilities = self.action_probabilities(state)
        if not done:
            next_action = self.select_action(next_state)
            next_reward = reward + self.Q_sa[next_state, next_action]

        else:
            next_reward = reward
        test = np.sum(probabilities * self.Q_sa[next_state, action])
        self.Q_sa[state, action] += self.alpha * (next_reward + test - self.Q_sa[state, action])
        
    
    def action_probabilities(self, state):
        probabilities = np.zeros(self.n_actions)
        [0,1,0,0]
        if np.random.rand() < self.epsilon:
            action_probabilities = np.ones(self.n_actions) / self.n_actions
            return action_probabilities
        else:
            for action in self.Q_sa[state]:
                probabilities[np.where(action == np.argmax(self.Q_sa[state]))] = 1
            return probabilities
    
    