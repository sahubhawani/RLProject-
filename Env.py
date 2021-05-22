# Import routines

import numpy as np
import math
import random
from sklearn import preprocessing

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = tuple([(pick_up,drop) for pick_up in (1,2,3,4,5) for drop in (1,2,3,4,5) if pick_up!=drop])
        self.state_space = [(loc, time, day) for loc in np.arange(1,m+1) for time in range(t) for day in range(d)]
        self.state_init = random.choice(self.state_space)
        self.state_input = (np.arange(1,m+1) , np.arange(0,t) , np.arange(0,d))
        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""

        ohe = preprocessing.OneHotEncoder()
        location_vector = ohe.fit_transform(self.state_input[0].reshape(-1,1)).todense()[:,state[0]-1]
        time_vector = ohe.fit_transform(self.state_input[1].reshape(-1,1)).todense()[:,state[1]]
        day_vector = ohe.fit_transform(self.state_input[2].reshape(-1,1)).todense()[:,state[2]]
        
        state_encod =  np.concatenate((location_vector,time_vector,day_vector),axis=0)
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

        
    #     return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 1:
            requests = np.random.poisson(2)
        elif location == 2:
            requests = np.random.poisson(12)
        elif location == 3:
            requests = np.random.poisson(4)
        elif location == 4:
            requests = np.random.poisson(7)
        else:
            requests = np.random.poisson(8)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i-1] for i in possible_actions_index]

        
        actions.append([0,0])

        return possible_actions_index,actions   
          



    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        if (action[0] == 0 and action[1] == 0):
            reward = -C
        else:
            reward = R * (Time_matrix[action[0]-1][action[1]-1][state[1]][state[2]]) - C * ((Time_matrix[action[0]-1][action[1]-1][state[1]][state[2]]) + (Time_matrix[state[0]-1][action[0]-1][state[1]][state[2]]))
        return reward




    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        if (action[0] == 0 and action[1] == 0):
            idle_next_time = state[1] + 1
            if not ((state[2]+(idle_next_time // 24)) > 6):
                if (idle_next_time // 24 >= 1):
                    next_state = (state[0], 0 , state[2] + 1)
                else:
                    next_state = (state[0],idle_next_time, state[2])
            else:
                if(idle_next_time // 24 >= 1):
                    next_state = (state[0], 0, 0) 
                else:
                    next_state = (state[0],idle_next_time, state[2])
                
            return next_state
        else:
            
            active_next_time = state[1] + Time_matrix[action[0]-1][action[1]-1][state[1]][state[2]]
            
            if not ((state[2]+(active_next_time // 24)) > 6):
                if (active_next_time // 24 >= 1):
                    next_state = (action[1],(active_next_time % 24), state[2]+(active_next_time // 24))
                else:
                    next_state = (action[1], active_next_time , state[2] )
            else:
                if(active_next_time // 24 >= 1):
                    next_state = (action[1],(active_next_time % 24) , (active_next_time // 24)-1)
                else:
                    next_state = (action[1], active_next_time , state[2] )
            return next_state




    def reset(self):
        return self.action_space, self.state_space, self.state_init
