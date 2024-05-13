#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 20:52:51 2024

@author: eminozyoruk
"""

import numpy as np
import pandas as pd



class HMM():
    '''
    Use as :
        myHMM = HMM(init_a, init_b, init_pi) 
        path, highest_prob = myHMM.viterbi() 
    '''
    def __init__(self, states: np.ndarray, init_probs: np.ndarray, trans: np.ndarray):
        '''
        Initialize an HMM.
        Parameters
        ----------
        states: S hidden states
        init_probs : initial probabilities of each state
        trans : S × S transition matrix
        obs: sequence of T observations

        Returns
        -------
        None.
        '''
        
        self.states = states
        self.init_probs = init_probs
        self.trans = trans
        self.emit = None
    
    def find_emission(self, class_prob: np.ndarray):
        '''
        Derive emission probabilities by using the output of DNN.

        Parameters
        ----------
        class_prob : S x T matrix
                    Output of DNN. P(state[t] = s | observation[t]) 
                    Class probabilities for each class s and time t.

        Returns
        -------
        emit: S x T matrix
             Input to HMM. P(observation[t] | state[t]=s)
             Emission probabilities.
        '''
        pass 
    
    def viterbi(self, emit: np.ndarray):
        '''
        Given an observation vector obs, run the Viterbi algorithm to find the highest probability path. 

        Parameters
        ----------
        emit: S × T emission matrix. 
              emit[s][t] is the emission probability at time t for state s: P(o[t] | state=s)

        Returns
        -------
        path: np.ndarray
            Path of states that has the highest probability
        prob: np.ndarray
            Probability of the path.
        '''
        num_states = len(self.states) 
        num_obs = len(observations) 
        
        # Initialize probability and previous state matrices
        prob = np.zeros((num_obs, num_states))
        prev = np.zeros((num_obs, num_states), dtype=int)
        highest_prob = -1
        
        # Initialize the first observation
        for s in range(num_states):
            prob[0][s] = self.init_probs[s] * emit[s][0]
                
        # Fill the matrices
        for t in range(1, num_obs):
            for s in range(num_states):
                max_prob = -1
                max_state = -1
                for r in range(num_states):
                    current_prob = prob[t-1][r] * self.trans[r][s] * emit[s][t]
                    if current_prob > max_prob:
                        max_prob = current_prob
                        max_state = r
                prob[t][s] = max_prob
                prev[t][s] = max_state

        # Backtrack to find the optimal path
        path = np.zeros(num_obs, dtype=int)
        path[num_obs-1] = np.argmax(prob[num_obs-1])
        highest_prob = np.max(prob[num_obs-1])

        for t in range(num_obs-2, -1, -1):
            path[t] = prev[t+1][path[t+1]]

        return path, highest_prob, prob 



# Example usage
'''
states = np.array([0, 1])  # Example state space
init_probs = np.array([0.6, 0.4])  # Example initial probabilities
trans_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])  # Example transition matrix
original_emit_matrix = np.array([[0.9, 0.1], 
                                 [0.2, 0.8]])  # Example emission matrix
observations = np.array([0, 1, 0, 1])  # Example observations

emit_matrix = np.column_stack(([0.9, 0.2],  # o=0 is observed. P(o=0|s=0)=0.9 and P(o=0|s=1)=0.2
                               [0.1, 0.8],  # P(o=1|s=0)=0.1 and P(o=1|s=1)=0.8
                               [0.9, 0.2],
                               [0.1, 0.8]
                               ))  # Example emission matrix

myHMM = HMM(states, init_probs, trans_matrix)
path, highest_prob, prob_matrix = myHMM.viterbi(emit_matrix)
print("Decoded Path:", path, "Highest probability: ", highest_prob, "Prob matrix:", prob_matrix)
'''