import numpy as np
import pandas as pd
from typing import List
from config import NUM_STATES, PHONEME_LIST
from prepare_data import batch_matrix_train
from get_prob import main
import matplotlib.pyplot as plt


class HMM():
    '''
    Use as :
        myHMM = HMM(init_a, init_b, init_pi) 
        path, highest_prob = myHMM.viterbi() 
    '''
    def __init__(self, init_probs: np.ndarray, trans: np.ndarray, num_states: int=NUM_STATES):
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
        
        self.states = np.array(range(num_states))
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
        path: np.ndarray (num_obs,1)
            Path of states that has the highest probability
        highest_prob: float
            Probability of the path that has the highest probability.
        prob: np.ndarray (num_obs, num_states)
            Max probability of arriving at state s in frame t.

        '''
        num_states = len(self.states) 
        num_obs = emit.shape[1]
        
        # Initialize probability and previous state matrices
        prob = np.zeros((num_obs, num_states))
        prev = np.zeros((num_obs, num_states), dtype=int)
        highest_prob = -1
        
        # Initialize the first observation
        for s in range(num_states):
            prob[0][s] = self.init_probs[s] + emit[s][0]

        # Fill the matrices
        for t in range(1, num_obs):
            for s in range(num_states):
                max_prob = -np.inf
                max_state = -1
                for r in range(num_states):
                    current_prob = prob[t-1][r] + self.trans[r][s] + emit[s][t]
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


def get_highest_prob_df(keyword: str = "never",
                        batch_per_file: int = 20,
                        batch_size: int = 60,
                        log_space: bool = True,
                        phoneme_list: List[str] = PHONEME_LIST,):
    '''
    Set the threshold for the HMM model for the window
    to be classified as containing the keyword.
    '''
    
    # get the emission matrix and true labels for the training data
    df_train_batch = batch_matrix_train(keyword, batch_per_file, batch_size, log_space)

    df_path = df_train_batch.copy()
    df_path['path'] = None
    df_path['highest_prob'] = None
    df_path['prob_matrix'] = None

    # get prior and transition probabilities (log)
    prior_vector, transition_matrix = main(keyword=keyword, phoneme_list=phoneme_list, log_space=log_space)

    # initialize the HMM model
    hmm_model = HMM(prior_vector, transition_matrix)

    # iterate over the training data
    for i, row in df_train_batch.iterrows():
        emission_matrix = row['emission_matrix']
        hmm_model.emit = emission_matrix
        path, highest_prob, prob_matrix = hmm_model.viterbi(emission_matrix)
        # store in the dataframe
        df_path.at[i, 'path'] = path
        df_path.at[i, 'highest_prob'] = highest_prob
        df_path.at[i, 'prob_matrix'] = prob_matrix

    return df_path


def _plot_highest_prob(df_path: pd.DataFrame):
    '''
    Plot the highest probability of the HMM model for each window.
    '''

    true_highest_prob = df_path[df_path['label'] == 1]['highest_prob']
    false_highest_prob = df_path[df_path['label'] == 0]['highest_prob']

    # Create a boxplot with both datasets
    plt.boxplot([true_highest_prob, false_highest_prob], labels=['True Highest Prob', 'False Highest Prob'])

    plt.title('Boxplot of True and False Highest Probabilities')
    plt.xlabel('Category')
    plt.ylabel('Values')
    plt.show()


def calculate_metrics(threshold: float,
                       df_path: pd.DataFrame
                       ) -> dict:
    """
    Calculate the training metrics for the HMM model, given a threshold.s
    """

    # label the samples based on the threshold
    df_path['pred_label'] = df_path['highest_prob'] > threshold

    # calculate the confusion matrix
    true_positive = df_path[(df_path['label'] == 1) & (df_path['pred_label'] == 1)].shape[0]
    false_positive = df_path[(df_path['label'] == 0) & (df_path['pred_label'] == 1)].shape[0]
    true_negative = df_path[(df_path['label'] == 0) & (df_path['pred_label'] == 0)].shape[0]
    false_negative = df_path[(df_path['label'] == 1) & (df_path['pred_label'] == 0)].shape[0]

    # calculate the metrics
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return {'precision': precision, 'recall': recall, 'f1_score': f1_score}


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