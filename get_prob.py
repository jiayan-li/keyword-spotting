from config import PHONEME_LIST

from data_utils import (load_data, 
                        process_audio_file, 
                        label_df_mfcc)
from find_files import get_train_test_paths
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from joblib import load 


def get_label_df(phn_path: str, 
                 wav_path: str,
                 win_length: int = 400,
                 hop_length: int = 80,) -> pd.DataFrame:
    """
    get the labeled mfcc dataframe from one audio file
    """
    # two paths should be the same except for the file extension
    # use regular expression 
    if phn_path[:-3] != wav_path[:-3]:
        raise ValueError('The two paths should be the same except for the file extension.')

    df_phn = load_data(phn_path, 'phoneme')
    df_mfcc = process_audio_file(wav_path, 
                                 win_length=win_length, 
                                 hop_length=hop_length)
    df_label = label_df_mfcc(df_mfcc, df_phn)

    # convert the phoneme to tuple
    df_label['phoneme'] = df_label['phoneme'].apply(
                            lambda x: tuple([i.strip() for i in x.split(',')]))
    
    return df_label

# we should only count prior probability of the input of the NN
def get_prior(df_label: pd.DataFrame,
              log_space: bool = True) -> pd.DataFrame:
    """
    Get the prior probability of the phonemes.
    The dataframe should have a column 'phoneme'.
    """
    
    prior = df_label['phoneme'].value_counts()
    prior = prior.reset_index()
    prior.columns = ['phoneme', 'count']
    # if log_space:
    #     prior['prior'] = prior['prior'].apply(lambda x: np.log(x))

    return prior

def transition_prob(phoneme: Tuple[str], 
                    next_phoneme: Tuple[str], 
                    df_label: pd.DataFrame,
                    log_space: bool = True
                    ) -> Tuple[int, int]:
    """
    Get the transition probability of the specified phonemes.
    """

    current_phoneme_count = sum(df_label['phoneme'] == phoneme)
    transition_count = sum((df_label['phoneme'] == phoneme) & 
                           (df_label['phoneme'].shift(-1) == next_phoneme))

    return current_phoneme_count, transition_count

    # if log_space:
    #     return np.log(transition_count) - np.log(current_phoneme_count)
    # else:
    #     return transition_count/current_phoneme_count

def get_transition_list(phoneme_list: List[str] = PHONEME_LIST
                        ) -> List[Tuple[Tuple[str], Tuple[str]]]:
    """
    get the list of transition tuples
    """

    transition_list = [[(phoneme_list[i], phoneme_list[i]), (phoneme_list[i], phoneme_list[i+1])] for i in range(len(phoneme_list) - 1)]
    
    # Flatten the list of lists
    transition_list = [item for sublist in transition_list for item in sublist]
    return transition_list

def get_transition_df(df_label: pd.DataFrame,
                      log_space: bool = True) -> pd.DataFrame:
    """
    Get the transition probability of the phonemes.
    The dataframe should have a column 'phoneme'.
    """

    # Get list of all possible transitions between phonemes
    transition_list = get_transition_list(PHONEME_LIST)
    
    # Create DataFrame to store transition data
    transition_df = pd.DataFrame(transition_list, columns=['source_phoneme', 'next_phoneme'])
    
    # Compute transition and source counts for each transition
    transition_df['source_count'], transition_df['transition_count'] = zip(*transition_df.apply(
        lambda x: transition_prob(x['source_phoneme'], x['next_phoneme'], df_label), axis=1))
    
    return transition_df


def get_prior_transition(keyword: str = 'never',
                         dataset_type: str = 'train',
                         log_space: bool = True,):

    if dataset_type:
        # list of tuples
        dataset = get_train_test_paths(keyword)[dataset_type]

    # store the prior and transition probability
    prior_combined = None
    transition_combined = None

    for path_tup in dataset[:5]:
        wav_path = path_tup[0]
        phn_path = path_tup[1]
        df_label = get_label_df(phn_path, wav_path)
        prior = get_prior(df_label)
        transition = get_transition_df(df_label)

        if prior_combined is None:
            prior_combined = prior.copy()
        else:
            prior_combined = pd.concat([prior_combined, prior])
            prior_combined = prior_combined.groupby('phoneme').sum().reset_index()
        
        if transition_combined is None:
            transition_combined = transition.copy()
        else:
            transition_combined = pd.concat([transition_combined, transition])
            transition_combined = transition_combined.groupby(['source_phoneme', 'next_phoneme']).sum().reset_index()

    return prior_combined, transition_combined