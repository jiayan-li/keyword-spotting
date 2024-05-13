from data_utils import (load_data, 
                        process_audio_file, 
                        label_df_mfcc)
from find_files import get_all_paths
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from config import PHONEME_LIST
import pickle

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
                    log_space: bool = True) -> float:
    """
    Get the transition probability of the specified phonemes.
    """

    current_phoneme_count = sum(df_label['phoneme'] == phoneme)
    # print(current_phoneme_count)
    if current_phoneme_count == 0:
        return 0.0
    
    transition_count = sum((df_label['phoneme'] == phoneme) & (df_label['phoneme'].shift(-1) == next_phoneme))
    # print(transition_count)

    return current_phoneme_count, transition_count

    # if log_space:
    #     return np.log(transition_count) - np.log(current_phoneme_count)
    # else:
    #     return transition_count/current_phoneme_count

def get_transition_list(phoneme_list: List[str] = PHONEME_LIST
                        ) -> List[Tuple[Tuple[str], Tuple[str]]]:
    
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

    transition_list = get_transition_list(PHONEME_LIST)
    transition_df = pd.DataFrame(transition_list, columns=['source_phoneme', 'next_phoneme'])
    transition_df['source_count'] = transition_df.apply(
        lambda x: transition_prob(x['source_phoneme'], x['next_phoneme'], df_label)[0], axis=1)
    transition_df['transition_count'] = transition_df.apply(
        lambda x: transition_prob(x['source_phoneme'], x['next_phoneme'], df_label)[1], axis=1)

    return transition_df

# split files into training and testing first
def get_prior_transition(keyword: str,
                         log_space: bool = True,):

    # check if saved locally
    try:
        with open(f'processed_data/wav_paths_{keyword}.pkl', 'rb') as f:
            wav_paths = pickle.load(f)
        with open(f'processed_data/phn_paths_{keyword}.pkl', 'rb') as f:
            phn_paths = pickle.load(f)

    except FileNotFoundError:
        # find the files with the keyword
        wav_paths = get_all_paths(keyword, 'wav')
        phn_paths = get_all_paths(keyword, 'phn')

        # save the two lists to pickle
        with open(f'processed_data/wav_paths_{keyword}.pkl', 'wb') as f:
            pickle.dump(wav_paths, f)
        with open(f'processed_data/phn_paths_{keyword}.pkl', 'wb') as f:
            pickle.dump(phn_paths, f)

    # store the prior and transition probability
    prior_combined = None
    transition_combined = None

    for i in range(len(wav_paths))[:5]:
        wav_path = wav_paths[i]
        phn_path = phn_paths[i]
        df_label = get_label_df(phn_path, wav_path)
        prior = get_prior(df_label, log_space)
        transition = get_transition_df(df_label, log_space)

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