from data_utils import (load_data, 
                        process_audio_file, 
                        label_df_mfcc)
from find_files import _get_path_list
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

def get_label_df(phn_path: str, 
                 wav_path: str,
                 win_length: int = 400,
                 hop_length: int = 80,) -> pd.DataFrame:

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

# we shoul only count prior probability of the input of the NN
def get_prior(df_label: pd.DataFrame,
              log_space: bool = True) -> pd.DataFrame:
    """
    Get the prior probability of the phonemes.
    The dataframe should have a column 'phoneme'.
    """
    
    prior = df_label['phoneme'].value_counts(normalize=True)
    prior = prior.reset_index()
    prior.columns = ['phoneme', 'prior']
    if log_space:
        prior['prior'] = prior['prior'].apply(lambda x: np.log(x))

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
    if log_space:
        return np.log(transition_count) - np.log(current_phoneme_count)
    else:
        return transition_count/current_phoneme_count

def get_transition_list(phoneme_list: List[str]
                                  = PHONEME_LIST) -> List[Tuple[str]]:
    
    return [tuple(phoneme_list[i:i+2]) for i in range(len(phoneme_list)-1)]

def get_transition_df(df_label: pd.DataFrame,
                      log_space: bool = True) -> pd.DataFrame:
    """
    Get the transition probability of the phonemes.
    The dataframe should have a column 'phoneme'.
    """

    transition_list = get_transition_list(PHONEME_LIST)
    transition_df = pd.DataFrame(transition_list, columns=['phoneme', 'next_phoneme'])
    transition_df['transition_prob'] = transition_df.apply(
        lambda x: transition_prob(x['phoneme'], x['next_phoneme'], df_label, log_space), axis=1)

    return transition_df