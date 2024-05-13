"""
functions to get the prior and transition probability of the phonemes
"""

from config import PHONEME_LIST
from utils import load_data, split_phoneme
from mfcc_label import process_audio_file, label_df_mfcc
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
    df_phn = split_phoneme(df_phn)
    df_mfcc = process_audio_file(wav_path, 
                                 win_length=win_length, 
                                 hop_length=hop_length)
    df_label = label_df_mfcc(df_mfcc, df_phn)

    # convert the phoneme to tuple
    df_label['phoneme'] = df_label['phoneme'].apply(
                            lambda x: tuple([i.strip() for i in x.split(',')]))
    
    return df_label

def get_prior_count(df_label: pd.DataFrame,
                    phoneme_list: List[str] = PHONEME_LIST
                    ) -> pd.DataFrame:
    """
    Get the prior count of the phonemes.
    The dataframe should have a column 'phoneme'.
    """
    
    phonemes = [phoneme for tup in df_label['phoneme'].values for phoneme in tup]
    prior = pd.Series(phonemes).value_counts().reset_index()
    prior.columns = ['phoneme', 'count']

    return prior

def get_transition_count(phoneme: Tuple[str], 
                         next_phoneme: Tuple[str], 
                         df_label: pd.DataFrame,
                        ) -> Tuple[int, int]:
    """
    takes in two phonemes and return the count of the first phoneme and 
    the transition count (from the first to next phoneme).
    """

    current_phoneme_count = sum(df_label['first_phoneme'] == phoneme)
    transition_count = sum((df_label['first_phoneme'] == phoneme) & 
                           (df_label['first_phoneme'].shift(-1) == next_phoneme))

    return current_phoneme_count, transition_count

def _get_transition_list(phoneme_list: List[str] = PHONEME_LIST
                        ) -> List[Tuple[Tuple[str], Tuple[str]]]:
    """
    get the list of transition tuples
    """

    # drop 'h#', '#b' from the list first
    phoneme_list = [i for i in phoneme_list if i not in ['h#', '#b']]

    transition_list = [[(phoneme_list[i], phoneme_list[i]), 
                        (phoneme_list[i], phoneme_list[i+1])] 
                        for i in range(len(phoneme_list) - 1)]
    
    # Flatten the list of lists
    transition_list = [item for sublist in transition_list for item in sublist]
    
    return transition_list

def get_transition_df(df_label: pd.DataFrame
                      ) -> pd.DataFrame:
    """
    Get the transition probability of the phonemes.
    The dataframe should have a column 'phoneme'.
    """

    # Get list of all possible transitions between phonemes
    transition_list = _get_transition_list(PHONEME_LIST)
    
    # Create DataFrame to store transition data
    transition_df = pd.DataFrame(transition_list, columns=['source_phoneme', 'next_phoneme'])
    
    # Get the count of each phoneme and transition
    transition_df['source_count'], transition_df['transition_count'] = zip(*transition_df.apply(
        lambda x: get_transition_count(x['source_phoneme'], x['next_phoneme'], df_label), axis=1))
    
    return transition_df

def _add_counts(df1: pd.DataFrame, 
                df2: pd.DataFrame,
                prob_type: str,) -> pd.DataFrame:
    """
    Add counts together from two DataFrames based on matching 'phoneme' values.
    """

    if prob_type not in ['prior', 'transition']:
        raise ValueError('type should be either prior or transition.')

    if prob_type == 'prior':
        merge_on = 'phoneme'
        merged_df = pd.merge(df1, df2, on=merge_on, how='outer', suffixes=('_df1', '_df2'))
        merged_df['count'] = merged_df['count_df1'].fillna(0) + merged_df['count_df2'].fillna(0)

    else:
        merge_on = ['source_phoneme', 'next_phoneme']
        merged_df = pd.merge(df1, df2, on=merge_on, how='outer', suffixes=('_df1', '_df2'))
        merged_df['source_count'] = merged_df['source_count_df1'].fillna(0) + merged_df['source_count_df2'].fillna(0)
        merged_df['transition_count'] = merged_df['transition_count_df1'].fillna(0) + merged_df['transition_count_df2'].fillna(0)
    

    columns_to_drop = [col for col in merged_df.columns if 'df1' in col or 'df2' in col]
    # print(columns_to_drop)
    merged_df = merged_df.drop(columns_to_drop, axis=1)
    
    return merged_df

def _adjust_row_order(df_transition: pd.DataFrame,
                      transition_list: List[Tuple[str, str]]  
                      ) -> pd.DataFrame:
    
    new_df = pd.DataFrame(index=pd.MultiIndex.from_tuples(transition_list, names=['source_phoneme', 'next_phoneme']))
    df_transition.set_index(['source_phoneme', 'next_phoneme'], inplace=True)
    
    return df_transition.reindex(new_df.index).reset_index(drop=False)

def get_prior_transition(keyword: str = 'never',
                         dataset_type: str = 'train',
                         phoneme_list: List[str] = PHONEME_LIST,
                         log_space: bool = True,
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main function to get the prior and transition probability of the phonemes
    for training dataset.
    """

    if dataset_type:
        # list of tuples
        dataset = get_train_test_paths(keyword)[dataset_type]

    # store the prior and transition probability
    prior_concat = pd.DataFrame(columns=['phoneme', 'count'])
    prior_concat['phoneme'] = phoneme_list
    prior_concat['count'] = 0
    transition_concat = pd.DataFrame(columns=['source_phoneme', 
                                              'next_phoneme', 
                                              'source_count', 
                                              'transition_count'])
    # Get list of all possible transitions between phonemes
    transition_list = _get_transition_list(PHONEME_LIST)
    
    # Create DataFrame to store transition data
    transition_concat = pd.DataFrame(transition_list, 
                                     columns=['source_phoneme', 'next_phoneme'])
    transition_concat['source_count'] = 0
    transition_concat['transition_count'] = 0

    for path_tup in dataset:
        wav_path = path_tup[0]
        phn_path = path_tup[1]
        df_label = get_label_df(phn_path, wav_path)
        prior = get_prior_count(df_label)
        transition = get_transition_df(df_label)

        # merget the count
        prior_concat = _add_counts(prior_concat, 
                                   prior, 
                                   'prior')
        transition_concat = _add_counts(transition_concat, 
                                        transition, 
                                        'transition')

    # adjust the row order
    transition_concat = _adjust_row_order(transition_concat, 
                                          transition_list)

    # get log probability
    if log_space:
        prior_concat['log_prior'] = np.log(prior_concat['count']) - np.log(sum(prior_concat['count']))
        transition_concat['log_transition'] = np.log(transition_concat['transition_count']) - np.log(transition_concat['source_count'])
    
    # save to processed_data
    prior_concat.to_csv(f'processed_data/prior_{dataset_type}_{keyword}.csv', index=False)
    transition_concat.to_csv(f'processed_data/transition_{dataset_type}_{keyword}.csv', index=False)

    return prior_concat, transition_concat