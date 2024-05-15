"""
functions to prepare data for system input and label
"""

import numpy as np
import pandas as pd
from get_prob import get_label_df
from mfcc_label import prepare_data
from utils import load_data, get_train_test_paths
from typing import Dict

def _identify_key_frames(
        word_path: str, 
        phoneme_path: str, 
        wav_path: str,
        keyword: str = 'never') -> Dict[str, int]:
    """
    identify the amount of frames for the keyword
    Args:
        all paths should be of the same speaker and same prompt
    """

    df_label = get_label_df(phoneme_path, wav_path)
    df_word = load_data(word_path, "word")

    start_sample = df_word[df_word['word'] == keyword]['start_sample'].values[0]
    end_sample = df_word[df_word['word'] == keyword]['end_sample'].values[0]

    # range of frames that covers the first phoneme of the keyword
    first_phn_range = df_label[(df_label['start_sample'] <= start_sample)
                                & (df_label['end_sample'] >= start_sample)]
    # range of frames that covers the last phoneme of the keyword
    last_phn_range = df_label[(df_label['start_sample'] <= end_sample) 
                              & (df_label['end_sample'] >= end_sample)]
                        
    first_phn_start_idx = first_phn_range.index[0]
    first_phn_end_idx = first_phn_range.index[-1]
    last_phn_start_idx = last_phn_range.index[0]
    last_phn_end_idx = last_phn_range.index[-1]
    
    return {'first_phn_start_idx': first_phn_start_idx,
            'first_phn_end_idx': first_phn_end_idx,
            'last_phn_start_idx': last_phn_start_idx,
            'last_phn_end_idx': last_phn_end_idx}


def _get_all_frame_amount(
        keyword: str = "never",
        dataset_type: str = "train"):
    """
    for choosing the optimal frame amounts in the HMM model
    """

    if dataset_type:
        # list of tuples
        dataset = get_train_test_paths(keyword, rerun=True)[dataset_type]

    frame_amounts = []

    for path_tup in dataset:
        wav_path = path_tup[0]
        phn_path = path_tup[1]
        wrd_path = path_tup[2]

        key_idx = _identify_key_frames(wrd_path, phn_path, wav_path, keyword)
        frame_num = key_idx['last_phn_end_idx'] - key_idx['first_phn_start_idx'] + 1
        frame_amounts.append(frame_num)

    return frame_amounts


def _assemble_frame_info(
        wav_path: str,
        df_label: pd.DataFrame,
        last_phn_start_idx: int,
        last_phn_end_idx: int,
        batch_end_idx: int,
        batch_size: int,
        log_space: bool = True):

    batch_info = {}
    batch_info['input'] = wav_path

    emission_matrix = df_label.loc[batch_end_idx-batch_size+1: batch_end_idx, 'label']
    # (14, batch_size) matrix
    emission_matrix = np.vstack(emission_matrix.to_numpy()).T
    # to (batch_size, 12)
    emission_matrix = emission_matrix[:12, :]
    if log_space:
        # add a small value to avoid log(0)
        emission_matrix = np.log(emission_matrix + 1e-10)
    else:
        pass

    batch_info['emission_matrix'] = emission_matrix

    # batch_end_idx = last_phn_end_idx+i
    if last_phn_start_idx <= batch_end_idx <= last_phn_end_idx:
        batch_info['label'] = 1
    else:
        batch_info['label'] = 0

    return batch_info

def prepare_batch_matrix(word_path: str, 
                         phoneme_path: str, 
                         wav_path: str,
                         num_random: int,
                         batch_size: int = 60,
                         log_space: bool = True,
                         random_seed: int = 42
                         ) -> np.ndarray:
    """
    Prepare a batch matrix for the HMM model of size (batch_size, 14)
    Args:
        word_path: path to the word file
        phoneme_path: path to the phoneme file
        wav_path: path to the wav file
        num_within_key_frames: number of frames within the keyword to be included
            for the batch matrix
        num_random: number of random frames to be included
        batch_size: number of frames to be included in the batch matrix
        log_space: whether to convert the emission matrix to log space
    """

    df_label = prepare_data(phoneme_path, wav_path)
    key_frame_idx = _identify_key_frames(word_path, phoneme_path, wav_path)
    last_phn_start_idx = key_frame_idx['last_phn_start_idx']
    last_phn_end_idx = key_frame_idx['last_phn_end_idx']
    
    # sample num_random idx from the range (batch_size, len(df_label))
    random_idx = np.random.randint(batch_size, len(df_label), 
                                   num_random, random_seed)

    file_info = []

    # sample random frames
    for end_id in random_idx:
        batch_info = _assemble_frame_info(wav_path, df_label, last_phn_start_idx, last_phn_end_idx, end_id, batch_size, log_space)
        file_info.append(batch_info)

    # sample frames within the keyword
    for end_id in range(last_phn_start_idx, last_phn_end_idx+1):
        batch_info = _assemble_frame_info(wav_path, df_label, last_phn_start_idx, last_phn_end_idx, end_id, batch_size, log_space)

        file_info.append(batch_info)

    return file_info


def batch_matrix_train(
        keyword: str = "never",
        num_random: int = 20,
        batch_size: int = 60,
        log_space: bool = True):
    """
    Main function to prepare batch matrix for all files in the dataset
    """

    dataset = get_train_test_paths(keyword, rerun=True)['train']

    all_batch_matrix = []

    for path_tup in dataset:
        wav_path = path_tup[0]
        phn_path = path_tup[1]
        wrd_path = path_tup[2]

        batch_matrix = prepare_batch_matrix(wrd_path, phn_path, wav_path, num_random, 
                                            batch_size, log_space)
        all_batch_matrix.extend(batch_matrix)

    df_train_batch = pd.DataFrame(all_batch_matrix)

    # save the dataframe
    # df_train_batch.to_csv(f"timit/data/{keyword}_train_batch.csv", index=False)

    return df_train_batch