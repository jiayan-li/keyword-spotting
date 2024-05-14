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


def prepare_batch_matrix(word_path: str, 
                         phoneme_path: str, 
                         wav_path: str,
                         batch_per_file: int = 20,
                         batch_size: int = 60,
                         ) -> np.ndarray:
    """
    Prepare a batch matrix for the HMM model of size (batch_size, 14)
    """

    df_label = prepare_data(phoneme_path, wav_path)
    key_frame_idx = _identify_key_frames(word_path, phoneme_path, wav_path)
    last_phn_start_idx = key_frame_idx['last_phn_start_idx']
    last_phn_end_idx = key_frame_idx['last_phn_end_idx']

    file_info = []

    for i in range(0-int(batch_per_file/2), int(batch_per_file/2)):
        batch_info = {}
        batch_info['input'] = wav_path

        emission_matrix = df_label.loc[last_phn_end_idx-batch_size+1+i: last_phn_end_idx+i, 'label']
        # (batch_size, 14) matrix
        emission_matrix = np.vstack(emission_matrix.to_numpy())
        # to (batch_size, 12)
        emission_matrix = emission_matrix[:, :-2]
        batch_info['emission_matrix'] = emission_matrix

        if i<= 0 and last_phn_end_idx+i >= last_phn_start_idx:
            batch_info['label'] = 1
        else:
            batch_info['label'] = 0

        file_info.append(batch_info)

    return file_info


def batch_matrix_train(
        keyword: str = "never",
        batch_per_file: int = 20,
        batch_size: int = 60):
    """
    Main function to prepare batch matrix for all files in the dataset
    """

    dataset = get_train_test_paths(keyword, rerun=True)['train']

    all_batch_matrix = []

    for path_tup in dataset:
        wav_path = path_tup[0]
        phn_path = path_tup[1]
        wrd_path = path_tup[2]

        batch_matrix = prepare_batch_matrix(wrd_path, phn_path, wav_path, batch_per_file, batch_size)
        all_batch_matrix.extend(batch_matrix)

    df_train_batch = pd.DataFrame(all_batch_matrix)

    # save the dataframe
    # df_train_batch.to_csv(f"timit/data/{keyword}_train_batch.csv", index=False)

    return df_train_batch