"""
functions to prepare data for system input and label
"""

from get_prob import get_label_df
from utils import load_data, get_train_test_paths

def _identify_frame_amount(
        word_path: str, 
        phoneme_path: str, 
        wav_path: str,
        keyword: str = 'never') -> int:
    """
    identify the amount of frames for the keyword
    Args:
        all paths should be of the same speaker and same prompt
    """

    df_label = get_label_df(phoneme_path, wav_path)
    df_word = load_data(word_path, "word")

    start_sample = df_word[df_word['word'] == keyword]['start_sample'].values[0]
    end_sample = df_word[df_word['word'] == keyword]['end_sample'].values[0]

    # identify the frames that have the word 'never'
    df_pos = df_label[df_label['start_sample'] >= start_sample][df_label['end_sample'] <= end_sample]
    
    return len(df_pos)


def _get_all_frame_amount(
        keyword: str = "never",
        dataset_type: str = "train"):
    """
    for choosing the optimal frame amount
    """

    if dataset_type:
        # list of tuples
        dataset = get_train_test_paths(keyword, rerun=True)[dataset_type]

    frame_amounts = []

    for path_tup in dataset:
        wav_path = path_tup[0]
        phn_path = path_tup[1]
        wrd_path = path_tup[2]

        frame_num = _identify_frame_amount(wrd_path, phn_path, wav_path, keyword)
        frame_amounts.append(frame_num)

    # optimal frame amount is the lowest frame amount
    return frame_amounts