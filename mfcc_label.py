"""
functions to extract MFCC features from audio files and label the MFCC features
"""

import numpy as np
import librosa
import config
from find_files import get_train_test_paths
from utils import load_data, split_phoneme
import pandas as pd

def _load_mfcc(
    file_path: str,
    desired_sr: int = 16000,  # 16 kHz, sampling rate of phonems/words/transcripts
    n_mfcc: int = 20,
    win_length: int = 1024,  # 64 ms
    hop_length: int = 512,  # 32 ms
    pad_mode="constant",
) -> np.ndarray:
    """
    Extract MFCC features from an audio file.

    Args:
    file_path: str, path to the audio file
    n_mfcc: int, number of MFCC features to extract
    win_length: int, frame size, how many samples in each frame.
    hop_length: int, frame step, amount of overlap between adjacent frames.
    """

    # Load audio file
    # y: number of samples in the audio file
    # sr: sampling rate (Hz or #sample/s), the number of samples per second of audio
    # duration (s) = y / sr
    # y should be equal to df_transcript['end_sample'].loc[0, 'end_sample']
    y, sr = librosa.load(file_path, sr=desired_sr)

    # Extract MFCC features
    mfcc = librosa.feature.mfcc(
        y=y,
        sr=sr,
        n_mfcc=n_mfcc,
        win_length=win_length,
        hop_length=hop_length,
        pad_mode=pad_mode,
    )

    return mfcc


def _construct_mfcc_df(
    mfcc_matrix: np.array, win_length: int = 1024, hop_length: int = 512
) -> pd.DataFrame:
    """
    Convert a matrix of MFCC features to a pandas DataFrame.
    """

    num_frames = mfcc_matrix.shape[1]
    mfcc_df = pd.DataFrame(columns=["start_sample", "end_sample", "mfcc"])
    mfcc_df["start_sample"] = [i * hop_length for i in range(num_frames)]
    mfcc_df["end_sample"] = [i * hop_length + win_length for i in range(num_frames)]
    mfcc_df["mfcc"] = [mfcc_matrix[:, i] for i in range(num_frames)]

    return mfcc_df


def process_audio_file(
    audio_file_path: str,
    cut_final_frame: bool = True,
    win_length: int = 1024,
    hop_length: int = 512,
) -> pd.DataFrame:
    """
    Process an audio file to extract MFCC features.
    Args:
        audio_file_path: str, path to the audio file
        cut_final_frame: bool, whether to cut the final frame of the audio file
            we want to cut the final frame because it's not complete
    """

    # Extract MFCC features
    mfcc = _load_mfcc(audio_file_path, win_length=win_length, hop_length=hop_length)
    if cut_final_frame:
        mfcc = mfcc[:, :-1]

    # Construct DataFrame
    mfcc_df = _construct_mfcc_df(mfcc, win_length=win_length, hop_length=hop_length)

    return mfcc_df


def label_df_mfcc(df_mfcc, df_phoneme):
    """
    Given two dataframes df_mfcc, df_phoneme, create phoneme labels in df_mfcc by using the labels in df_phoneme.
    Parameters:
        df_mfcc: pd.dataframe. Unlabeled mfcc features.
        df_phoneme: pd.dataframe. Dataset that includes phoneme labels and their start-end times.

    Returns:
        new_df_mfcc. Modify the df_mfcc dataset and create labels.
    """
    new_df = df_mfcc.copy()
    import pandas as pd

    # Add a new column for the phoneme labels
    new_df["phoneme"] = ""
    new_df["first_phoneme"] = ""

    # Function to find the phoneme label for each row in new_df
    def find_phoneme(start, end):
        # Filter df_phoneme to find rows where the time interval overlaps with the mfcc interval
        overlaps = df_phoneme[
            (df_phoneme["start_sample"] < end) & (df_phoneme["end_sample"] > start)
        ]

        if overlaps.empty:
            # Default return if there is no overlap
            print(f"For start {start} and end {end}, there is no time-overlapping row.")
            return "unlabeled", "unlabeled"

        # Check if there is any overlap
        phonemes = overlaps["phoneme"].tolist()
        # Return all unique phonemes preserving their order
        seen = set()
        unique_phonemes = []
        for phoneme in phonemes:
            if phoneme not in seen:
                seen.add(phoneme)
                unique_phonemes.append(phoneme)
        return ", ".join(unique_phonemes), unique_phonemes[0]

    # Apply the function to each row in df_mfcc
    # Apply the function to each row in df_mfcc and expand the results into the appropriate columns
    results = new_df.apply(
        lambda row: find_phoneme(row["start_sample"], row["end_sample"]),
        axis=1,
        result_type="expand",
    )
    new_df["phoneme"] = results[0]
    new_df["first_phoneme"] = results[1]
    return new_df


def vectorize_label_df_mfcc(df_mfcc, df_phoneme):
    """
    Given a dataframe that contains MFCC and phoneme labels as string, create vector labels.
    Parameters:
        df_mfcc: pd.dataframe. That contains MFCC and start-end times of frames.
        df_phoneme: pd.dataframe. Contains the phoneme labels and their start-end times.
    Return:
        new_df: New dataframe that contains the labels for each dataframe.
    """
    new_df = df_mfcc.copy()

    def find_state_weights(start, end):
        # Filter df_phoneme to find rows where the time interval overlaps with the mfcc interval
        overlaps = df_phoneme[
            (df_phoneme["start_sample"] < end) & (df_phoneme["end_sample"] > start)
        ]

        # If there is no overlap
        if overlaps.empty:
            print(
                f"Caution: There is no time-overlapping rows for start {start} and end {end}"
            )
            return None

        # Calculate overlap for each phoneme
        overlaps["overlap_length"] = overlaps.apply(
            lambda row: min(end, row["end_sample"]) - max(start, row["start_sample"]),
            axis=1,
        )
        total_overlap = overlaps["overlap_length"].sum()

        # Calculate and return overlap weights normalized to sum to 1
        if total_overlap <= 0:
            raise ValueError("total_overlap is not larger than zero.")

        overlaps["weights"] = overlaps["overlap_length"] / total_overlap
        # Return a dictionary with phonemes as keys and their weights as values
        weight_dic = {}
        for i, row in overlaps.iterrows():
            if row["phoneme"] not in weight_dic:
                weight_dic[row["phoneme"]] = row["weights"]
            else:
                weight_dic[row["phoneme"]] += row["weights"]

        return weight_dic

    new_df["state_weights"] = new_df.apply(
        lambda row: find_state_weights(row["start_sample"], row["end_sample"]), axis=1
    )
    new_df = new_df[new_df["state_weights"].notna()]
    """
    def distribute_weights(weight_dic):
        We do not use this function in the new version of state definitions (start, mid, end)
        Distribute the weights of phonemes if there are intersections.
        
        # Do not distribute if there is only one label.
        if len(weight_dic) == 1:
            return weight_dic
        
        # Do not distribute if the only labels are silence h# and background #b. 
        if set(weight_dic.keys()) == {'h#, #b'}:
            return weight_dic
        
        # Distribute: Divide the probability of each label by 2.
        # Set the probability of the label that is at the intersection to 0.5. 
        new_label = []
        for key, val in weight_dic.items():
            new_label.append(key)
            weight_dic[key] /= 2
        
        new_label_str = ''.join(new_label)
        weight_dic[new_label_str] = 0.5
        return weight_dic
    """

    def create_label_vector(weight_dic):
        # Create label vector that has the same dimension as the number of phoneme classes.
        num_labels = len(config.PHONEME_DICT)
        label = np.zeros(num_labels)
        for key, val in weight_dic.items():
            if key in config.PHONEME_DICT:
                idx = config.PHONEME_DICT[key]
                label[idx] = val
            else:
                raise ValueError(f"State {key} does not exist in PHONEME_DICT")
        return label

    new_df["label"] = new_df.apply(
        lambda row: create_label_vector(row["state_weights"]), axis=1
    )
    return new_df


def prepare_data(
    phn_path: str,
    wav_path: str,
    win_length: int = 400,
    hop_length: int = 80,
) -> pd.DataFrame:
    """
    get the labeled mfcc dataframe from one audio file
    """

    # two paths should be the same except for the file extension
    # use regular expression

    if phn_path[:-3] != wav_path[:-3]:
        raise ValueError(
            "The two paths should be the same except for the file extension."
        )

    df_phn = load_data(phn_path, "phoneme")
    df_phn = split_phoneme(df_phn)
    df_mfcc = process_audio_file(wav_path, win_length=win_length, hop_length=hop_length)
    df_label = label_df_mfcc(df_mfcc, df_phn)
    df_label = vectorize_label_df_mfcc(df_label, df_phn)

    df_label = df_label[["mfcc", "label", "state_weights"]]

    return df_label


def concat_df(
    keyword: str = "never",
    dataset_type: str = "train",
    win_length: int = 400,
    hop_length: int = 80,
):
    """
    Main function to get the prior and transition probability of the phonemes
    for training dataset.
    """

    if dataset_type:
        # list of tuples
        dataset = get_train_test_paths(keyword)[dataset_type]

    concat_df = None

    for path_tup in dataset:
        wav_path = path_tup[0]
        phn_path = path_tup[1]

        df = prepare_data(
            phn_path, wav_path, win_length=win_length, hop_length=hop_length
        )

        # append df to concat_df vertically
        if concat_df is None:
            concat_df = df
        else:
            concat_df = pd.concat([concat_df, df], ignore_index=True)

    # save the dataframe
    concat_df.to_csv(f"processed_data/dnn_{keyword}_{dataset_type}.csv", index=False)

    return concat_df

def main(
    keyword: str = "never",
    dataset_type: str = ["train", "test"],
    win_length: int = 400,
    hop_length: int = 80,
    rerun: bool = False,
):
    """
    Main function to get the ground truth labels of the MFCC features
    """
    if not rerun:
        try:
            # Check if the data has been processed
            for dt in dataset_type:
                pd.read_csv(f"processed_data/dnn_{keyword}_{dt}.csv")
            return None
        except:
            pass
    for dt in dataset_type:
        concat_df(keyword, dt, win_length, hop_length)

    return None