import pandas as pd
import numpy as np
import re
import librosa

def _list_to_df(data: list, variable_name: str) -> pd.DataFrame:
    """
    Convert a list of lists to a pandas DataFrame.
    """
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=['start_sample', 'end_sample', variable_name])

    # Convert start_sample and end_sample columns to integers
    df['start_sample'] = df['start_sample'].astype(int)
    df['end_sample'] = df['end_sample'].astype(int)
    df['diff_sample'] = df['end_sample'] - df['start_sample']

    return df

def _string_to_list(string: str) -> list:
    """
    Convert a string to a list of strings.
    """
    
    pattern = r'(\d+)\s(\d+)\s(.+)'

    match = re.match(pattern, string)
    if match:
        group1 = match.group(1)
        group2 = match.group(2)
        group3 = match.group(3)

        return [[group1, group2, group3]]
    else:
        raise ValueError('Transcript data not parsed correctly.')

def load_data(file_path: str, data_type: str) -> pd.DataFrame:
    """
    Load time-aligned phoneme or word data.
        type: str, one of "phoneme", "word"
        Ex: df_phoneme = load_data('timit/data/TRAIN/DR4/MGAG0/SI2209.PHN', 'phoneme')
            df_word = load_data('timit/data/TRAIN/DR4/MGAG0/SI2209.WRD', 'word')
    """

    if data_type not in ['phoneme', 'word']:
        raise ValueError('data_type must be one of "phoneme" or "word"')

    # read the phonemes
    with open(file_path, 'r') as file:
        phn = file.readlines()
        parsed_data = [line.split() for line in phn]
        
        # Create DataFrame
        df = _list_to_df(parsed_data, data_type)

    return df
   
def load_transcript(file_path: str) -> str:
    """
    Load the transcript data.
        Ex: df_transcript = load_transcript('timit/data/TRAIN/DR4/MGAG0/SX61.TXT')
    """

    # read the transcript
    with open(file_path, 'r') as file:
        transcript = file.read()

        transcript_list = _string_to_list(transcript)
        transcript = _list_to_df(transcript_list, 'transcript')

        return transcript

def _load_mfcc(file_path: str, 
            desired_sr: int = 16000,       # 16 kHz, sampling rate of phonems/words/transcripts
            n_mfcc: int = 20,
            win_length: int = 1024,        # 64 ms
            hop_length: int = 512,         # 32 ms
            pad_mode='constant'
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
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc,
                                win_length=win_length, hop_length=hop_length,
                                pad_mode=pad_mode)

    return mfcc

def _construct_mfcc_df(mfcc_matrix: np.array,
                    win_length: int = 1024,
                    hop_length: int = 512) -> pd.DataFrame:
    """
    Convert a matrix of MFCC features to a pandas DataFrame.
    """

    num_frames = mfcc_matrix.shape[1]
    mfcc_df = pd.DataFrame(columns=['start_sample', 'end_sample', 'mfcc'])
    mfcc_df['start_sample'] = [i * hop_length for i in range(num_frames)]
    mfcc_df['end_sample'] = [i * hop_length + win_length for i in range(num_frames)]
    mfcc_df['mfcc'] = [mfcc_matrix[:, i] for i in range(num_frames)]

    return mfcc_df

def process_audio_file(audio_file_path: str,
                       cut_final_frame: bool = True,
                       win_length: int = 1024,
                       hop_length: int = 512) -> pd.DataFrame:
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