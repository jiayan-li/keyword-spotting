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

    if data_type == 'phoneme':
        target_phoneme = ['epi', 'pau', 'h#', 'n', 'eh', 'v', 'axr']
        df['phoneme'] = df['phoneme'].apply(lambda x: "#b" if x not in target_phoneme else x)
        df['phoneme'] = df['phoneme'].apply(lambda x: "h#" if x in ['epi', 'pau'] else x)

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

def label_df_mfcc(df_mfcc, df_phoneme):
    '''
    Given two dataframes df_mfcc, df_phoneme, create phoneme labels in df_mfcc by using the labels in df_phoneme.
    Parameters:
        df_mfcc: pd.dataframe. Unlabeled mfcc features.
        df_phoneme: pd.dataframe. Dataset that includes phoneme labels and their start-end times.

    Returns:
        None. Modify the df_mfcc dataset and create labels.
    '''

    import pandas as pd
    # Add a new column for the phoneme labels
    df_mfcc['phoneme'] = ''
    # Function to find the phoneme label for each row in df_mfcc
    def find_phoneme(start, end):
        # I assume there are at most 2 rows where time interval overlaps.
        # Filter df_phoneme to find rows where the time interval overlaps with the mfcc interval
        overlaps = df_phoneme[(df_phoneme['start_sample'] <= end) & (df_phoneme['end_sample'] >= start)]
        
        # Check if there is any overlap
        if not overlaps.empty:
            phonemes = overlaps['phoneme'].tolist()
            # Return all unique phonemes preserving their order
            seen = set()
            unique_phonemes = []
            for phoneme in phonemes:
                if phoneme not in seen:
                    seen.add(phoneme)
                    unique_phonemes.append(phoneme)
            return ', '.join(unique_phonemes)
        
        # Default return if there is no overlap
        print(f'For start {start} and end {end}, there is no time-overlapping row.')
        return 'unlabeled'  # Or some other default value if no suitable match is found

    # Apply the function to each row in df_mfcc
    df_mfcc['phoneme'] = df_mfcc.apply(lambda row: find_phoneme(row['start_sample'], row['end_sample']), axis=1)

    def find_ratio(start, end):
        # Filter df_phoneme to find rows where the time interval overlaps with the mfcc interval
        overlaps = df_phoneme[(df_phoneme['start_sample'] <= end) & (df_phoneme['end_sample'] >= start)]

        # If there is any overlap
        if not overlaps.empty:
            # Calculate overlap for each phoneme
            overlaps['overlap_length'] = overlaps.apply(
                lambda row: min(end, row['end_sample']) - max(start, row['start_sample']), axis=1
            )
            total_overlap = overlaps['overlap_length'].sum()
            
            # Calculate and return overlap weights normalized to sum to 1
            if total_overlap > 0:
                overlaps['weights'] = overlaps['overlap_length'] / total_overlap
                return overlaps['weights'].tolist()
        
        # Default return if there is no overlap or if total overlap sum is zero
        return [0]  # Or some other default value if no suitable match is found



