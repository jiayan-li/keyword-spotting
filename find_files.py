"""
functions to preprocess the files
"""

import re
import pandas as pd
from typing import List, Optional, Tuple, Dict
from utils import load_data
import glob
import random
from joblib import load, dump

def _seperate_string(input_string: str) -> list:
    
    # Define the regex pattern to capture the sentence and the substring in parentheses
    pattern = r'^(.*?)\s*\((.*?)\)$'

    # Apply the pattern to the input string
    matches = re.match(pattern, input_string)

    # Extract the sentence and the substring
    if matches:
        sentence = matches.group(1)
        substring = matches.group(2)

        # Create the list
        result = [sentence, substring]
        
        # Print the result
        return result
    else:
        return None
    
def _search_keyword(keyword: str, prompt: str) -> bool:
    """
    the keyword has to be a whole word
    """

    # Define the regex pattern
    pattern = r'\b' + keyword + r'\b'

    # Search for the keyword in the prompt
    matches = re.search(pattern, prompt)

    # Return True if the keyword is found, False otherwise
    return matches is not None


def _get_path_list(prompt_id: str, 
                   file_type: Optional[str] = 'PHN') -> List[str]:
    
    if file_type is None:
        return glob.glob(f'timit/data/*/*/*/{prompt_id}.*')

    prompt_id = prompt_id.upper()
    file_type = file_type.upper()

    return glob.glob(f'timit/data/*/*/*/{prompt_id}.{file_type}')
    
def _count_phoneme(phoneme: str, prompt_id: str) -> int:
    """
    see if the phoneme is in the prompt
    """

    path = _get_path_list(prompt_id, 'PHN')[0]
    df = load_data(path, 'phoneme')
    # count the number of target phoneme
    count = df[df['phoneme']==phoneme].shape[0]

    return count

def process_all_prompts(
        keyword: str = 'never',
        tartget_phoneme: List[str] = ['n', 'eh', 'v', 'axr'],
        prompts_file_path: str = 'timit/PROMPTS.txt',
        rerun: bool = False) -> pd.DataFrame:
    """
    generate a dataframe that contains the following columns:
    prompt, prompt_id, file_count, contain_keyword, n, eh, v, axr
    """

    if not rerun:
        try:
            df = pd.read_csv('processed_data/prompts_info.csv')
            return df
        except FileNotFoundError:
            pass

    with open(prompts_file_path, 'r') as file:
        prompt = file.readlines()
        # remove the newline character
        prompt = [line.strip() for line in prompt]

    # Create a list to store the results
    result = []
    for line in prompt:
        # separate the prompt and prompt_id
        extracted = _seperate_string(line)
        
        # if there's a prompt in the line
        if extracted:
            prompt = extracted[0]
            prompt_id = extracted[1]

            # get the number of files
            file_count = len(_get_path_list(prompt_id)) 
            # check if the keyword is in the prompt
            contain_keyword = _search_keyword(keyword, prompt)
            # count the number of target phoneme
            phoneme_count = [_count_phoneme(phoneme, prompt_id) 
                                    for phoneme in tartget_phoneme]
            
            extracted.extend([file_count, contain_keyword])
            extracted.extend(phoneme_count)
            result.append(extracted)
            # print(extracted)

    columns=['prompt', 'prompt_id', 'file_count', 'contain_keyword']
    columns.extend(tartget_phoneme)

    # Create a dataframe
    df = pd.DataFrame(result, columns=columns)

    # save
    df.to_csv('processed_data/prompts_info.csv', index=False)

    return df

def _get_prompt_id_list(df: pd.DataFrame) -> List[str]:
    """
    Get the list of prompt_id that contain the keyword
    """

    # upper case the string
    df['prompt_id'] = df['prompt_id'].str.upper()
    prompt_id = df[df['contain_keyword']==True].prompt_id.to_list()

    return prompt_id

def _shuffle_and_split(input_list: list,
                       train_ratio: float = 0.5) -> Tuple[List, List]:
    # Shuffle the input list randomly
    random.shuffle(input_list)

    # Calculate the split index
    split_index = int(len(input_list) * train_ratio)
    
    # Split the list into two parts
    part1 = input_list[:split_index]
    part2 = input_list[split_index:]
    
    return part1, part2

def get_all_paths(keyword: str = 'never', 
                  file_type: Optional[str] = 'wav') -> List[str]:
    """
    Get the list of paths of the files of the specified type that contain the keyword
    """
    try:
        if file_type == 'wav':
            path_list = load(f'processed_data/wav_paths_{keyword}.joblib')
        elif file_type == 'phn':
            path_list = load(f'processed_data/phn_paths_{keyword}.joblib')
        else:
            raise ValueError("Invalid file_type. Supported types are 'wav' and 'phn'.")

    except:
        df = process_all_prompts(keyword)  # Assuming this function is defined elsewhere
        prompt_id = _get_prompt_id_list(df)  # Assuming this function is defined elsewhere
        path_list = []
        for prompt_id in prompt_id:
            path_list.extend(_get_path_list(prompt_id, file_type))  # Assuming this function is defined elsewhere

        dump(path_list, f'processed_data/{file_type}_paths_{keyword}.joblib')

    return path_list

def get_train_test_paths(keyword: str = 'never',
                         train_ratio: float = 0.5,
                         rerun: bool = False
                         ) -> Dict[str, List[Tuple]]:
    
    """
    main function to get train test set from the files that contain the keyword

    return:
        dataset: a dictionary with two keys 'train' and 'test'.
    """

    if not rerun:
        try:
            dataset = load(f'processed_data/train_test_dataset_{keyword}.joblib')
            return dataset
        except:
            pass

    keyword_wav_paths = get_all_paths(keyword, 'wav')
    keyword_root_paths = [path[:-4] for path in keyword_wav_paths]

    # Split the list of paths into training and testing sets
    train_root_paths, test_root_paths = _shuffle_and_split(
                                            keyword_root_paths, 
                                            train_ratio)
    
    dataset = {
        'train': [(p+'.WAV', p+'.PHN') for p in train_root_paths],
        'test': [(p+'.WAV', p+'.PHN') for p in test_root_paths]
        }

    dump(dataset, f'processed_data/train_test_dataset_{keyword}.joblib')

    return dataset