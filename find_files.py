import re
import pandas as pd
from typing import List, Optional
from data_utils import load_data
import glob

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
        prompts_file_path: str = 'timit/PROMPTS.txt') -> pd.DataFrame:

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

def get_all_paths(keyword: str = 'never', 
                  file_type: Optional[str] = 'wav'
                        ) -> List[str]: 
    """
    Get the list of path of the files of specified type that contain the keyword
    """
    
    df = process_all_prompts(keyword)
    prompt_id = _get_prompt_id_list(df)
    path_list = []
    for id in prompt_id:
        path = _get_path_list(id, file_type)
        path_list.extend(path)

    return path_list