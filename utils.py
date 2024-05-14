"""
utils functions used by other scripts to preprocess the files
split files containing keyword into train and test
"""

import pandas as pd
import re
from joblib import load, dump
from typing import List, Optional, Tuple, Dict
import random
import glob

def _list_to_df(data: list, variable_name: str) -> pd.DataFrame:
    """
    Convert a list of lists to a pandas DataFrame.
    """

    # Create DataFrame
    df = pd.DataFrame(data, columns=["start_sample", "end_sample", variable_name])

    # Convert start_sample and end_sample columns to integers
    df["start_sample"] = df["start_sample"].astype(int)
    df["end_sample"] = df["end_sample"].astype(int)
    df["diff_sample"] = df["end_sample"] - df["start_sample"]

    return df


def _string_to_list(string: str) -> list:
    """
    Convert a string to a list of strings.
    """

    pattern = r"(\d+)\s(\d+)\s(.+)"

    match = re.match(pattern, string)
    if match:
        group1 = match.group(1)
        group2 = match.group(2)
        group3 = match.group(3)

        return [[group1, group2, group3]]
    else:
        raise ValueError("Transcript data not parsed correctly.")


def load_data(file_path: str, data_type: str) -> pd.DataFrame:
    """
    Load time-aligned phoneme or word data.
        type: str, one of "phoneme", "word"
        Ex: df_phoneme = load_data('timit/data/TRAIN/DR4/MGAG0/SI2209.PHN', 'phoneme')
            df_word = load_data('timit/data/TRAIN/DR4/MGAG0/SI2209.WRD', 'word')
    """

    if data_type not in ["phoneme", "word"]:
        raise ValueError('data_type must be one of "phoneme" or "word"')

    # read the phonemes
    with open(file_path, "r") as file:
        phn = file.readlines()
        parsed_data = [line.split() for line in phn]

        # Create DataFrame
        df = _list_to_df(parsed_data, data_type)

    if data_type == "phoneme":
        target_phoneme = ["epi", "pau", "h#", "n", "eh", "v", "axr"]
        df["phoneme"] = df["phoneme"].apply(
            lambda x: "#b" if x not in target_phoneme else x
        )
        df["phoneme"] = df["phoneme"].apply(
            lambda x: "h#" if x in ["epi", "pau"] else x
        )

    return df


def load_transcript(file_path: str) -> str:
    """
    Load the transcript data.
        Ex: df_transcript = load_transcript('timit/data/TRAIN/DR4/MGAG0/SX61.TXT')
    """

    # read the transcript
    with open(file_path, "r") as file:
        transcript = file.read()

        transcript_list = _string_to_list(transcript)
        transcript = _list_to_df(transcript_list, "transcript")

        return transcript


def split_phoneme(df_phoneme: pd.DataFrame) -> pd.DataFrame:
    """
    if the phoneme is one of 'n', 'eh', 'v', 'axr', split into three rows
    with equal gap between the start_sample and end_sample
    """

    index_to_drop = []
    rows_to_add = []

    # iterate through the rows
    for i, row in df_phoneme.iterrows():
        if row["phoneme"] in ["n", "eh", "v", "axr"]:
            # later to drop the row
            index_to_drop.append(i)

            # calculate the gap
            gap = row["diff_sample"] / 3

            # split the row into three rows
            row1 = [
                row["start_sample"],
                row["start_sample"] + gap,
                "b-" + row["phoneme"],
                row["diff_sample"],
            ]
            row2 = [row1[1], row1[1] + gap, "m-" + row["phoneme"], row1[3]]
            row3 = [row2[1], row["end_sample"], "e-" + row["phoneme"], row2[3]]

            rows_to_add.append(row1)
            rows_to_add.append(row2)
            rows_to_add.append(row3)

        else:
            pass

    # drop the rows
    df_phoneme = df_phoneme.drop(index_to_drop)

    # add the rows
    df_phoneme = pd.concat(
        [df_phoneme, pd.DataFrame(rows_to_add, columns=df_phoneme.columns)],
        ignore_index=True,
    )

    # sort by start_sample
    df_phoneme = df_phoneme.sort_values("start_sample").reset_index(drop=True)

    return df_phoneme


def _seperate_string(input_string: str) -> list:

    # Define the regex pattern to capture the sentence and the substring in parentheses
    pattern = r"^(.*?)\s*\((.*?)\)$"

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
    pattern = r"\b" + keyword + r"\b"

    # Search for the keyword in the prompt
    matches = re.search(pattern, prompt)

    # Return True if the keyword is found, False otherwise
    return matches is not None


def _get_path_list(prompt_id: str, file_type: Optional[str] = "PHN") -> List[str]:

    if file_type is None:
        return glob.glob(f"timit/data/*/*/*/{prompt_id}.*")

    prompt_id = prompt_id.upper()
    file_type = file_type.upper()

    return glob.glob(f"timit/data/*/*/*/{prompt_id}.{file_type}")


def _count_phoneme(phoneme: str, prompt_id: str) -> int:
    """
    see if the phoneme is in the prompt
    """

    path = _get_path_list(prompt_id, "PHN")[0]
    df = load_data(path, "phoneme")
    # count the number of target phoneme
    count = df[df["phoneme"] == phoneme].shape[0]

    return count

def _get_prompt_id_list(df: pd.DataFrame) -> List[str]:
    """
    Get the list of prompt_id that contain the keyword
    """

    # upper case the string
    df["prompt_id"] = df["prompt_id"].str.upper()
    prompt_id = df[df["contain_keyword"] == True].prompt_id.to_list()

    return prompt_id


def _shuffle_and_split(input_list: list, train_ratio: float = 0.5) -> Tuple[List, List]:
    # Shuffle the input list randomly
    random.shuffle(input_list)

    # Calculate the split index
    split_index = int(len(input_list) * train_ratio)

    # Split the list into two parts
    part1 = input_list[:split_index]
    part2 = input_list[split_index:]

    return part1, part2


def process_all_prompts(
    keyword: str = "never",
    tartget_phoneme: List[str] = ["n", "eh", "v", "axr"],
    prompts_file_path: str = "timit/PROMPTS.txt",
    rerun: bool = False,
) -> pd.DataFrame:
    """
    generate a dataframe that contains the following columns:
    prompt, prompt_id, file_count, contain_keyword, n, eh, v, axr
    """

    if not rerun:
        try:
            df = pd.read_csv("processed_data/prompts_info.csv")
            return df
        except FileNotFoundError:
            pass

    with open(prompts_file_path, "r") as file:
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
            phoneme_count = [
                _count_phoneme(phoneme, prompt_id) for phoneme in tartget_phoneme
            ]

            extracted.extend([file_count, contain_keyword])
            extracted.extend(phoneme_count)
            result.append(extracted)
            # print(extracted)

    columns = ["prompt", "prompt_id", "file_count", "contain_keyword"]
    columns.extend(tartget_phoneme)

    # Create a dataframe
    df = pd.DataFrame(result, columns=columns)

    # save
    df.to_csv("processed_data/prompts_info.csv", index=False)

    return df


def get_all_paths(
    keyword: str = "never", file_type: Optional[str] = "wav"
) -> List[str]:
    """
    Get the list of paths of the files of the specified type that contain the keyword
    """
    try:
        if file_type == "wav":
            path_list = load(f"processed_data/wav_paths_{keyword}.joblib")
        elif file_type == "phn":
            path_list = load(f"processed_data/phn_paths_{keyword}.joblib")
        else:
            raise ValueError("Invalid file_type. Supported types are 'wav' and 'phn'.")

    except:
        df = process_all_prompts(keyword)  # Assuming this function is defined elsewhere
        prompt_id = _get_prompt_id_list(
            df
        )  # Assuming this function is defined elsewhere
        path_list = []
        for prompt_id in prompt_id:
            path_list.extend(
                _get_path_list(prompt_id, file_type)
            )  # Assuming this function is defined elsewhere

        dump(path_list, f"processed_data/{file_type}_paths_{keyword}.joblib")

    return path_list

def get_train_test_paths(
    keyword: str = "never", train_ratio: float = 0.5, rerun: bool = False
) -> Dict[str, List[Tuple]]:
    """
    main function to get train test set from the files that contain the keyword

    return:
        dataset: a dictionary with two keys 'train' and 'test'.
    """

    if not rerun:
        try:
            dataset = load(f"processed_data/train_test_dataset_{keyword}.joblib")
            return dataset
        except:
            pass

    keyword_wav_paths = get_all_paths(keyword, "wav")
    keyword_root_paths = [path[:-4] for path in keyword_wav_paths]

    # Split the list of paths into training and testing sets
    train_root_paths, test_root_paths = _shuffle_and_split(
        keyword_root_paths, train_ratio
    )

    dataset = {
        "train": [(p + ".WAV", p + ".PHN", p + ".WRD") for p in train_root_paths],
        "test": [(p + ".WAV", p + ".PHN", p + ".WRD") for p in test_root_paths],
    }

    dump(dataset, f"processed_data/train_test_dataset_{keyword}.joblib")

    return dataset