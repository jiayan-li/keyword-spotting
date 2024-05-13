"""
utils functions used by other scripts
"""

import pandas as pd
import re


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
