import pandas as pd
import datasets
from collections import Counter


def get_data(dataset_name="ucberkeley-dlab/measuring-hate-speech", columns=["text", "hatespeech"]):
    """
    Helper method which fetches the requested dataset, narrows it down to the
    relevant columns, aggregates second column to the most frequent value
    based on the first column, and returns it

    Parameters
    ----------
    dataset_name : str, optional
        Name of the dataset to be downloaded. For this project, the default
        value is "ucberkeley-dlab/measuring-hate-speech".
    columns : list, optional
        A list of columns to be extracted. For this project, the default value
        is  ["text", "hatespeech"].

    Returns
    -------
    data : pandas.DataFrame
        The fetched and processed dataset.

    """
    print("Fetching data...")
    dataset = datasets.load_dataset(dataset_name, "binary")
    data = dataset["train"].to_pandas()[columns]

    print("Processing...")
    data[columns[1]] = pd.to_numeric(
        data[columns[1]],
        downcast="integer"
    )

    data.loc[data[columns[1]] == 2, columns[1]] = 1

    data = data.groupby(columns[0]).agg(
        lambda x: Counter(x).most_common(1)[0][0]
    ).reset_index()

    print("Done!")
    return data


def clean_text(text):
    BAD_WORDS = {
        "nigga": "n***a",
        "fuck": "f**k",
        "bitch": "b***h",
        "dick": "d**k",
        "cock": "c**k",
        "ass": "a**",
        "pussy": "p***y",
        "sex": "s**",
        "nigger": "n****r",
        "faggot": "f****t",
        "slut": "s**t",
        "shit": "s**t",
        "retard": "r****d",
        "killed": "k****d",
        "suck": "s**k",
        "hoe": "h**",
        "ugly": "u**y",
        "nazi": "n**i",
        "cunt": "c**t",
        "cum": "c**"
    }
    text = text.lower()
    for word, replacement in BAD_WORDS.items():
        text = text.replace(word, replacement)
    return text