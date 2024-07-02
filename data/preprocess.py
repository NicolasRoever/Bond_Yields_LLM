import pandas as pd
import numpy as np
import dspy
from sklearn.model_selection import train_test_split



def create_balanced_subset(df, column, n, seed=42) -> pd.DataFrame:
    """
    Create a balanced subset of the dataframe based on the specified column.

    Parameters:
    df (pd.DataFrame): The input dataframe.
    column (str): The column to balance on.
    n (int): The number of samples in the balanced subset.
    seed (int): The random seed for reproducibility.

    Returns:
    pd.DataFrame: A balanced subset of the input dataframe.
    """
    subsets = []
    unique_values = df[column].unique()
    for value in unique_values:
        subset = df[df[column] == value]
        if len(subset) > n // len(unique_values):
            subset = subset.sample(n=n // len(unique_values), random_state=seed)
        subsets.append(subset)
    balanced_subset = pd.concat(subsets)
    return balanced_subset.sample(frac=1, random_state=seed)


def dataframe_to_examples(df) -> list[dspy.Example]:
    """
    Convert a dataframe to a list of dspy.Example objects.

    Parameters:
    df (pd.DataFrame): The input dataframe.

    Returns:
    list: A list of dspy.Example objects.
    """
    examples = []
    for index, row in df.iterrows():
        examples.append(dspy.Example(excerpt=row['Snippet'], 
                                     country_keyword=row['Keyword'],
                                     snippet_id=row['Snippet_ID'],
                                     sentiment_score=row['Final Combined']).with_inputs('excerpt', 'country_keyword'))
    return examples



def create_dspy_examples_train_test_validation_sets(data, train_size=25, test_size=25, validation_size=50, random_seed=42):
    """
    Create balanced training, testing, and validation sets of dspy.Example objects from a DataFrame.

    Parameters:
    data (pd.DataFrame): The input dataframe containing the dataset.
    train_size (int): The number of examples in the training set.
    test_size (int): The number of examples in the testing set.
    validation_size (int): The number of examples in the validation set.
    random_seed (int): The random seed for reproducibility.

    Returns:
    tuple: A tuple containing three lists of dspy.Example objects for training, testing, and validation.
    """
    
    # Add Relevance_Score_Yes_No column based on Final Relevance Score
    data["Relevance_Score_Yes_No"] = np.where(data["Final Relevance Score"] == 1, 'no', 'yes')

    # Split the data into a training set and a temporary set
    train_data, temp_data = train_test_split(data, train_size=train_size, test_size = (test_size + validation_size), stratify=data['Final Combined'], random_state=random_seed)

    # Split the temporary set into testing and validation sets
    test_data, validation_data = train_test_split(temp_data, train_size = test_size, test_size=validation_size, stratify=temp_data['Final Combined'], random_state=random_seed)

    # Convert the dataframes to lists of dspy.Example objects
    train_examples = dataframe_to_examples(train_data)
    test_examples = dataframe_to_examples(test_data)
    validation_examples = dataframe_to_examples(validation_data)

    return train_examples, test_examples, validation_examples


