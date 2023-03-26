import numpy as np
import pandas as pd


def mean_imputation(data):
    """
    Impute missing values in the input DataFrame by filling in the mean value of each column.

    Parameters:
    data (pandas.DataFrame): Input data with missing values.

    Returns:
    pandas.DataFrame: A copy of the input data with missing 
                        values imputed with the mean value of each column.
    """
    data1 = data.copy()
    mean = np.mean(data1, axis=0)
    for i in range(2, data1.shape[1]):
        data1.iloc[:, i].fillna(mean[i-1], inplace=True)
    return data1


def normalize(data):
    """
    Normalize the input data by subtracting the mean and dividing by 
    the standard deviation of each feature to center the input data around
    zero and scaling it to have unit variance.

    Parameters:
    data (pandas.DataFrame): Input data to be normalized.

    Returns:
    Tuple: A tuple containing the normalized features, the original labels, 
            the mean of each feature, and the standard deviation of each feature.
    """
    y = data['diagnosis']
    X = data.drop(['diagnosis'], axis=1)
    mean = np.mean(X, axis=0)
    stddev = np.std(X, axis=0)
    X = (X - mean) / stddev
    df = pd.concat([X, y], axis=1)
    return df, mean, stddev


def train_test_split(df, train_size=0.67, shuffle=True, random_state=42):
    """
    Splits a DataFrame into training and test sets.

    Args:
        df (pandas.DataFrame): The DataFrame to split.
        train_size (float, optional): The fraction of the data to include in the training set. Defaults to 0.67.
        shuffle (bool, optional): Whether to shuffle the data before splitting. Defaults to True.
        random_state (int, optional): The random seed to use for shuffling the data. Defaults to 42.

    Returns:
        tuple: A tuple containing the following four elements:
            X_train (pandas.DataFrame): The feature matrix of the training set.
            X_test (pandas.DataFrame): The feature matrix of the test set.
            y_train (pandas.Series): The target series of the training set.
            y_test (pandas.Series): The target series of the test set.

    Raises:
        ValueError: If train_size is not between 0 and 1.
    
    This function splits the normalized data into training and test sets according to the 
    specified train_size. If shuffle is True, the data is randomly shuffled before splitting. 
    If random_state is given, it is used to seed the random number generator for reproducibility.
    """
    if (shuffle):
        df = df.sample(frac=1, random_state=random_state)
    y = df['diagnosis'].map({'M': 1, 'B': -1})
    X = df.drop(['diagnosis'], axis=1)
    split_idx = int(len(df) * train_size)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    return X_train, X_test, y_train, y_test


def evaluate(y_true, y_pred, verbose = True):
    """
    Computes and prints the confusion matrix, accuracy, precision, and recall
    for a given set of true and predicted target values.

    Args:
        y_true (numpy array): An array of true target values.
        y_pred (numpy array): An array of predicted target values.

    Returns:
        None

    Raises:
        None
    """
    confusion_matrix = {"true_positive": 0, "true_negative": 0,
                        "false_positive": 0, "false_negative": 0}

    for true_value, pred_value in zip(y_true, y_pred):
        if true_value == 1 and pred_value == 1:
            confusion_matrix["true_positive"] += 1
        elif true_value == -1 and pred_value == -1:
            confusion_matrix["true_negative"] += 1
        elif true_value == -1 and pred_value == 1:
            confusion_matrix["false_positive"] += 1
        elif true_value == 1 and pred_value == -1:
            confusion_matrix["false_negative"] += 1

    accuracy = 100*(confusion_matrix["true_positive"] +
                    confusion_matrix["true_negative"]) / len(y_true)

    if confusion_matrix["true_positive"] == 0 and confusion_matrix["false_positive"] == 0:
        precision = 0
    else:
        precision = 100*confusion_matrix["true_positive"] / (
            confusion_matrix["true_positive"] + confusion_matrix["false_positive"])

    if confusion_matrix["true_positive"] == 0 and confusion_matrix["false_negative"] == 0:
        recall = 0
    else:
        recall = 100*confusion_matrix["true_positive"] / (
            confusion_matrix["true_positive"] + confusion_matrix["false_negative"])
    if (verbose == True):
        print(f"Confusion Matrix: {confusion_matrix}")
        print(f"Accuracy: {accuracy}%")
        print(f"Precision: {precision}%")
        print(f"Recall: {recall}%")
    return accuracy
