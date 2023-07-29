import joblib
import numpy as np
import pandas as pd

import lightgbm as lgb
import plotly.io as pio
import plotly.express as px

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold


pio.renderers.default = 'plotly_mimetype+notebook'


lgb_parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': ['binary_logloss'],
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'seed': 0,
}


def check_missing_columns(df: pd.DataFrame) -> pd.Series:
    """Check if any of the columns in the dataframe contains missing values.

    Args:
        df (pd.DataFrame): dataframe to check.

    Returns:
        pd.Series: series of boolean values to denote whether the column contains missing values.
    """
    return df.isnull().any()


def check_categories(test: pd.DataFrame, train: pd.DataFrame, categorical_fields: list = []) -> list:
    """Check if test dataset contains categorical values not present in train dataset. 

    Args:
        test (pd.DataFrame): test dataset.
        train (pd.DataFrame): train dataset.
        categorical_fields (list, optional): columns to check. Defaults to [].

    Returns:
        list: list of columns in test dataset which contains categorical values not present in train dataset
    """
    columns_missing_categories = []

    for field in categorical_fields:
        unique_train_categories = set(train[field].unique())
        unique_test_categories = set(test[field].unique())

        print(f'Checking categorical column: {field}')
        print(f'Unique categories in train: {unique_train_categories}')
        print(f'Unique categories in test: {unique_test_categories}')

        missing_values = unique_test_categories - unique_train_categories
        print(f'Categories in test but not train: {missing_values}\n')

        if missing_values:
            columns_missing_categories.append(field)
    
    return columns_missing_categories


def convert_to_datetime(df: pd.DataFrame, datetime_col: str = 'request_date', format: str = '%d-%b-%y') -> pd.DataFrame:
    """Convert date string to datetime object in column.

    Args:
        df (pd.DataFrame): dataframe containing at least one date string column.
        datetime_col (str, optional): name of date string column. Defaults to 'request_date'.
        format (str, optional): format of date string. Defaults to '%d-%b-%y'.

    Returns:
        pd.DataFrame: processed dataframe with date string casted to datetime object.
    """
    df[datetime_col] = pd.to_datetime(df[datetime_col], format = format)
    return df


def convert_amt_cols_to_float(df: pd.DataFrame, amt_cols: list = ['loan_amount', 'insured_amount']) -> pd.DataFrame:
    """Convert currency string columns to float

    Args:
        df (pd.DataFrame): dataframe containing currency string columns.
        amt_cols (list, optional): currency string columns. Defaults to ['loan_amount', 'insured_amount'].

    Returns:
        pd.DataFrame: processed dataframe with currency string columns casted to float.
    """
    df = df.copy()
    
    for col in amt_cols:

        # Check that each value in the amount column is a string
        types = df[col].apply(type).value_counts()
        print(types)
        assert len(types) == 1
        assert str(types.index[0]) == "<class 'str'>"

        # Remove $ and , characters
        df[col] = df[col].str.replace(r'[^0-9\.]', '', regex = True).astype(float)

        # Perform sanity check to check if there are any np.nan values after conversion
        assert not df[col].isna().any()

        # Compute the statistics of the values
        print(f'{df[col].describe()}\n')

    return df


def create_loan_insured_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create loan_insured_amount_diff and insured_loan_ratio features.

    Args:
        df (pd.DataFrame): dataframe containing loan_amount and insured_amount columns.

    Returns:
        pd.DataFrame: processed dataframe with loan_insured_amount_diff and insured_loan_ratio features added.
    """
    df['loan_insured_amount_diff'] = df['loan_amount'] - df['insured_amount']
    df['insured_loan_ratio'] = df['insured_amount'] / df['loan_amount']
    return df


def plot_categories_distribution(df: pd.DataFrame, category_col: str, title: str = None, width: int = 500, height: int = 800):
    """Plot distribution of categorical values.

    Args:
        df (pd.DataFrame): dataframe containing categorical columns.
        category_col (str): name of categorical column.
        title (str): title of plot.
        width (int, optional): width of plot. Defaults to 500.
        height (int, optional): height of plot. Defaults to 800.

    Returns:
        Figure: distribution plot of categorical values.
    """
    title = f'Percentage of each {category_col} in train dataset' if title is None else title
    fig = px.bar(
        (df[category_col].astype(str).value_counts() / df.shape[0] * 100).sort_values().reset_index().rename(columns = {'count': 'percentage'}), 
        x = 'percentage', y = category_col, title = title
    )
    fig.update_yaxes(tickmode = 'linear')
    fig.update_layout(width = width, height = height)
    return fig


def plot_category_default_distribution(df: pd.DataFrame, category_col: str, dummy_col: str = 'id', height: int = 800, width: int = 600, title: str = None):
    """Plot probability of default given categorical value.

    Args:
        df (pd.DataFrame): dataframe containing category column.
        category_col (str): name of category column.
        dummy_col (str, optional): dummy column to use for groupby. Defaults to 'id'.
        height (int, optional): height of figure. Defaults to 800.
        width (int, optional): width of figure. Defaults to 600.
        title (str, optional): figure title. Defaults to None.

    Returns:
        Figure: plot of probability given categorical value.
    """
    title = f'Probability of default_status = 1 given {category_col}' if title is None else title
    label_col = 'default_status'
    group_cols = [category_col, label_col]
    required_cols = group_cols + [dummy_col]
    renamed_col = 'Probability'

    df_probs = df[required_cols].groupby(group_cols).count() / df[[category_col, dummy_col]].groupby([category_col]).count()
    df_probs = df_probs.reset_index().rename(columns = {dummy_col: renamed_col})
    df_probs = df_probs[df_probs[label_col] == 1].sort_values(by = renamed_col)
    df_probs[category_col] = df_probs[category_col].astype(str)

    fig = px.bar(df_probs, y = category_col, x = renamed_col, height = height, width = width, title = title)
    fig.update_yaxes(tickmode = 'linear')
    return fig


def encode_categorical_features(train: pd.DataFrame, test: pd.DataFrame, 
                                categorical_features: list = ['industry', 'state', 'business_new', 'business_type', 'other_loans'], 
                                encoder_path: str = './encoder/encoders.pkl') -> tuple:
    """Encode categorical feature values to numerical values.

    Args:
        train (pd.DataFrame): train dataset.
        test (pd.DataFrame): test dataset.
        categorical_features (list, optional): names of categorical feature columns. Defaults to ['industry', 'state', 'business_new', 'business_type', 'other_loans'].
        encoder_path (str, optional): file path to save encoder dict. Defaults to './encoder/encoders.pkl'.

    Returns:
        tuple: encoded train dataset, encoded test dataset, encoder_dict
    """
    encoder_dict = {}

    for feature in categorical_features:
        encoder = LabelEncoder()
        train[feature] = encoder.fit_transform(train[feature])
        test[feature] = encoder.transform(test[feature])
        encoder_dict[feature] = encoder

    joblib.dump(encoder_dict, encoder_path)
    return train, test, encoder_dict


def generate_submissions(model: lgb.Booster, test: pd.DataFrame, features: list, 
                         threshold: int = 0.5, id_col: str = 'id',
                         label_col: str = 'default_status', score_col: str = 'model_score',
                         submissions_path: str = 'submissions_addison.csv'):
    """Run inference on test dataset and save inference results to a csv file.

    Args:
        model (lgb.Booster): fitted model.
        test (pd.DataFrame): test dataframe.
        features (list): list of features used in model training.
        threshold (int, optional): probability threshold for predicting default_status = 1 class. Defaults to 0.5.
        id_col (str, optional): name of id column. Defaults to 'id'.
        label_col (str, optional): name of prediction column. Defaults to 'default_status'.
        submissions_path (str, optional): output csv path of submissions. Defaults to 'submissions_addison.csv'.
    """
    print(f'test shape: {test.shape}')
    submissions_cols = [id_col, label_col]
    scores = model.predict(test[features])
    test[score_col] = scores
    test[label_col] = (scores > threshold).astype(int)
    submissions = test[submissions_cols]
    print(f'Submissions shape: {submissions.shape}')
    submissions.to_csv(submissions_path, index = False)
    print(f'Saved submissions to {submissions_path}.')


def train_lgb_model(train: pd.DataFrame, num_boost_rounds: int, features: list, 
                    cat_features: list, val: pd.DataFrame = None, label_col: str = 'default_status',
                    lgb_parameters: dict = lgb_parameters, model_path: str = None,
                    callbacks: list = None) -> lgb.Booster:
    """Train lightGBM model.

    Args:
        train (pd.DataFrame): dataframe containing train dataset.
        num_boost_rounds (int): number of boosting rounds to train model.
        features (list): full list of features used for model training.
        cat_features (list): list of categorical features used for model training.
        val (pd.DataFrame, optional): dataframe containing validation dataset. Defaults to None.
        label_col (str, optional): name of label column. Defaults to 'default_status'.
        lgb_parameters (dict, optional): training hyperparameters. Defaults to lgb_parameters.
        model_path (str, optional): file path to save model to. Defaults to None.
        callbacks (list, optional): list of callbacks to apply during model training. Defaults to None.

    Returns:
        lgb.Booster: fitted model.
    """
    train_data = lgb.Dataset(train[features], categorical_feature = cat_features, label = train[label_col], free_raw_data = False)
    val_data = None

    if val is not None:
        val_data = lgb.Dataset(val[features], categorical_feature = cat_features, label = val[label_col], free_raw_data = False)

    model = lgb.train(
        lgb_parameters, train_data, valid_sets = val_data, num_boost_round = num_boost_rounds,
        callbacks = callbacks
    )
    if model_path is not None:
        model.save_model(model_path)

    return model


def nested_stratified_kfold_cv(train: pd.DataFrame, features: list, cat_features: list, 
                               num_folds: int = 5, threshold: float = 0.5, 
                               label_col: str = 'default_status', scaling_factor: float = 1) -> dict:
    """Perform nested stratified k-fold cross-validation.

    Args:
        train (pd.DataFrame): train dataframe.
        features (list): full list of features used for model training.
        cat_features (list): list of categorical features used for model training.
        num_folds (int, optional): number of folds for cross validation. Defaults to 5.
        threshold (float, optional): probability threshold for predicting default_status = 1 class. Defaults to 0.5.
        label_col (str, optional): name of label column. Defaults to 'default_status'.

    Returns:
        dict: stratified kfold results.
    """
    skf = StratifiedKFold(n_splits = num_folds, shuffle = True, random_state = 0)

    # Initialize metrics/hyperparameters containers
    val_acc_list = []
    test_acc_list = []
    val_num_boost_rounds_list = []
    test_num_boost_rounds_list = []

    # Outer loop splits the dataset into train (N - 1 partitions) and test (1 partition) datasets
    for train_idx_, test_idx in skf.split(train, train[label_col]):
        train_outer = train.iloc[train_idx_]
        test_outer = train.iloc[test_idx]

        print(f'Outer train shape: {train_outer.shape}')
        print(f'test shape: {test_outer.shape}')

        inner_num_boost_rounds_list = []

        # Inner loop splits the train dataset (N - 1 partitions) further into a smaller train (K - 1 partitions) and validation (1 partition) dataset for hyperparameter tuning
        for train_idx, val_idx in skf.split(train_outer, train_outer[label_col]):
            train_inner = train_outer.iloc[train_idx]
            val_inner = train_outer.iloc[val_idx]

            print(f'Inner train shape: {train_inner.shape}')
            print(f'val shape: {val_inner.shape}')

            model = train_lgb_model(
                train = train_inner, num_boost_rounds = 100, 
                features = features, cat_features = cat_features,
                val = val_inner, callbacks = [lgb.early_stopping(stopping_rounds = 5)]
            )

            print(f'Best iteration: {model.best_iteration}')
            val_preds = model.predict(val_inner[features], num_iteration = model.best_iteration) > threshold
            val_acc = accuracy_score(val_inner[label_col], val_preds)
            print(f'Validation accuracy: {val_acc}')
            
            val_acc_list.append(val_acc)
            inner_num_boost_rounds_list.append(model.best_iteration)
            val_num_boost_rounds_list.append(model.best_iteration)

        # Train on full N - 1 partitions dataset and test on 1 partition held-out test set
        # Extrapolate the num_boost_rounds based on the ratio K / (K - 1)
        avg_val_num_boost_rounds = np.mean(inner_num_boost_rounds_list)
        print(f'Average best iteration: {avg_val_num_boost_rounds}')

        scaled_num_boost_rounds = int(avg_val_num_boost_rounds * scaling_factor)
        print(f'Scaled average best iteration: {scaled_num_boost_rounds}')

        model = train_lgb_model(
            train = train_outer, num_boost_rounds = scaled_num_boost_rounds,
            features = features, cat_features = cat_features
        )
        test_preds = model.predict(test_outer[features], num_iteration = model.best_iteration) > threshold
        test_acc = accuracy_score(test_outer[label_col], test_preds)
        print(f'Test accuracy: {test_acc}')
        test_acc_list.append(test_acc)
        test_num_boost_rounds_list.append(scaled_num_boost_rounds)

    return {
        'val_acc': val_acc_list,
        'test_acc': test_acc_list,
        'val_num_boost_rounds': val_num_boost_rounds_list,
        'test_num_boost_rounds': test_num_boost_rounds_list
    }