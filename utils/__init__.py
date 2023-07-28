import pandas as pd
import plotly.express as px


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