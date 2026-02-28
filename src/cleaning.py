import pandas as pd
import numpy as np

def load_data(path):
    return pd.read_csv(path)

# handle missing values
def handle_missing_values(df):
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        df[col].fillna(df[col].median(), inplace=True)

    cat_cols = df.select_dtypes(include='object').columns
    for col in cat_cols:
        df[col].fillna(df[col].mode()[0], inplace=True)

    return df

# remove duplicates
def remove_duplicates(df):
    return df.drop_duplicates()

# remove outliers using IQR
def remove_outliers_iqr(df):
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5*IQR) & (df[col] <= Q3 + 1.5*IQR)]
    return df

# drop unnecessary column
def drop_irrelevant(df):
    if 'PassengerId' in df.columns:
        df = df.drop(columns=['PassengerId'])
    return df