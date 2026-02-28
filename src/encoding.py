import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from category_encoders import TargetEncoder

def one_hot(df, cols):
    return pd.get_dummies(df, columns=cols, drop_first=True)

def label(df, cols):
    le = LabelEncoder()
    for c in cols:
        df[c] = le.fit_transform(df[c])
    return df

def ordinal(df, cols):
    oe = OrdinalEncoder()
    df[cols] = oe.fit_transform(df[cols])
    return df

def frequency(df, cols):
    for c in cols:
        freq = df[c].value_counts()/len(df)
        df[c] = df[c].map(freq)
    return df

def target(df, cols, target):
    te = TargetEncoder(cols=cols)
    df[cols] = te.fit_transform(df[cols], df[target])
    return df