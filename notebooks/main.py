import numpy as np
from src.cleaning import *
from src.encoding import *
from src.scaling import *

# load dataset
df = load_data("dataset/train.csv")

# ---------------- CLEANING ----------------
df = handle_missing_values(df)
df = remove_duplicates(df)
df = remove_outliers_iqr(df)
df = drop_irrelevant(df)

# ---------------- SKEWNESS ----------------
num_cols = df.select_dtypes(include=np.number).columns
for c in num_cols:
    if abs(df[c].skew()) > 1:
        df[c] = np.log1p(df[c])

# ---------------- ENCODING ----------------
cat_cols = df.select_dtypes(include='object').columns.tolist()

df1 = one_hot(df.copy(), cat_cols[:2])
df2 = label(df.copy(), cat_cols[2:4])
df3 = ordinal(df.copy(), cat_cols[4:6])
df4 = frequency(df.copy(), cat_cols[6:8])
df5 = target(df.copy(), cat_cols[0:2], 'Survived')

# ---------------- SCALING ----------------
num = df.select_dtypes(include=np.number)
minmax(num)
maxabs(num)
standard(num)
normalize(num)

print("Preprocessing Done Successfully")