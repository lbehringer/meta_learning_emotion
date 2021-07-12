import json
import numpy as np
import pandas as pd

df = pd.read_json('data/pavoque/sad.json')
df = df.T

# name columns
df.columns = ['features', 'emotion', 'gender']
# print(type(df["features"][1]))  # features are of list type

# convert all features from list to np.array
df["features"] = df["features"].apply(lambda x: np.array(x))

print(df["features"].head(5))
# print(type(df["features"][1]))
print(df["emotion"].head(5))
# print(type(df["label"][1]))

# print(df["features"])
print(df["features"].shape)
print(type(df["features"]))
row = df.iloc[0]
f_0 = np.array(row.features)
print(type(f_0))
print(f_0.shape)
X = df.features.values
# print(X)
print(X.shape)
# print([np.min(a) for a in X])