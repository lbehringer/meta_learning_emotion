import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
import json

import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import seaborn as sb

# dict_test = dict()
# dict_test['sad'] = dict()
# dict_test['sad']['embeddings'] = [['12345']]
# dict_test['sad']['embeddings'] += [['23456']]
# print(dict_test)

df = pd.read_json('emotion_embeddings_2.json')
df = df.T
# convert all features from list to np.array
#df["features"] = df["features"].apply(lambda x: np.array(x))
print(df['features'][0][0])
print(type(df['features']))
#features = df['features'].tolist()
#print(features)
X = df['features']  # contains embeddings for all emotions
X_embs = X[0]  # contains embeddings for first emotion

# WE NEED X_embeddings FOR EACH ROW (= for each emotion label)

y = df['label']

#print('X.shape: ' + str(X.shape))
#print(X[1].shape, y.shape)
#print(X[1][0].shape)

# create df with columns (count len of X_embeddings[0])
num_embs = len(X_embs)  # get number of embeddings in X_embs
print('num_embs = ' + str(num_embs))
num_cols = len(X_embs[0])  # get number of embedding features
print("num_cols = " + str(num_cols))
df_features = pd.DataFrame(X_embs[0])  # create df with first embedding
df_features = df_features.T  # transpose to row vector
# df_features = pd.DataFrame(np.zeros((1,150)), dtype='float32') # zero-initialized df not needed, index0 used instead

# append all embeddings to df_features, one row at a time
for i in range(num_embs-1):  # iterate over all embeddings (num_embs-1 since we already created the df with index 0)
	emb = pd.DataFrame(X_embs[i+1])  # i+1 since we already created the df with index 0
	emb = emb.T
	df_features = df_features.append(emb)  # append embedding to df_features

print(df_features.head())  # this should return a df with <num_embs> rows and <num_cols> columns
print(df_features.shape)

#print(df_features.shape)
#print(df_features[1])

# for each embedding (for element in X[0], pd.append(new_embedding) to df

#print(X[0][0])

test = [element for element in X[1]]
#print(test[0])


#feat_cols = ['value'+str(i) for i in range(X.shape[1])]

#df2 = pd.DataFrame(X, columns=feat_cols)
#df2['y'] = y
#df2['label'] = df2['y'].apply(lambda i: str(i))

X, y = None, None

#print('Size of the dataframe: {}'.format(df2.shape))


# For reproducability of the results
#np.random.seed(42)

#rndperm = np.random.permutation(df.shape[0])

# PCA

# pca = PCA(n_components=2)
# pca_result = pca.fit_transform(df['features'].values)
#
# df['pca-one'] = pca_result[:, 0]
# df['pca-two'] = pca_result[:, 1]
# df['pca-three'] = pca_result[:, 2]
#
# print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#
#
#
#
#
# # TSNE
# tsne = TSNE()