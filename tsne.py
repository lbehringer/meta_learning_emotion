import numpy as np
import pandas as pd
pd.set_option('display.max_rows', None)
import json

import sklearn
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#%matplotlib inline  # uncomment this when using jupyter notebook
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

import seaborn as sns

def tsne(in_file, out_file, embedding_size, perplexity, iter):
	#with pd.option_context('display.float_format', '{:0.20f}'.format):
	df = pd.read_json(in_file)
	df = df.T
	print(df.columns.values)
	# convert all features from list to np.array
	#df["features"] = df["features"].apply(lambda x: np.array(x))
	#print(df.head())
	#print(df['embeddings'][0])
	#print(type(df['features']))
	#features = df['features'].tolist()
	#print(features)
	X = df['embeddings']  # contains embeddings for all emotions


	#X_embs = X[0]  # contains embeddings for first emotion

	# WE NEED X_embeddings FOR EACH ROW (= for each emotion label)

	y = df['label']  # contains labels for all emotions
	print(y)

	#print('X.shape: ' + str(X.shape))
	#print("y shape = " + str(y.shape))
	#print(y[0])
	#print(X[1][0].shape)

	X_embs = X[0]
	print(X_embs[0])
	num_labels = len(X)
	num_embs = len(X_embs)  # get number of embeddings in X_embs
	# print('num_embs = ' + str(num_embs))
	num_cols = len(X_embs[0])  # get number of embedding features
	# print("num_cols = " + str(num_cols))
	col_dict = dict()
	for k in range(num_cols):
		col_dict[k] = 'feat' + str(k)
	# print("col_dict = " + str(col_dict))

	df_features = pd.DataFrame(X[0][0], index=col_dict)  # create df with first embedding
	df_features = df_features.append(pd.Series(y[0]), ignore_index=True)  # append label
	df_features = df_features.T  # transpose to row vector


	# iterate over all emotions

	for i in range(num_labels):
		X_embs = X[i]

		# create df with columns (count len of X_embeddings[0])
		num_embs = len(X_embs)  # get number of embeddings in X_embs
		# print('num_embs = ' + str(num_embs))
		num_cols = len(X_embs[i])  # get number of embedding features
		# print("num_cols = " + str(num_cols))
		col_dict = dict()
		for k in range(num_cols):
			col_dict[k] = 'feat' + str(k)
		# print("col_dict = " + str(col_dict))
		if i == 0:
			# append all embeddings to df_features, one row at a time
			for j in range(num_embs - 1):  # iterate over all embeddings (num_embs-1 since we already created the df with index 0)
				emb = pd.DataFrame(X_embs[j + 1])  # j+1 since we already created the df with index 0
				emb = emb.append(pd.Series(y[i]), ignore_index=True)  # append label
				emb = emb.T
				df_features = df_features.append(emb)  # append embedding to df_features
		else:
			for j in range(num_embs):
				emb = pd.DataFrame(X_embs[j])
				emb = emb.append(pd.Series(y[i]), ignore_index=True)
				emb = emb.T
				df_features = df_features.append(emb)


	df_features.rename(columns=col_dict, inplace=True)  # rename columns
	df_features.rename(columns={embedding_size: "label"}, inplace=True)
	feat_cols = ['feat'+str(i) for i in range(num_cols)]

	#print(df_features.head())
	#print(df_features.shape[0])
	#print(df_features.shape)  # this should return a df with <num_embs> rows and <num_cols> columns


	X, y = None, None

	# For reproducability of the results
	# np.random.seed(42)
	#
	# rndperm = np.random.permutation(df_features.shape[0])

	N = embedding_size-1
	# create subset
	# df_subset = df_features.loc[rndperm[:N],:].copy()
	data_subset = pd.DataFrame(data=df_features[feat_cols].values)
	data_subset.rename(columns=col_dict, inplace=True)
	#print(data_subset.shape)
	#print(df_features['label'].shape)
	data_subset['label'] = df_features['label'].values



	# TSNE
	tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=iter)
	tsne_results = tsne.fit_transform(data_subset[feat_cols])  # this should be df_features[feat_cols].values
	#print(tsne_results)
	#print(data_subset.shape)
	tsne_2d_one = tsne_results[:,0]
	tsne_2d_two = tsne_results[:,1]
	data_subset['tsne-2d-one'] = tsne_2d_one
	#print(data_subset['tsne-2d-one'])
	data_subset['tsne-2d-two'] = tsne_2d_two

	plt.figure(figsize=(16,10))
	sns.scatterplot(
		x="tsne-2d-one", y="tsne-2d-two",
		hue="label",
		palette=sns.color_palette("hls", num_labels),
		data=data_subset,
		legend="full",
		alpha=0.3
	)

	plt.savefig(out_file)