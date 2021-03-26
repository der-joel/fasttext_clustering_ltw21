from logging import basicConfig
from gensim.models.fasttext import FastText, FastTextKeyedVectors
from nltk.cluster import KMeansClusterer, cosine_distance, euclidean_distance
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from os.path import join
from config import LOGLEVEL, GENSIM_SAVE_DIR, CLUSTERING_MEANS, CLUSTERING_REPEATS, CLUSTERING_DISTANCE_FUNC

df = pd.DataFrame({'A': [tuple(["a", "b"]), tuple(["a", "b"]), tuple(["a", "c"])]})
print(df["A"].unique())
print(set(df["A"]))
exit(0)

# old code

# setup logging
basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=LOGLEVEL)

# TODO: remove or load from config
pd.options.display.max_rows = 10000

# load word vectors
print(f"loading word embeddings from {GENSIM_SAVE_DIR} ...")
filename = join(GENSIM_SAVE_DIR, "vectors.kv")
keyed_vectors = FastTextKeyedVectors.load(filename)

# notes
# word in vocabulary: "word" in model.wv
# word -> vector: model.wv.word_vec("word")
# most similar word: model.most_similar("word")

# initialize clusterer
print(f"clustering vectors using {CLUSTERING_DISTANCE_FUNC.__name__} as distance measure with {CLUSTERING_MEANS} means "
      f"and {CLUSTERING_REPEATS} repeats ...")
k_means = KMeansClusterer(CLUSTERING_MEANS, CLUSTERING_DISTANCE_FUNC, CLUSTERING_REPEATS)

# get all word vectors
vectors = list(keyed_vectors.vectors)[0:100000]
vocabulary = list(keyed_vectors.vocab)[0:100000]

# cluster vectors
# TODO: cluster per document
# TODO: only cluster words in the original corpus?
assigned_clusters = k_means.cluster(vectors, assign_clusters=True)

# add result to pandas data frame
dataframe = pd.DataFrame({
    "word": vocabulary,
    "cluster": assigned_clusters,
    "vector": vectors
})

# calculate cluster centroids
cluster_centroids = k_means.means()
dataframe["cluster_centroid"] = dataframe["cluster"].apply(lambda c: cluster_centroids[c])

# calculate centroid names (most similar word vector)
print(f"calculating cluster names ...")
cluster_names = []
for centroid in cluster_centroids:
    most_similar = keyed_vectors.most_similar(positive=[centroid], topn=10)
    cluster_names.append([x[0] for x in most_similar])
dataframe["cluster_name"] = dataframe["cluster"].apply(lambda c: cluster_names[c])

# print result
print("clusters:")
for c in cluster_names:
    print(c)
print("---------")
print(dataframe[["word", "cluster_name"]])
# TODO: visualize results for each text corpus
