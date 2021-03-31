from logging import basicConfig
from gensim.models.fasttext import FastText, FastTextKeyedVectors
from nltk.cluster import KMeansClusterer, cosine_distance, euclidean_distance
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from os.path import join, basename, splitext
from itertools import chain
from config import LOGLEVEL, GENSIM_SAVE_DIR, CLUSTERING_MEANS, CLUSTERING_REPEATS, CLUSTERING_DISTANCE_FUNC, \
    PREPROCESSING_RESULT_DIR, CLUSTER_NAME_TOPN, CSV_RESULT_STORAGE_DIR
from utils import list_files_in_dir

# setup logging
basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=LOGLEVEL)

# load tokens of the corpus from preprocessed file
print(f"loading corpus from {PREPROCESSING_RESULT_DIR} ...")
corpus = {}
for path in list_files_in_dir(PREPROCESSING_RESULT_DIR):
    document = []
    print(f"loading tokens from {path} ...")
    with open(path, "r") as file:
        for token in file:
            document.append(token.rstrip("\n"))
    corpus[path] = document

# load word vectors
print(f"loading word embeddings from {GENSIM_SAVE_DIR} ...")
filename = join(GENSIM_SAVE_DIR, "vectors.kv")
keyed_vectors = FastTextKeyedVectors.load(filename)

# notes
# word in vocabulary: "word" in model.wv
# word -> vector: model.wv.word_vec("word")
# most similar word: model.most_similar("word")

# initialize clusterer
k_means = KMeansClusterer(CLUSTERING_MEANS, CLUSTERING_DISTANCE_FUNC, CLUSTERING_REPEATS, avoid_empty_clusters=True)

# cluster vectors with k-means for each document
results = {}
for path, document in corpus.items():
    vectors = []
    # get vectors for each token (remove it from the document if no vector representation was found)
    for token in document:
        # skip empty tokens
        if not token or token.isspace():
            continue
        # get word vector for token
        try:
            vec = keyed_vectors.get_vector(token)
        except KeyError:
            print(f"could not find {token} in the loaded word embeddings (removing from document) ...")
            document.remove(token)
            continue
        # add vector to the cluster list
        vectors.append(vec)
    # check that list lengths match
    if not len(vectors) == len(document):
        print("an error occurred :(")
        exit(1)
    # cluster word vectors using k-means
    print(f"clustering vectors of document {path} using {CLUSTERING_DISTANCE_FUNC.__name__} as distance measure "
          f"with {CLUSTERING_MEANS} means and {CLUSTERING_REPEATS} repeats ...")
    assigned_clusters = k_means.cluster(vectors, assign_clusters=True)
    # add result to pandas data frame
    dataframe = pd.DataFrame({
        "word": document,
        "cluster": assigned_clusters,
    })
    # calculate cluster centroids
    cluster_centroids = k_means.means()
    dataframe["cluster_centroid"] = dataframe["cluster"].apply(lambda c: cluster_centroids[c])
    # calculate cluster names from corpus (most similar word vector to cluster centroid)
    print(f"calculating cluster names (from corpus) ...")
    cluster_names = []
    for centroid in cluster_centroids:
        most_similar = keyed_vectors.most_similar(positive=[centroid], topn=CLUSTER_NAME_TOPN)
        cluster_names.append(tuple([kv[0] for kv in most_similar]))
    dataframe["cluster_name_corpus"] = dataframe["cluster"].apply(lambda c: cluster_names[c])
    # TODO: this takes forever use better approach
    # calculate cluster names from vocabulary (most similar word vector to cluster centroid)
    """ 
    print(f"calculating cluster names (from vocabulary) ...")
    cluster_names = []
    for centroid in cluster_centroids:
        most_similar = keyed_vectors.most_similar_to_given(centroid, vectors)
        print(most_similar)
        cluster_names.append(tuple([kv[0] for kv in most_similar]))
    dataframe["cluster_name_vocabulary"] = dataframe["cluster"].apply(lambda c: cluster_names[c])
    """
    # save result to disk
    print(f"clustering completed for {path} ...")
    storage_path = join(CSV_RESULT_STORAGE_DIR, basename(splitext(path)[0]) + ".csv")
    dataframe.to_csv(storage_path)
    print(f"saved result to {storage_path}")
