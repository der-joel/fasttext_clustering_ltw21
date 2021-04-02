from logging import basicConfig
from gensim.models.fasttext import FastTextKeyedVectors
from nltk.cluster import KMeansClusterer
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from os.path import join, basename, splitext
from config import LOGLEVEL, GENSIM_SAVE_DIR, CLUSTERING_MEANS, CLUSTERING_REPEATS, CLUSTERING_DISTANCE_FUNC, \
    PREPROCESSING_RESULT_DIR, CLUSTER_NAME_TOPN, PICKLE_RESULT_STORAGE_DIR
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

# initialize clusterer
k_means = KMeansClusterer(CLUSTERING_MEANS, CLUSTERING_DISTANCE_FUNC, CLUSTERING_REPEATS, avoid_empty_clusters=True)

# cluster vectors with k-means for each document
results = {}
for path, document in corpus.items():
    vectors = []
    # get vectors for each token (remove it from the document if no vector representation was found)
    for index, token in enumerate(document):
        # remove empty tokens
        if not token or token.isspace():
            del document[index]
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
        "vector": vectors
    })
    # calculate cluster centroids
    cluster_centroids = k_means.means()
    dataframe["cluster_centroid"] = dataframe["cluster"].apply(lambda c: cluster_centroids[c])
    # calculate cluster names from vocabulary (most similar word vector to cluster centroid)
    print(f"calculating cluster names (from vocabulary) ...")
    cluster_names = []
    for centroid in cluster_centroids:
        most_similar = keyed_vectors.most_similar(positive=[centroid], topn=CLUSTER_NAME_TOPN)
        cluster_names.append(tuple([kv[0] for kv in most_similar]))
    dataframe["cluster_name_vocabulary"] = dataframe["cluster"].apply(lambda c: cluster_names[c])
    # calculate cluster names from document only (most similar word vector to cluster centroid)
    cluster_names = []
    print(f"calculating cluster names (from document) ...")
    for centroid in cluster_centroids:
        # this could also be achieved calling keyed_vectors.most_similar_to_given(centroid, vectors)
        # but this approach is way too slow due to unnecessary iterations
        # TODO: can be optimised by only calculating distance to cluster members
        distances = cdist([centroid], vectors, "cosine")[0]
        min_indexes = distances.argsort()[:CLUSTER_NAME_TOPN]
        most_similar = [document[i] for i in min_indexes]
        cluster_names.append(tuple(most_similar))
    dataframe["cluster_name_document"] = dataframe["cluster"].apply(lambda c: cluster_names[c])
    # save result to disk
    print(f"clustering completed for {path} ...")
    storage_path = join(PICKLE_RESULT_STORAGE_DIR, basename(splitext(path)[0]) + ".pickle")
    dataframe.to_pickle(storage_path)
    print(f"saved result to {storage_path}")
