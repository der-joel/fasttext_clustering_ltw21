"""
the configuration file of this project
"""
from logging import INFO
from gensim.utils import simple_preprocess
from nltk.cluster import cosine_distance, euclidean_distance

# app settings
LOGLEVEL = INFO
PICKLE_RESULT_STORAGE_DIR = "results/pickle"
HTML_RESULT_STORAGE_DIR = "results/html"
PNG_RESULT_STORAGE_DIR = "results/png"
# corpus settings
CORPUS_DIR = "corpus/pdf"
CORPUS_ENCODING = "utf-8"
CORPUS_REMOVE_DUPLICATES = True
TOKEN_IGNORE_LIST = "corpus/stopwords/german_stopwords_full.txt"
# preprocessing settings
PREPROCESSING_RESULT_DIR = "corpus/txt"
PREPROCESSING_TOKENIZER = simple_preprocess
PREPROCESSING_TOKENIZER_ARGS = []
PREPROCESSING_TOKENIZER_KWARGS = {
    "min_len": 5
}
# gensim settings
GENSIM_SAVE_DIR = "gensim"
FASTTEXT_BASE_MODEL = "embeddings/fasttext/deepset_model.bin"
# k-means settings
CLUSTERING_MEANS = 5
CLUSTERING_REPEATS = 1
CLUSTERING_DISTANCE_FUNC = cosine_distance
CLUSTER_NAME_TOPN = 10
