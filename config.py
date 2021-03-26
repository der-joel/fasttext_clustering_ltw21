"""
the configuration file of this project
"""
from logging import INFO
from gensim.utils import simple_preprocess, tokenize, simple_tokenize
from nltk.cluster import cosine_distance, euclidean_distance

# app settings
LOGLEVEL = INFO
# corpus settings
CORPUS_DIR = "corpus/pdf"
CORPUS_ENCODING = "utf-8"
TOKEN_IGNORE_LIST = "corpus/stopwords-de.txt"
# preprocessing settings
PREPROCESSING_RESULT_DIR = "corpus/txt"
PREPROCESSING_TOKENIZER = simple_preprocess
PREPROCESSING_TOKENIZER_ARGS = []
PREPROCESSING_TOKENIZER_KWARGS = {
    "min_len": 4,
    "max_len": 15
}
# gensim settings
GENSIM_SAVE_DIR = "gensim"
FASTTEXT_BASE_MODEL = "embeddings/fasttext/deepset_model.bin"
# k-means settings
CLUSTERING_MEANS = 10
CLUSTERING_REPEATS = 2
CLUSTERING_DISTANCE_FUNC = cosine_distance
CLUSTER_NAME_TOPN = 4
