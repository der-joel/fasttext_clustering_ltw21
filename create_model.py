"""
Executing this file will create a gensim model from the preprocessed corpus.
The preprocessed corpus needs to be available.
Currently the preprocessed corpus needs to fit into RAM.
"""
from gensim.models.fasttext import FastText, load_facebook_model
from gensim.test.utils import common_corpus  # example
from os.path import isfile, join
from config import PREPROCESSING_RESULT_DIR, GENSIM_SAVE_DIR, LOGLEVEL, FASTTEXT_BASE_MODEL, CORPUS_DIR
from logging import basicConfig
from utils import list_files_in_dir

# setup logging
basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=LOGLEVEL)

# load tokens of the corpus from preprocessed file
print(f"loading corpus from {PREPROCESSING_RESULT_DIR} ...")
corpus = []
for path in list_files_in_dir(PREPROCESSING_RESULT_DIR):
    document = []
    print(f"loading tokens from {path} ...")
    with open(path, "r") as file:
        for token in file:
            document.append(token.rstrip("\n"))
    corpus.append(document)

# create a fasttext model from the preprocessed corpus
if FASTTEXT_BASE_MODEL:
    # load pretrained fasttext model
    print(f"loading fasttext model from {FASTTEXT_BASE_MODEL} ...")
    model = load_facebook_model(FASTTEXT_BASE_MODEL)
    # extend vocabulary of the model with the words from the corpus
    print(f"extending vocabulary of model ...")
    model.build_vocab(corpus, update=True)
else:
    print(f"creating fasttext model from preprocessed corpus in {PREPROCESSING_RESULT_DIR} ...")
    # TODO: train model with words from *.txt files (documents)
    exit(1)


# save the created model to the disk
filename = join(GENSIM_SAVE_DIR, "model.kv")
print(f"saving model to {filename} ...")
model.save(filename)

# save the word vectors of the created model to the disk
filename = join(GENSIM_SAVE_DIR, "vectors.kv")
print(f"saving word vectors to {filename} ...")
model.wv.save(filename)
