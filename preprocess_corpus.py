"""
Executing this file will preprocess all documents stored in the corpus directory.
The results will be stored in the preprocessing result directory.
Currently the textual representation of each documents must fit into RAM.
"""
import textract
from utils import list_files_in_dir
from os import listdir
from os.path import isfile, join, splitext, basename
from gensim.utils import to_unicode
from config import PREPROCESSING_RESULT_DIR, PREPROCESSING_TOKENIZER, CORPUS_DIR, CORPUS_ENCODING, \
    PREPROCESSING_TOKENIZER_KWARGS, GENSIM_SAVE_DIR, PREPROCESSING_TOKENIZER_ARGS, TOKEN_IGNORE_LIST


# preprocess every file in the corpus directory and save it to the result directory
print(f"converting files in {CORPUS_DIR} to pdf ...")
for path in list_files_in_dir(CORPUS_DIR):
    # get file content as byte string with textract
    raw_text = textract.process(path, encoding=CORPUS_ENCODING)
    # decode byte string
    raw_text = to_unicode(raw_text, encoding=CORPUS_ENCODING)
    # tokenize document text
    tokens = PREPROCESSING_TOKENIZER(raw_text, *PREPROCESSING_TOKENIZER_ARGS, **PREPROCESSING_TOKENIZER_KWARGS)
    # remove stopwords defined in the token ignore list
    with open(TOKEN_IGNORE_LIST, "r") as f:
        for stopword in f:
            try:
                tokens.remove(stopword)
            except ValueError:
                # the token is not in the corpus
                pass
    # save as *.txt file
    filename = splitext(basename(path))[0]
    write_path = join(PREPROCESSING_RESULT_DIR, filename + ".txt")
    with open(write_path, "w") as f:
        f.write("\n".join(tokens))
    # print success
    print(f"{filename}(\"{path}\") was processed successfully and saved as \"{write_path}\"")
