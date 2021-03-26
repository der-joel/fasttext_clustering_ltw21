from os import listdir
from os.path import isfile, join, splitext
from gensim.utils import tokenize


def list_files_in_dir(path):
    """
    get all files in the given directory
    :param path: a path to the directory
    :return: a list containing paths to every file in the directory
    """
    result = []
    for filename in listdir(path):
        # get full path to file
        filepath = join(path, filename)
        # if its not a directory add it to the list
        if isfile(filepath):
            result.append(filepath)
    return result
