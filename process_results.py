import pandas as pd
from config import HTML_RESULT_STORAGE_DIR, CSV_RESULT_STORAGE_DIR
from utils import list_files_in_dir
from os.path import join, basename, splitext


for path in list_files_in_dir(CSV_RESULT_STORAGE_DIR):
    print(f"processing {path} ...")
    # read result csv from disk
    df = pd.read_csv(path)
    # reshape dataframe (aggregation)
    df = df.groupby(df["cluster_name_corpus"]).aggregate({"cluster_name_vocabulary": "first",
                                                          "word": lambda x: ", ".join(x)})
    # save as html
    storage_path = join(HTML_RESULT_STORAGE_DIR, basename(splitext(path)[0]) + ".html")
    with open(storage_path, "w") as f:
        # prepend meta attribute so that german umlauts are displayed properly
        f.write(r"<meta charset='utf-8'>")
        f.write(df.to_html())
    print(f"saved html to {storage_path}")
