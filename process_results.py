import pandas as pd
from config import HTML_RESULT_STORAGE_DIR, CSV_RESULT_STORAGE_DIR
from utils import list_files_in_dir
from os.path import join, basename, splitext


for path in list_files_in_dir(CSV_RESULT_STORAGE_DIR):
    print(f"processing {path} ...")
    # read result csv from disk
    df = pd.read_csv(path)
    # reshape dataframe (aggregation)
    df = df.groupby(df["cluster_name_corpus"]).aggregate({#"cluster_name_vocabulary": "first",
                                                          "word": lambda x: ", ".join(x)})
    # count word sum per cluster
    df["word_count"] = df["word"].apply(lambda x: len(x.split(", ")))
    # save as html
    storage_path = join(HTML_RESULT_STORAGE_DIR, basename(splitext(path)[0]) + ".html")
    with open(storage_path, "w") as f:
        # prepend bootstrap cdn for table style
        f.write(
            """
            <head>
            <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
            </head>
            """
        )
        f.write(df.to_html(classes="table table-hover"))
    print(f"saved html to {storage_path}")
