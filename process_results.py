import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from config import HTML_RESULT_STORAGE_DIR, PICKLE_RESULT_STORAGE_DIR, PNG_RESULT_STORAGE_DIR
from utils import list_files_in_dir
from os.path import join, basename, splitext
from sklearn.manifold import TSNE
from matplotlib.font_manager import FontProperties


for path in list_files_in_dir(PICKLE_RESULT_STORAGE_DIR):
    print(f"reading dataframe from disk: {path} ...")
    # read result csv from disk
    df = pd.read_pickle(path)

    print(f"creating scatter plot ...")
    # make plot font smaller
    fontP = FontProperties()
    fontP.set_size('x-small')
    # create plot
    fig, ax = plt.subplots(figsize=(20, 20))
    unique_clusters = df["cluster"].unique()
    for index, cluster_id in enumerate(unique_clusters):
        data = df[df["cluster"] == cluster_id]
        # create label for this cluster
        name_vocab = data["cluster_name_vocabulary"].iloc[0]
        name_doc = data["cluster_name_document"].iloc[0]
        label = f"Vocabulary: {', '.join(name_vocab)} \nDocument: {', '.join(name_doc)}"
        # get all word embeddings for this cluster
        vectors = list(data["vector"])
        # reduce dimensions of word embeddings using t-SNE
        plot_data = TSNE(n_components=2).fit_transform(vectors).transpose()
        # add cluster to scatter plot
        ax.scatter(plot_data[0],
                   plot_data[1],
                   label=label)
    # shrink axis height by 10% at the bottom
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    # add legend at the bottom
    ax.legend(title="Cluster Names",
              loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=len(unique_clusters))
    # add axis labels
    plt.ylabel('y')
    plt.xlabel('x')
    # save plot as png
    storage_path = join(PNG_RESULT_STORAGE_DIR, basename(splitext(path)[0]) + ".png")
    plt.savefig(storage_path)
    print(f"saved scatter plot to {storage_path}")

    # reshape dataframe (aggregation)
    df = df.groupby(df["cluster"]).aggregate({"cluster_name_vocabulary": "first",
                                              "cluster_name_document": "first",
                                              "word": lambda x: ", ".join(x)})

    print(f"creating html-table for {path} ...")
    # count word sum per cluster
    df["word_count"] = df["word"].apply(lambda x: len(x.split(", ")))
    # sort by word count
    df.sort_values("word_count", ascending=False)
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
    print(f"saved html-table to {storage_path}")
