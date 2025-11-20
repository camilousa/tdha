import joblib
import numpy as np
import pandas as pd

def calculate_rois_means(ts_list):
    a = []
    for row in ts_list:
        rois_means = np.array(row).mean(axis=1)

        a.append(rois_means)

    df = pd.DataFrame(a)
    df.columns = rois_names['name'].to_list()
    return df


dir_list = [
    {
    "dir_name": "NYU",
    "single_phenotype_file": True,
    },
    {
    "dir_name": "NeuroIMAGE",
    "single_phenotype_file": True,
    },
    {
    "dir_name": "KKI",
    "single_phenotype_file": True,
    },
    {
    "dir_name": "OHSU",
    "single_phenotype_file": True,
    },
    {
    "dir_name": "Peking",
    "single_phenotype_file": False,
    },
    ]

n_rois = 18


#output_dir = f"data/raw-bold-data/pca-rois-mean-matrix/{n_rois}-selected-rois-by-site"
output_dir = f"data/raw-bold-data/pca-rois-mean-matrix"

for dir in dir_list:
    dir_name = dir["dir_name"]
    file_path = f"data/raw-bold-data/{n_rois}-rois-dataset/{dir_name}.pkl"
    file_path_rois = f"data/raw-bold-data/{n_rois}-rois-dataset/{dir_name}_rois_names.csv"
    print("reading...", file_path)
    data = joblib.load(file_path)
    rois_names = pd.read_csv(file_path_rois)
    print(rois_names['name'].to_list())

    ids, ts_list, labels = data["id"], data["data"], data["labels"]
    
    print("calculating means...")
    df = calculate_rois_means(ts_list)
    df["label"] = data["labels"]
    output_file = f"{output_dir}/{dir_name}.csv"
    print("saving to ...", output_file)
    df.to_csv(output_file, index=False)