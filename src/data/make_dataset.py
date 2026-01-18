import io
import os

import pandas as pd
import requests


def download_data(dataset_name, output_path):
    print("starting")
    url = f"https://zenodo.org/record/4624805/files/{dataset_name}.csv?download=1"
    content = requests.get(url).content
    df = pd.read_csv(io.StringIO(content.decode("utf-8")), index_col=0, parse_dates=True)

    os.makedirs(output_path, exist_ok=True)
    file_path = os.path.join(output_path, f"{dataset_name}.csv")
    df.to_csv(file_path)
    print(f"Saved {dataset_name} to {file_path}")


def main():
    datasets = ["NP", "PJM", "BE", "FR", "DE"]
    raw_data_path = "data/raw"

    for dataset in datasets:
        print(f"Downloading {dataset}...")
        download_data(dataset, raw_data_path)


if __name__ == "__main__":
    main()
