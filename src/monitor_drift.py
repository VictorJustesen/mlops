import os
import pandas as pd
from evidently.legacy.report import Report
from evidently.legacy.metric_preset import DataDriftPreset
import sys
from src.models.train_rnn import SequenceDataset
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))
from datetime import datetime
import torch
import typer

app = typer.Typer()

def check_data_drift(trained_data, new_data, model_path, data_path="data/grouped", output_dir="data/drift"):
    """
    Check data drift between reference and current datasets for specified regions.
    utilizing the model hyperparameters for data sequencing.
    IMPORTANT: This function assumes that the training data used for the model
    is available in the data_path directory, and has been split accordingly.
    Args:
        trained_data (list): List of region names for the reference dataset.
        new_data (list): List of region names for the current dataset.
        model_path (str): Path to the trained model checkpoint.
        data_path (str): Path to the directory containing region CSV files.
        output_dir (str): Directory to save the generated reports.
    """
    print("Creating output directory if it doesn't exist...")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare file lists for SequenceDataset (train data only)
    print(f"Preparing file lists for trained regions: {trained_data}")
    trained_train_files = [os.path.join(data_path, f"{region}_train.csv") for region in trained_data]
    print(f"Preparing file lists for new regions: {new_data}")
    new_train_files = [os.path.join(data_path, f"{region}_train.csv") for region in new_data]

    # Check all files exist
    print("Checking for missing files...")
    missing_trained = [f for f in trained_train_files if not os.path.exists(f)]
    missing_new = [f for f in new_train_files if not os.path.exists(f)]
    if missing_trained:
        print(f"Missing train files for trained regions: {missing_trained}")
    if missing_new:
        print(f"Missing train files for new regions: {missing_new}")
    if missing_trained or missing_new:
        print("Aborting drift check due to missing files.")
        return

    # Load window_size and input_features from model checkpoint
    print(f"Loading model checkpoint from: {model_path}")
    # Use only raw features and target (no windowing)
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found: {model_path}")
        return
    ckpt = torch.load(model_path, map_location="cpu")
    hparams = ckpt.get('hyper_parameters', ckpt.get('hparams', {}))
    input_features = hparams.get('input_features', ["Price", "Load", "Production"])
    print(f"Loaded input_features={input_features} from model checkpoint.")

    def load_raw_df(file_list, input_features):
        dfs = []
        for f in file_list:
            if os.path.exists(f):
                df = pd.read_csv(f)
                # Only keep the input features and target (assume target is 'Price')
                keep_cols = [col for col in input_features if col in df.columns]
                if 'Price' not in keep_cols:
                    keep_cols.append('Price')
                dfs.append(df[keep_cols])
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame(columns=input_features + ['Price'])

    trained_df = load_raw_df(trained_train_files, input_features)
    new_df = load_raw_df(new_train_files, input_features)

    print("Running Evidently data drift report...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=trained_df, current_data=new_df)

    timestamp = datetime.now().strftime("%H%M")
    report_path = os.path.join(output_dir, f"Data_drift_report_{timestamp}.html")
    print(f"Saving drift report to {report_path} ...")
    report.save_html(report_path)
    print(f"Data drift report for all regions saved to {report_path}")

@app.command()
def cli(
    trained_regions: str = typer.Option(..., help="Comma-separated list of already trained regions"),
    new_regions: str = typer.Option(..., help="Comma-separated list of new regions to check"),
    model_path: str = typer.Option(..., help="Path to model checkpoint (.ckpt)"),
    data_path: str = typer.Option("data/grouped", help="Path to region CSV files"),
    output_dir: str = typer.Option("data/drift", help="Directory to save drift report")
):
    trained_list = [r.strip() for r in trained_regions.split(",")]
    new_list = [r.strip() for r in new_regions.split(",")]
    check_data_drift(
        trained_data=trained_list,
        new_data=new_list,
        model_path=model_path,
        data_path=data_path,
        output_dir=output_dir
    )

def pairwise_drift_inspector(target_region, data_path="data/grouped", output_dir="data/drift", window_size=168, input_features=None):
    """
    Compare the target region's train dataset against every other region's train dataset in the data_path folder.
    Generates a drift report for each comparison.
    Args:
        target_region (str): Region name to use as reference.
        data_path (str): Path to the directory containing region CSV files.
        output_dir (str): Directory to save the generated reports.
        window_size (int): Window size for sequencing.
        input_features (list): List of input features to use.
    """
    if input_features is None:
        input_features = ["Price", "Load", "Production"]
    os.makedirs(output_dir, exist_ok=True)
    target_file = os.path.join(data_path, f"{target_region}_train.csv")
    if not os.path.exists(target_file):
        print(f"Target train file not found: {target_file}")
        return
    print(f"Sequencing target region: {target_region}")
    target_seq = SequenceDataset([target_file], window_size=window_size, input_features=input_features)
    target_df = pd.DataFrame([x.flatten().tolist() + [float(y)] for x, y in zip(target_seq.sequences, target_seq.targets)],
                            columns=[f"f{i}" for i in range(window_size * len(input_features))] + ["target"])

    # Find all other train files
    all_files = [f for f in os.listdir(data_path) if f.endswith('_train.csv') and not f.startswith(target_region)]
    for other_file in all_files:
        other_region = other_file.split('_train.csv')[0]
        other_path = os.path.join(data_path, other_file)
        print(f"Sequencing comparison region: {other_region}")
        other_seq = SequenceDataset([other_path], window_size=window_size, input_features=input_features)
        other_df = pd.DataFrame([x.flatten().tolist() + [float(y)] for x, y in zip(other_seq.sequences, other_seq.targets)],
                                columns=[f"f{i}" for i in range(window_size * len(input_features))] + ["target"])
        print(f"Running drift report for {target_region} vs {other_region} ...")
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=target_df, current_data=other_df)
        timestamp = datetime.now().strftime("%H%M")
        report_path = os.path.join(output_dir, f"Pairwise_drift_{target_region}_vs_{other_region}_{timestamp}.html")
        report.save_html(report_path)
        print(f"Saved drift report: {report_path}")
@app.command()
def pairwise_cli(
    target_region: str = typer.Option(..., help="Region to use as reference for pairwise drift check"),
    data_path: str = typer.Option("data/grouped", help="Path to region CSV files"),
    output_dir: str = typer.Option("data/drift", help="Directory to save drift reports"),
    window_size: int = typer.Option(168, help="Window size for sequencing"),
    input_features: str = typer.Option("Price,Load,Production", help="Comma-separated input features")
):
    features_list = [f.strip() for f in input_features.split(",")]
    pairwise_drift_inspector(
        target_region=target_region,
        data_path=data_path,
        output_dir=output_dir,
        window_size=window_size,
        input_features=features_list
    )

if __name__ == "__main__":
    app()