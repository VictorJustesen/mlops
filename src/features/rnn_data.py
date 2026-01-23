import os

import pandas as pd

RAW_DATA_PATH = "data/raw"
RNN_DATA_PATH = "data/rnn"


def time_series_split_save(input_dir=RAW_DATA_PATH, output_dir=RNN_DATA_PATH, split_ratio=0.8):
    """
    For each CSV in input_dir, map columns to [Date, Price, Load, Production],
    split by time, and save to output_dir.
    """
    # Mapping for each region/file: {region: {date: ..., price: ..., load: ..., production: ...}}
    # If a file is not listed, try to infer columns by keywords.
    column_map = {
        "BE": {
            "date": "Date",
            "price": "Prices",
            "load": "System load forecast",
            "production": "Generation forecast",
        },
        "FR": {
            "date": "Date",
            "price": "Prices",
            "load": "System load forecast",
            "production": "Generation forecast",
        },
        "DE": {
            "date": None,
            "price": "Price",
            "load": "Ampirion Load Forecast",
            "production": "PV+Wind Forecast",
        },
        "NP": {
            "date": "Date",
            "price": "Price",
            "load": "Grid load forecast",
            "production": "Wind power forecast",
        },
        "PJM": {
            "date": "Date",
            "price": "Zonal COMED price",
            "load": "System load forecast",
            "production": "Zonal COMED load foecast",
        },
    }

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            region = filename.split(".")[0]
            file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(file_path, skipinitialspace=True)
            # Get mapping for this region, or try to infer
            mapping = column_map.get(region, None)
            if mapping is None:
                # Try to infer columns by keywords
                mapping = {}
                for col in df.columns:
                    cl = col.lower()
                    if "date" in cl:
                        mapping["date"] = col
                    elif "price" in cl:
                        mapping["price"] = col
                    elif "load" in cl and "forecast" in cl:
                        if "system" in cl or "grid" in cl or "ampirion" in cl:
                            mapping["load"] = col
                        else:
                            mapping["production"] = col
                    elif "generation" in cl or "production" in cl or "wind" in cl or "pv" in cl:
                        mapping["production"] = col
                # Fallbacks
                mapping.setdefault("date", df.columns[0])
                mapping.setdefault("price", df.columns[1])
                mapping.setdefault("load", df.columns[2])
                mapping.setdefault(
                    "production", df.columns[3] if len(df.columns) > 3 else df.columns[2]
                )

            # Handle DE: date is index
            if region == "DE" and mapping["date"] is None:
                df = df.rename(columns={df.columns[0]: "Date"})
                mapping["date"] = "Date"

            # Standardize columns
            df_out = pd.DataFrame()
            df_out["Date"] = pd.to_datetime(df[mapping["date"]])
            df_out["Price"] = df[mapping["price"]]
            df_out["Load"] = df[mapping["load"]]
            df_out["Production"] = df[mapping["production"]]
            df_out = df_out.sort_values("Date")

            split_idx = int(len(df_out) * split_ratio)
            train_df = df_out.iloc[:split_idx]
            test_df = df_out.iloc[split_idx:]
            train_out = os.path.join(output_dir, f"{region}_train.csv")
            test_out = os.path.join(output_dir, f"{region}_test.csv")
            train_df.to_csv(train_out, index=False)
            test_df.to_csv(test_out, index=False)
            print(
                f"[TSS] Saved {region} train: {train_out} ({len(train_df)}) | "
                f"test: {test_out} ({len(test_df)})"
            )


GROUPED_DATA_PATH = "data/grouped"


def grouped_time_split_save(input_dir=RAW_DATA_PATH, output_dir=GROUPED_DATA_PATH, split_ratio=0.8):
    """
    For each CSV in input_dir, map columns to [Date, Price, Load, Production],
    split by whole weeks (grouped by year and week), and save to output_dir.
    Every 5th week is assigned to test, others to train. Ensures no temporal leakage
    and is safe for RNNs.
    """
    column_map = {
        "BE": {
            "date": "Date",
            "price": "Prices",
            "load": "System load forecast",
            "production": "Generation forecast",
        },
        "FR": {
            "date": "Date",
            "price": "Prices",
            "load": "System load forecast",
            "production": "Generation forecast",
        },
        "DE": {
            "date": None,
            "price": "Price",
            "load": "Ampirion Load Forecast",
            "production": "PV+Wind Forecast",
        },
        "NP": {
            "date": "Date",
            "price": "Price",
            "load": "Grid load forecast",
            "production": "Wind power forecast",
        },
        "PJM": {
            "date": "Date",
            "price": "Zonal COMED price",
            "load": "System load forecast",
            "production": "Zonal COMED load foecast",
        },
    }

    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith(".csv"):
            region = filename.split(".")[0]
            file_path = os.path.join(input_dir, filename)
            df = pd.read_csv(file_path, skipinitialspace=True)
            mapping = column_map.get(region, None)
            if mapping is None:
                mapping = {}
                for col in df.columns:
                    cl = col.lower()
                    if "date" in cl:
                        mapping["date"] = col
                    elif "price" in cl:
                        mapping["price"] = col
                    elif "load" in cl and "forecast" in cl:
                        if "system" in cl or "grid" in cl or "ampirion" in cl:
                            mapping["load"] = col
                        else:
                            mapping["production"] = col
                    elif "generation" in cl or "production" in cl or "wind" in cl or "pv" in cl:
                        mapping["production"] = col
                mapping.setdefault("date", df.columns[0])
                mapping.setdefault("price", df.columns[1])
                mapping.setdefault("load", df.columns[2])
                mapping.setdefault(
                    "production", df.columns[3] if len(df.columns) > 3 else df.columns[2]
                )

            if region == "DE" and mapping["date"] is None:
                df = df.rename(columns={df.columns[0]: "Date"})
                mapping["date"] = "Date"

            df_out = pd.DataFrame()
            df_out["Date"] = pd.to_datetime(df[mapping["date"]])
            df_out["Price"] = df[mapping["price"]]
            df_out["Load"] = df[mapping["load"]]
            df_out["Production"] = df[mapping["production"]]
            df_out = df_out.sort_values("Date")

            # Add year, week number columns
            df_out["Year"] = df_out["Date"].dt.isocalendar().year
            df_out["Week"] = df_out["Date"].dt.isocalendar().week

            # Assign every 5th week to test, others to train
            week_keys = (
                df_out[["Year", "Week"]]
                .drop_duplicates()
                .sort_values(["Year", "Week"])
                .values.tolist()
            )
            train_idx, test_idx = [], []
            for i, (year, week) in enumerate(week_keys):
                group = df_out[(df_out["Year"] == year) & (df_out["Week"] == week)]
                if (i + 1) % 5 == 0:
                    test_idx.extend(group.index)
                else:
                    train_idx.extend(group.index)

            train_df = df_out.loc[train_idx].drop(columns=["Year", "Week"])
            test_df = df_out.loc[test_idx].drop(columns=["Year", "Week"])
            train_out = os.path.join(output_dir, f"{region}_train.csv")
            test_out = os.path.join(output_dir, f"{region}_test.csv")
            train_df.to_csv(train_out, index=False)
            test_df.to_csv(test_out, index=False)
            print(
                f"[Grouped] Saved {region} train: {train_out} ({len(train_df)}) | "
                f"test: {test_out} ({len(test_df)})"
            )


# Add a main function for CLI entry point
def main():
    """
    Run both time_series_split_save and grouped_time_split_save for CLI usage.
    """
    time_series_split_save()
    grouped_time_split_save()


if __name__ == "__main__":
    main()
