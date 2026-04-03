from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from lightgbm_smart import detect_similarity


BASE_DIR = Path(__file__).resolve().parent
TEST_CONTRACTS_DIR = BASE_DIR / "testContracts"
PAIR_COLUMNS = ["fid1", "fid2", "type", "name", "names", "value", "unit", "operater", "memberName", "other"]


def load_labels(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] < 3:
        raise ValueError(f"{csv_path} must contain at least three columns")
    df = df.iloc[:, :3].copy()
    df.columns = ["function_id_1", "function_id_2", "label"]
    if not df.empty and not str(df.iloc[0]["label"]).strip().isdigit():
        df = df.iloc[1:].reset_index(drop=True)
    df["function_id_1"] = df["function_id_1"].astype(str)
    df["function_id_2"] = df["function_id_2"].astype(str)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    return df


def index_source_files(data_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(data_dir.rglob("*")):
        if path.is_file():
            index.setdefault(path.stem, path)
    return index


def clear_generated_files() -> None:
    for path in (
        TEST_CONTRACTS_DIR / "SRs.txt",
        TEST_CONTRACTS_DIR / "Features.txt",
        TEST_CONTRACTS_DIR / "test_pairs.csv",
    ):
        if path.exists():
            path.unlink()


def build_pair_features(file1: Path, file2: Path) -> pd.DataFrame:
    clear_generated_files()
    subprocess.run([sys.executable, "-m", "solidity_parser", "parse", str(file1)], cwd=BASE_DIR, check=True)
    subprocess.run([sys.executable, "-m", "solidity_parser", "parse", str(file2)], cwd=BASE_DIR, check=True)
    subprocess.run([sys.executable, "get_feature.py"], cwd=BASE_DIR, check=True)
    pair_csv = TEST_CONTRACTS_DIR / "test_pairs.csv"
    if not pair_csv.exists():
        return pd.DataFrame(columns=PAIR_COLUMNS)
    df = pd.read_csv(pair_csv, header=None)
    if df.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS)
    df.columns = PAIR_COLUMNS
    return df


def prepare_model_input(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    for col in PAIR_COLUMNS[2:]:
        data[col] = data[col].astype("category")
    return data


def build_split_features(fc_pair_root: Path, split: str, output_path: Path) -> None:
    labels_df = load_labels(fc_pair_root / f"{split}.csv")
    source_index = index_source_files(fc_pair_root / f"{split}_data")
    rows: list[list[str | int]] = []

    for _, row in labels_df.iterrows():
        file1 = source_index.get(row["function_id_1"])
        file2 = source_index.get(row["function_id_2"])
        if file1 is None or file2 is None:
            continue
        features_df = build_pair_features(file1, file2)
        if features_df.empty:
            continue
        for feature_row in features_df.itertuples(index=False, name=None):
            rows.append(list(feature_row) + [int(row["label"])])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {output_path}")


def evaluate_fc_pair(fc_pair_root: Path, model_path: Path, features_path: Path | None = None) -> None:
    labels_df = load_labels(fc_pair_root / "test.csv")
    source_index = index_source_files(fc_pair_root / "test_data")
    model = joblib.load(model_path)

    if features_path is not None and features_path.exists():
        all_features = pd.read_csv(features_path, header=None)
        all_features.columns = PAIR_COLUMNS + ["label"]
    else:
        all_features = None

    y_true: list[int] = []
    y_pred: list[int] = []

    for _, row in labels_df.iterrows():
        if all_features is None:
            file1 = source_index.get(row["function_id_1"])
            file2 = source_index.get(row["function_id_2"])
            if file1 is None or file2 is None:
                continue
            features_df = build_pair_features(file1, file2)
        else:
            prefix1 = f"{row['function_id_1']}_"
            prefix2 = f"{row['function_id_2']}_"
            features_df = all_features[
                all_features[0].astype(str).str.startswith(prefix1)
                & all_features[1].astype(str).str.startswith(prefix2)
            ].iloc[:, :10].copy()

        if features_df.empty:
            continue

        model_input = prepare_model_input(features_df)
        predictions = model.predict(model_input.iloc[:, 2:10], num_iteration=model.best_iteration, predict_disable_shape_check='true')
        binary_predictions = [1 if value > 0.5 else 0 for value in predictions]
        _, is_clone = detect_similarity(model_input.copy(), binary_predictions)
        y_true.append(int(row["label"]))
        y_pred.append(1 if is_clone else 0)

    if not y_true:
        raise ValueError("No FC-pair test pairs were evaluated")

    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))
    fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
    fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
    accuracy = float(np.mean(y_true_arr == y_pred_arr))
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0

    print(f"Evaluated pairs: {len(y_true_arr)}")
    print(f"Accuracy:{accuracy}")
    print(f"Precision:{precision}")
    print(f"Recall:{recall}")
    print(f"F1-score:{f1}")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("fc_pair_root", type=Path)
    build_parser.add_argument("--split", choices=["train", "test", "both"], default="both")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("fc_pair_root", type=Path)
    eval_parser.add_argument("--model", type=Path, default=BASE_DIR / "model.pkl")
    eval_parser.add_argument("--features", type=Path, default=BASE_DIR / "datasets" / "FC-pair" / "test_features.csv")

    args = parser.parse_args()

    if args.command == "build":
        splits = ["train", "test"] if args.split == "both" else [args.split]
        for split in splits:
            build_split_features(
                fc_pair_root=args.fc_pair_root,
                split=split,
                output_path=BASE_DIR / "datasets" / "FC-pair" / f"{split}_features.csv",
            )
    elif args.command == "eval":
        evaluate_fc_pair(
            fc_pair_root=args.fc_pair_root,
            model_path=args.model,
            features_path=args.features,
        )


if __name__ == "__main__":
    main()
