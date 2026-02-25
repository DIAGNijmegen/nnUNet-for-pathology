#!/usr/bin/env python3

import os
import sys


def find_latest_epoch_line(fold_dir):
    fold_log_files = sorted([
        os.path.join(fold_dir, f)
        for f in os.listdir(fold_dir)
        if f.startswith('training_log_')
    ])

    # newest → oldest
    for log_file in reversed(fold_log_files):
        with open(log_file, 'r') as f:
            log = f.readlines()

        # bottom → top
        for line in reversed(log):
            if ('Epoch' in line) and ('Epoch time:' not in line):
                return line.strip()

    return None


def main(dataset_nr: int):
    nnUNet_results = os.getenv('nnUNet_results')
    if nnUNet_results is None:
        raise EnvironmentError("Environment variable 'nnUNet_results' is not set.")

    dataset = 'Dataset' + str(dataset_nr).zfill(3)

    print("SEARCH:\t", dataset_nr, "-->", dataset)

    # --- find dataset ---
    dataset_names = [
        d for d in os.listdir(nnUNet_results)
        if d.startswith(dataset)
    ]

    if len(dataset_names) == 0:
        raise ValueError(f"Dataset not found for dataset number: {dataset}")
    elif len(dataset_names) > 1:
        raise ValueError(f"Multiple datasets found:\n{dataset_names}")

    dataset_name = dataset_names[0]
    dataset_dir = os.path.join(nnUNet_results, dataset_name)

    print("NAME:\t", dataset_name)

    # --- find models ---
    model_names = sorted(os.listdir(dataset_dir))

    if len(model_names) == 0:
        raise ValueError(f"No models found for dataset {dataset_name}")

    print("MODELS:\t", model_names)

    # --- determine folds from first model ---
    first_model_dir = os.path.join(dataset_dir, model_names[0])
    folds = sorted([
        f for f in os.listdir(first_model_dir)
        if f.startswith('fold')
    ])

    print("FOLDS:\t", folds)

    # --- iterate folds first ---
    print("-" * 50)
    for fold in folds:
        print(f"{fold.upper()}")

        for model_name in model_names:
            model_dir = os.path.join(dataset_dir, model_name)
            fold_dir = os.path.join(model_dir, fold)

            print(f"      -\t{model_name}")

            if not os.path.isdir(fold_dir):
                print("\t(no fold directory found)")
                continue

            epoch_line = find_latest_epoch_line(fold_dir)

            if epoch_line is None:
                print("\tNo epoch line found.")
            else:
                print(f"\t{epoch_line}")
        print("-" * 50)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python check_nnunet_epochs.py <dataset_number>")
        sys.exit(1)

    dataset_nr = int(sys.argv[1])
    main(dataset_nr)
