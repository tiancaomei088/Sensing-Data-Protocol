import os
import random
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from . import readers
from .datasets import CSIDataset
from .utils import load_config, train_model, resize_csi_to_fixed_length
from .processors.base_processor import BaseProcessor
from .models import CSIModel
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau


def pipeline(input_path: str, output_folder: str, dataset: str):
    # params
    ipath = input_path
    os.makedirs(output_folder, exist_ok=True)
    opath = Path(output_folder)
    dataset_name = dataset

    params = load_config(dataset_name)
    batch = params["batch"]
    lr = params["lr"]
    wd = params["wd"]
    num_epochs = params["num_epochs"]
    padding_length = params["padding_length"]

    test_split = 0.4
    val_split = 0.3
    random_seeds = [random.randint(0, 999) for _ in range(5)]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # begin to preprocess, training and eval
    csi_data_list = readers.load_data(ipath, dataset_name)

    processor = BaseProcessor()
    res = processor.process(csi_data_list, dataset=dataset_name)

    unadjusted_data = res[0]
    processed_data = resize_csi_to_fixed_length(unadjusted_data, target_length=padding_length)
    print(f"processed_data's shape: {processed_data[0].shape}")

    labels = res[1]
    groups = res[2]

    unique_labels = sorted(list(set(labels)))
    label_map = {label: i for i, label in enumerate(unique_labels)}
    zero_indexed_labels = [label_map[label] for label in labels]

    unique_groups = sorted(list(set(groups)))
    group_map = {group: i for i, group in enumerate(unique_groups)}
    zero_indexed_groups = [group_map[group] for group in groups]

    print(f"all unique labels idx: {list(set(zero_indexed_labels))}")
    print(f"all unique groups idx: {list(set(zero_indexed_groups))}")
    print(f"total sample: {len(processed_data)}, \
            total labels: {len(zero_indexed_labels)}, total groups: {len(zero_indexed_groups)}")
    print(f"the following 5 seeds will be used: {random_seeds}")


    """
    ===============================
        Training and Evaluation
    ===============================
    """

    processed_data = np.array(processed_data)
    zero_indexed_labels = np.array(zero_indexed_labels)
    zero_indexed_groups = np.array(zero_indexed_groups)
    top1_accuracies = []

    for i, current_seed in enumerate(random_seeds):
        print(f"\n{'=' * 25} epoch {i + 1}/{len(random_seeds)} \
                begin (Random State: {current_seed}) {'=' * 25}\n")

        splitter_1 = GroupShuffleSplit(n_splits=1, test_size=test_split, random_state=current_seed)
        train_idx, temp_idx = \
            next(splitter_1.split(processed_data, zero_indexed_labels, groups=zero_indexed_groups))

        train_data = processed_data[train_idx]
        train_labels = zero_indexed_labels[train_idx]

        temp_data = processed_data[temp_idx]
        temp_labels = zero_indexed_labels[temp_idx]
        temp_groups = zero_indexed_groups[temp_idx]

        splitter_2 = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=current_seed)
        test_idx, val_idx = next(splitter_2.split(temp_data, temp_labels, groups=temp_groups))

        test_data = temp_data[test_idx]
        test_labels = temp_labels[test_idx]

        val_data = temp_data[val_idx]
        val_labels = temp_labels[val_idx]

        train_data = np.stack(train_data, axis=0)
        val_data = np.stack(val_data, axis=0)
        test_data = np.stack(test_data, axis=0)
        print(f"num of samples in train_data: {len(train_data)}, \
                num of samples in test_data: {len(test_data)}, num of samples in val_data: {len(val_data)}")
        print(f"shape of first sample of train_data: {train_data[0].shape}, \
                shape of last sample of train_data: {train_data[-1].shape}")

        train_dataset = CSIDataset(train_data, train_labels)
        test_dataset = CSIDataset(test_data, test_labels)
        val_dataset = CSIDataset(val_data, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch, num_workers=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch, num_workers=16, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=batch, num_workers=16, shuffle=False)

        num_classes = len(unique_labels)
        model = CSIModel(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        checkpoint_path = opath / f"best_checkpoint_{current_seed}.pth"

        print("\n--- begin training ---")
        training_history = train_model(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            device=device,
            checkpoint_path=checkpoint_path
        )

        print(f"\n--- training complete, save training_history to: \
               {opath / f"training_history_{current_seed}.csv"} ---")

        df = pd.DataFrame(training_history)
        df.to_csv(opath / f"training_history_{current_seed}.csv", index_label='epoch')

        print("\n--- save successfully, begin to evaluate model ---")
        cp = checkpoint_path
        if not os.path.isfile(cp):
            raise FileNotFoundError(f" no model in file path: {cp}")

        print(f"loading model from {cp} ...")
        checkpoint = torch.load(cp, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print("loading success, and switch to eval mode")

        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for batch_idx, (csi_data, test_labels) in enumerate(test_loader):
                csi_data = csi_data.to(device)
                test_labels = test_labels.to(device)

                outputs = model(csi_data)

                _, predicted_classes = torch.max(outputs.data, 1)
                all_predictions.extend(predicted_classes.cpu().numpy())
                all_labels.extend(test_labels.cpu().numpy())

        print("eval complete")

        current_top1_acc = accuracy_score(all_labels, all_predictions)
        top1_accuracies.append(current_top1_acc)
        print(f"\n Top-1 acc of current epoch: {current_top1_acc:.4f}")

        print("\n" + "=" * 50)
        print("classification report:")
        print(classification_report(all_labels, all_predictions))
        print("=" * 50 + "\n")

        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f"Confusion Matrix (Random State: {current_seed})", fontsize=16)
        plt.ylabel("Actual Label", fontsize=12)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.tight_layout()

        figure_path = opath / f"cm_rs_{current_seed}.png"
        plt.savefig(figure_path)
        plt.close()

    accuracies_np = np.array(top1_accuracies)
    mean_accuracy = np.mean(accuracies_np)
    variance_accuracy = np.var(accuracies_np)

    print(f"All {len(random_seeds)} Top-1 acc: {[f'{acc:.4f}' for acc in top1_accuracies]}")
    print(f"Avg Top-1 acc: {mean_accuracy:.4f}")
    print(f"Variance of Top-1 acc: {variance_accuracy:.6f}")
    print("=" * 72)

    print(f"\n All pipeline complete")