import os
import torch
import random
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from train_func import train_model
from model import CSIModel


class CSIDataset(Dataset):
    def __init__(self, data_list, labels):
        data_list = np.abs(data_list)
        self.data_list = torch.from_numpy(data_list).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx): 
        return self.data_list[idx], self.labels[idx]


if __name__ == '__main__':
    # param setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FILE_PATH = "../data/elderAL/processed_data.npz"
    HISTORY_PATH = "./training_history.json"
    TEST_SIZE = 0.4
    VAL_SIZE = 0.3
    RANDOM_STATE = [random.randint(0,999) for _ in range(5)]
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-3
    NUM_EPOCHS = 3
    USE_CHECKPOINT = False

    top1_accuracies = []
    data = []
    labels = []
    groups = []

    with np.load(FILE_PATH) as d:
        labels = d['labels']
        data = d['data']
        groups = d['group']

    print(f"total sample: {len(data)}, total labels: {len(labels)}, total groups: {len(groups)}")
    print(f"unique groups: {len(np.unique(groups))}")
    print(f"first 20 group value: {groups[:20]}")
    print(f"the following 5 seeds will be used: {RANDOM_STATE}")
    for i, current_random_state in enumerate(RANDOM_STATE):
        print(f"\n{'='*25} epoch {i+1}/{len(RANDOM_STATE)} begin (Random State: {current_random_state}) {'='*25}\n")

        splitter_1 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=current_random_state)
        train_idx, temp_idx = next(splitter_1.split(data, labels, groups=groups))
        
        train_data = data[train_idx]
        train_labels = labels[train_idx]
        
        temp_data = data[temp_idx]
        temp_labels = labels[temp_idx]
        temp_groups = groups[temp_idx]

        splitter_2 = GroupShuffleSplit(n_splits=1, test_size=VAL_SIZE, random_state=current_random_state)
        test_idx, val_idx = next(splitter_2.split(temp_data, temp_labels, groups=temp_groups))
        
        test_data = temp_data[test_idx]
        test_labels = temp_labels[test_idx]
        
        val_data = temp_data[val_idx]
        val_labels = temp_labels[val_idx]

        train_data = np.stack(train_data, axis=0)
        val_data = np.stack(val_data, axis=0)
        test_data = np.stack(test_data, axis=0)
        print(f"num of samples in train_data: {len(train_data)}, num of samples in test_data: {len(test_data)}, num of samples in val_data: {len(val_data)}")
        print(f"shape of first sample of train_data: {train_data[0].shape}, shape of last sample of train_data: {train_data[-1].shape}")

        train_dataset = CSIDataset(train_data, train_labels)
        test_dataset = CSIDataset(test_data, test_labels)
        val_dataset = CSIDataset(val_data, val_labels)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=16, shuffle=False)


        TIME_STEPS, FREQ_BINS, ANTENNA = train_dataset.data_list[0].shape
        NUM_CLASSES = len(np.unique(labels))
        print(f"shape of inuput data (T,F,A): {TIME_STEPS},{FREQ_BINS},{ANTENNA}, number of classes: {NUM_CLASSES}")

        model = CSIModel(
            task_type='classification',
            num_classes=NUM_CLASSES,
        ).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

        for times in range(1):
            CHECKPOINT_PATH = f"./results/checkpoints/best_checkpoint_{current_random_state}_{times + 1}.pth"
            print("\n--- begin training ---")
            training_history = train_model(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=NUM_EPOCHS,
                device=device,
                checkpoint_path=CHECKPOINT_PATH,
                use_checkpoint=False
            )

            print("\n--- training complete, begin to evaluate model ---")
            test_model = CSIModel(task_type='classification', num_classes=NUM_CLASSES).to(device)
            checkpoint_path = CHECKPOINT_PATH
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(f" no model in file path: {checkpoint_path}")

            print(f"loading model from {checkpoint_path} ...")
            checkpoint = torch.load(checkpoint_path, map_location=device)
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

            print("\n" + "="*50)
            print("classificaton report:")
            print(classification_report(all_labels, all_predictions))
            print("="*50 + "\n")

            cm = confusion_matrix(all_labels, all_predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix (Random State: {current_random_state}) Times: {times}', fontsize=16)
            plt.ylabel('Actual Label', fontsize=12)
            plt.xlabel('Predicted Label', fontsize=12)
            plt.tight_layout()

            figure_path = f'./results/confusion/cm_rs_{current_random_state}_{times + 1}.png'
            plt.savefig(figure_path)
            plt.close()

    accuracies_np = np.array(top1_accuracies)
    mean_accuracy = np.mean(accuracies_np)
    variance_accuracy = np.var(accuracies_np)
    
    print(f"All {len(RANDOM_STATE)} Top-1 acc: {[f'{acc:.4f}' for acc in top1_accuracies]}")
    print(f"Avg Top-1 acc: {mean_accuracy:.4f}")
    print(f"Variance of Top-1 acc: {variance_accuracy:.6f}")
    print("="*72)