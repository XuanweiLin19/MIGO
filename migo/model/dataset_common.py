import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset


class MultiModalDataset(Dataset):
    def __init__(self, data_mod1, data_mod2, ori_data_mod1, ori_data_mod2, labels, batch_info=None):
        self.ori_data_mod1 = ori_data_mod1
        self.data_mod1 = data_mod1
        self.ori_data_mod2 = ori_data_mod2
        self.data_mod2 = data_mod2
        self.labels = labels
        self.batch_info = batch_info

    def __len__(self):
        return len(self.data_mod1)

    def __getitem__(self, idx):
        if self.batch_info is not None:
            return (
                self.ori_data_mod1[idx],
                self.data_mod1[idx],
                self.ori_data_mod2[idx],
                self.data_mod2[idx],
                self.labels[idx],
                self.batch_info[idx],
            )
        return (
            self.ori_data_mod1[idx],
            self.data_mod1[idx],
            self.ori_data_mod2[idx],
            self.data_mod2[idx],
            self.labels[idx],
        )


def split_data(
    scRNA_ori_train_datasets,
    scRNA_train_datasets,
    scATAC_ori_train_datasets,
    scATAC_train_datasets,
    label_train_datasets,
    batch_info=None,
    train_size1=0.8,
    return_indices=False,
):
    train_size = train_size1
    num_samples = len(scRNA_train_datasets)
    num_train = int(train_size * num_samples)
    idx = np.arange(num_samples)
    np.random.shuffle(idx)
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]

    RNA_train_data = scRNA_train_datasets[train_idx]
    ATAC_train_data = scATAC_train_datasets[train_idx]
    ori_RNA_train_data = scRNA_ori_train_datasets[train_idx]
    ori_ATAC_train_data = scATAC_ori_train_datasets[train_idx]

    RNA_test_data = scRNA_train_datasets[test_idx]
    ATAC_test_data = scATAC_train_datasets[test_idx]
    ori_RNA_test_data = scRNA_ori_train_datasets[test_idx]
    ori_ATAC_test_data = scATAC_ori_train_datasets[test_idx]

    label_train_data = label_train_datasets[train_idx]
    label_test_data = label_train_datasets[test_idx]

    if batch_info is not None:
        batch_train_data = batch_info[train_idx]
        batch_test_data = batch_info[test_idx]
        if return_indices:
            return (
                RNA_train_data,
                ATAC_train_data,
                label_train_data,
                RNA_test_data,
                ATAC_test_data,
                label_test_data,
                ori_RNA_train_data,
                ori_ATAC_train_data,
                ori_RNA_test_data,
                ori_ATAC_test_data,
                batch_train_data,
                batch_test_data,
                train_idx,
                test_idx,
            )
        return (
            RNA_train_data,
            ATAC_train_data,
            label_train_data,
            RNA_test_data,
            ATAC_test_data,
            label_test_data,
            ori_RNA_train_data,
            ori_ATAC_train_data,
            ori_RNA_test_data,
            ori_ATAC_test_data,
            batch_train_data,
            batch_test_data,
        )

    if return_indices:
        return (
            RNA_train_data,
            ATAC_train_data,
            label_train_data,
            RNA_test_data,
            ATAC_test_data,
            label_test_data,
            ori_RNA_train_data,
            ori_ATAC_train_data,
            ori_RNA_test_data,
            ori_ATAC_test_data,
            train_idx,
            test_idx,
        )
    return (
        RNA_train_data,
        ATAC_train_data,
        label_train_data,
        RNA_test_data,
        ATAC_test_data,
        label_test_data,
        ori_RNA_train_data,
        ori_ATAC_train_data,
        ori_RNA_test_data,
        ori_ATAC_test_data,
    )


def split_data_train_val_test(
    scRNA_ori_train_datasets,
    scRNA_train_datasets,
    scATAC_ori_train_datasets,
    scATAC_train_datasets,
    label_train_datasets,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    seed=46,
):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "The sum of Train, Validate, and Test must be 1"

    num_samples = len(scRNA_train_datasets)
    indices = np.arange(num_samples)

    if seed is not None:
        np.random.seed(seed)
    shuffled_indices = np.random.permutation(indices)

    train_end = int(train_size * num_samples)
    val_end = train_end + int(val_size * num_samples)

    train_idx = shuffled_indices[:train_end]
    val_idx = shuffled_indices[train_end:val_end]
    test_idx = shuffled_indices[val_end:]

    RNA_train = scRNA_train_datasets[train_idx]
    RNA_val = scRNA_train_datasets[val_idx]
    RNA_test = scRNA_train_datasets[test_idx]

    ATAC_train = scATAC_train_datasets[train_idx]
    ATAC_val = scATAC_train_datasets[val_idx]
    ATAC_test = scATAC_train_datasets[test_idx]

    label_train = label_train_datasets[train_idx]
    label_val = label_train_datasets[val_idx]
    label_test = label_train_datasets[test_idx]

    ori_RNA_train = scRNA_ori_train_datasets[train_idx]
    ori_RNA_val = scRNA_ori_train_datasets[val_idx]
    ori_RNA_test = scRNA_ori_train_datasets[test_idx]

    ori_ATAC_train = scATAC_ori_train_datasets[train_idx]
    ori_ATAC_val = scATAC_ori_train_datasets[val_idx]
    ori_ATAC_test = scATAC_ori_train_datasets[test_idx]

    return (
        RNA_train,
        ATAC_train,
        label_train,
        RNA_val,
        ATAC_val,
        label_val,
        RNA_test,
        ATAC_test,
        label_test,
        ori_RNA_train,
        ori_ATAC_train,
        ori_RNA_val,
        ori_ATAC_val,
        ori_RNA_test,
        ori_ATAC_test,
    )


def create_data_loaders(train_data, test_data, batch_size=100):
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


class DataSplitter:
    def __init__(self, data_mod1, data_mod2, labels):
        self.data_mod1 = data_mod1
        self.data_mod2 = data_mod2
        self.labels = labels

    def split_data(self, test_size=0.2):
        data = np.concatenate((self.data_mod1, self.data_mod2), axis=1)
        data = pd.DataFrame(data)
        data["label"] = self.labels

        msk = np.random.rand(len(data)) < (1 - test_size)
        train_data = data[msk]
        test_data = data[~msk]

        return train_data, test_data


def load_datasets(scRNA_path, scATAC_path, label_path):
    scRNA_datasets = pd.read_csv(scRNA_path, header=None)
    scRNA_datasets = torch.tensor(scRNA_datasets.values)
    scATAC_datasets = pd.read_csv(scATAC_path, header=None)
    scATAC_datasets = torch.tensor(scATAC_datasets.values)
    label = pd.read_csv(label_path, header=None)
    label_encoder = LabelEncoder()
    label_encoded = label_encoder.fit_transform(label.values)
    label_tensor = torch.tensor(label_encoded)

    return scRNA_datasets, scATAC_datasets, label_tensor
