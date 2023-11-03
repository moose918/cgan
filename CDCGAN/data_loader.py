import json
from collections import namedtuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

JoinedFrame = namedtuple("JoinedFrame", ["prev_frame", "curr_frame"])

class BinaryDataset(Dataset):
    def __init__(self, file="lvls.json", z_dims=2):
        self.data = torch.FloatTensor(self.prep_dataset(file, z_dims))
        self.mid_point = self.data.shape[3] // 2

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        point = self.data[index]
        j_frame = JoinedFrame(
            prev_frame=point[:, :, :, : self.mid_point],
            curr_frame=point[:, :, :, self.mid_point :],  # (ch=2, h=16,w=16)
        )
        return j_frame

    def cuda(self):
        self.data = self.data.to(device="cuda")

    def prep_dataset(self, file, z_dims):
        with open(file, "r") as fp:
            levels = np.array(json.load(fp))

        onehot = np.eye(z_dims, dtype="uint8")[
            levels
        ]  # create a one hot mapping for the features
        onehot = np.rollaxis(onehot, 3, 1)  # (num_samples, chann.=2, h=16, w=16)
        padded = np.full((onehot.shape[0], onehot.shape[1], 32, 32), 0.0)
        padded[:, :, 8:-8, 8:-8] = onehot
        return padded


if __name__ == "__main__":
    # dataset = MarioDataset() # BinaryDataset() # todo
    dataset = BinaryDataset() # BinaryDataset() # todo
    breakpoint()
