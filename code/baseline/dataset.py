import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class AudioDataset(Dataset):
    def __init__(self, data_path, transform):
        self.transform = transform
        self.data_path = data_path
        self.file_path = [os.path.join(str(i//1000), str(i)+'.npy') for i in range(10000)]

    def __getitem__(self, idx):
        arr = np.load(self.data_path + self.file_path[idx])
        arr = self.transform(arr)
        # 길이 안맞는 tensor 패딩
        if arr.shape[2] < 1876:
            b = np.zeros((1,48, 1876-arr.shape[2]))
            arr = torch.cat([arr,torch.tensor(b)], dim=2)
        return arr
        
    def __len__(self):
        return len(self.file_path)

if __name__ == '__main__':
    data_path = '../../data/arena_mel/'
    dataset = AudioDataset(data_path)
    # print(dataset[0])

    batch_size = 4
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle=True)
    print(next(iter(dataloader)).shape)