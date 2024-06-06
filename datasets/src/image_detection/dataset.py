from enum import Enum
from typing import List, Union
import random

import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self,
                 num_timesteps_each_image: int,
                 num_switches: int,
                 device: str,
                 max_samples: int = 1024 * 2) -> None:
        self.num_classes = 10
        self.num_timesteps_each_image = num_timesteps_each_image
        self.num_switches = num_switches
        self.images = [num for num in range(0, self.num_classes)]
        self.len = max_samples
        self.device = device

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return self.len

    def __getitem__(self,
                    idx: Union[int,
                               List[int],
                               torch.Tensor]) -> torch.Tensor:
        out = []  # List to store image indices
        out_labels = []  # List to store switch labels
        for i in range(self.num_switches):
            image_index = random.randint(0, len(self.images)-1)
            out.extend([image_index] * self.num_timesteps_each_image)
            out_labels.append(image_index)

        # out: List of image indices
        #   Dimensions: (num_switches * num_timesteps_each_image,)

        # Convert out to a one-hot encoding of num_classes
        out_one_hot = torch.zeros((len(out), self.num_classes), dtype=torch.float, device=self.device)
        # out_one_hot: Tensor of shape (num_switches * num_timesteps_each_image, num_classes)
        #   Dimensions: (num_switches * num_timesteps_each_image, num_classes)

        out_one_hot[torch.arange(len(out), device=self.device), out] = 1
        # out_one_hot: Tensor of shape (num_switches * num_timesteps_each_image, num_classes)
        #   Dimensions: (num_switches * num_timesteps_each_image, num_classes)

        # out_labels: List of switch labels
        #   Dimensions: (num_switches,)

        return out_one_hot, torch.tensor(out_labels, dtype=torch.long, device=self.device)
        # Returns:
        #   - out_one_hot: Tensor of shape (num_switches * num_timesteps_each_image, num_classes)
        #   - out_labels: Tensor of shape (num_switches,)