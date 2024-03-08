from enum import Enum
import logging
from typing import List, Union

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from datasets.src.zenke_2a.constants import TEST_DATA_PATH, TRAIN_DATA_PATH
from datasets.src.zenke_2a.datagen import generate_sequential_dataset

NUM_TIMESTEPS = 100
PLANNED_BATCH_SIZE = 1


class DatasetType(Enum):
    TRAIN = 0
    TEST = 1


class SequentialDataset(Dataset):
    """
    A PyTorch Dataset superclass for handling sequential data generated by the script.
    """

    def __init__(self, dataset_type: DatasetType, dataframe_tmp: pd.DataFrame, num_timesteps: int = NUM_TIMESTEPS,
                 planned_batch_size: int = PLANNED_BATCH_SIZE) -> None:
        """
        Initializes the dataset by loading the data from a CSV file.

        Parameters:
        csv_file (str): Path to the CSV file containing the generated data.
        """

        # check the dims of this dataframe and if the dataframe dims don't
        # match, then regenerate the data
        samples = dataframe_tmp.groupby('sample')
        sample_data = samples.get_group(0)
        sample_data = sample_data[['x', 'y']].to_numpy()
        sample_tensor = torch.tensor(sample_data, dtype=torch.float)

        if len(dataframe_tmp) % planned_batch_size != 0 or sample_tensor.shape[0] != num_timesteps:
            logging.warning(
                "Dataframe dimensions do not match the planned batch size or number of timesteps. Regenerating data...")
            dataframe_tmp = generate_sequential_dataset(num_samples=planned_batch_size, num_datapoints=num_timesteps)
            path = TRAIN_DATA_PATH if dataset_type == DatasetType.TRAIN else TEST_DATA_PATH
            dataframe_tmp.to_csv(path, index=False)
            self.dataframe = pd.read_csv(path)
            del dataframe_tmp
        else:
            self.dataframe = dataframe_tmp

        self.samples = self.dataframe.groupby('sample')
        self.num_timesteps = num_timesteps
        return

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: Union[int, List[int], torch.Tensor]) -> torch.Tensor:
        """
        Retrieves a sample from the dataset at the specified index.

        Parameters:
        idx (int): Index of the sample to retrieve.

        Returns:
        torch.Tensor: Tensor containing all datapoints of the requested sample.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()  # type: ignore[union-attr]

        sample_data = self.samples.get_group(idx)
        sample_data = sample_data[['x', 'y']].to_numpy()
        sample_tensor = torch.tensor(sample_data, dtype=torch.float)

        return sample_tensor[0:self.num_timesteps, :]


if __name__ == "__main__":
    dataframe = pd.read_csv(TRAIN_DATA_PATH)
    sequential_dataset = SequentialDataset(DatasetType.TRAIN, dataframe, num_timesteps=10)

    data_loader = DataLoader(sequential_dataset, batch_size=10, shuffle=False)

    for i, batch in enumerate(data_loader):
        print(f"Batch {i} - Sample data: {batch.shape}")
        break  # Only showing the first batch for demonstration purposes
