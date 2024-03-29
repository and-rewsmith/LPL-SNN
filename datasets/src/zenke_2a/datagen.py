import numpy as np
import pandas as pd
from tqdm import tqdm

from datasets.src.zenke_2a.constants import TEST_DATA_PATH, TRAIN_DATA_PATH

NUM_TIMESTEPS = 100
PLANNED_BATCH_SIZE = 1

# TODO: can use guassian mixture model to generate data


def generate_sequential_dataset(
        num_samples: int = PLANNED_BATCH_SIZE,
        num_datapoints: int = NUM_TIMESTEPS,
        num_clusters: int = 2,
        cluster_switch_prob: float = 0.02,
        cluster_spread: float = 0.5) -> pd.DataFrame:
    """
    Generates a sequential dataset with rare abrupt switches between clusters, organized into batches,
    with a defined number of samples per batch.

    Parameters:
    num_batches (int): Number of batches.
    samples_per_batch (int): Number of samples per batch.
    num_clusters (int): Number of clusters.
    cluster_switch_prob (float): Probability of switching to a different cluster.
    cluster_spread (float): Variability within the cluster.

    Returns:
    pd.DataFrame: DataFrame containing the sequential dataset with batch information.
    """
    data = []  # List to store data before converting to DataFrame

    for sample in tqdm(range(num_samples), desc='Generating samples'):
        current_cluster = np.random.randint(0, num_clusters)
        for i in range(num_datapoints):
            # Decide whether to switch clusters
            if np.random.rand() < cluster_switch_prob:
                current_cluster = (current_cluster + 1) % num_clusters

            # Cluster center along the x-axis
            center_x = current_cluster * 2.0

            # Generate the data point
            data_point = [sample, np.random.normal(
                center_x, 0.1), np.random.random()]
            data.append(data_point)

    # Convert list to DataFrame
    df = pd.DataFrame(data, columns=['sample', 'x', 'y'])

    # Normalize columns independently
    df[['x', 'y']] = (df[['x', 'y']] - df[['x', 'y']].min()) / \
        (df[['x', 'y']].max() - df[['x', 'y']].min())

    return df


if __name__ == "__main__":
    sequential_data = generate_sequential_dataset()
    sequential_data.to_csv(TRAIN_DATA_PATH, index=False)
    print(f"Sequential dataset saved to {TRAIN_DATA_PATH}")

    test_sequential_data = generate_sequential_dataset()
    test_sequential_data.to_csv(TEST_DATA_PATH, index=False)
    print(f"Sequential test dataset saved to {TEST_DATA_PATH}")
