import torch


class Settings:
    def __init__(self,
                 layer_sizes: list[int],
                 data_size: int,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int,
                 encode_spike_trains: bool,
                 dt: float,
                 percentage_inhibitory: float,
                 exc_to_inhib_conn_c: float,
                 exc_to_inhib_conn_sigma_squared: float,
                 layer_sparsity: float,
                 device: torch.device) -> None:
        self.layer_sizes = layer_sizes
        self.data_size = data_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.encode_spike_trains = encode_spike_trains
        self.dt = dt
        self.percentage_inhibitory = percentage_inhibitory
        self.exc_to_inhib_conn_c = exc_to_inhib_conn_c
        self.exc_to_inhib_conn_sigma_squared = exc_to_inhib_conn_sigma_squared
        self.layer_sparsity = layer_sparsity
        self.device = device


class LayerSettings:
    def __init__(self, layer_id: int, prev_size: int, size: int, next_size: int,
                 batch_size: int, learning_rate: float, data_size: int, dt: float,
                 percentage_inhibitory: float, exc_to_inhib_conn_c: float,
                 exc_to_inhib_conn_sigma_squared: float, layer_sparsity: float, device: torch.device) -> None:
        self.layer_id = layer_id
        self.prev_size = prev_size
        self.size = size
        self.next_size = next_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_size = data_size
        self.dt = dt
        self.percentage_inhibitory = percentage_inhibitory
        self.exc_to_inhib_conn_c = exc_to_inhib_conn_c
        self.exc_to_inhib_conn_sigma_squared = exc_to_inhib_conn_sigma_squared
        self.layer_sparsity = layer_sparsity
        self.device = device
