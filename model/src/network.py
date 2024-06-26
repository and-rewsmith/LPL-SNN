import logging
from typing import List

from torch import nn
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.src.layer import Layer
from model.src.settings import LayerSettings, Settings


# TODO: Implement functionality to reset the network in between batches
class Net(nn.Module):
    def __init__(self, settings: Settings) -> None:
        super().__init__()

        self.settings = settings

        # make settings for each layer
        network_layer_settings = []
        for i, size in enumerate(settings.layer_sizes):
            prev_size = settings.data_size if i == 0 else settings.layer_sizes[i - 1]
            next_size = settings.layer_sizes[i + 1] if i < len(
                settings.layer_sizes) - 1 else 0
            layer_id = i
            layer_settings = LayerSettings(layer_id,
                                           prev_size,
                                           size,
                                           next_size,
                                           settings.batch_size,
                                           settings.learning_rate,
                                           settings.data_size,
                                           settings.dt,
                                           settings.percentage_inhibitory,
                                           settings.exc_to_inhib_conn_c,
                                           settings.exc_to_inhib_conn_sigma_squared,
                                           settings.layer_sparsity,
                                           settings.decay_beta,
                                           settings.threshold_scale,
                                           settings.threshold_decay,
                                           settings.tau_mean,
                                           settings.tau_var,
                                           settings.tau_stdp,
                                           settings.tau_rise_alpha,
                                           settings.tau_fall_alpha,
                                           settings.tau_rise_epsilon,
                                           settings.tau_fall_epsilon,
                                           settings.device)
            network_layer_settings.append(layer_settings)

        # make layers
        self.layers = nn.ModuleList()
        for i, layer_spec in enumerate(network_layer_settings):
            layer = Layer(layer_spec)
            self.layers.append(layer)

        # connect layers
        for i, layer in enumerate(self.layers):
            if i > 0:
                layer.set_prev_layer(self.layers[i - 1])
            if i < len(network_layer_settings) - 1:
                layer.set_next_layer(self.layers[i + 1])

        # set masks
        for i, layer in enumerate(self.layers):
            layer.set_sparsity_masks()

    def layer_activations(self) -> List[torch.Tensor]:
        return [layer.retreive_activations() for layer in self.layers]

    def process_data_single_timestep(self, data: torch.Tensor) -> None:
        data = data.to(self.settings.device)

        for i, layer in enumerate(self.layers):
            if i == 0:
                spk = layer.forward(data)
            else:
                spk = layer.forward()

            layer.train_synapses(spk, data)

    # TODO: handle test data
    def process_data_online(self, train_loader: DataLoader) -> None:
        for epoch in range(self.settings.epochs):
            for i, batch in enumerate(train_loader):
                batch = batch.to(self.settings.device)

                # permute to (num_steps, batch_size, data_size)
                batch = batch.permute(1, 0, 2)

                if self.settings.encode_spike_trains:
                    batch = (batch > 0.5).float()

                logging.info(
                    f"Epoch {epoch} - Batch {i} - Sample data: {batch.shape}")

                for timestep in tqdm(range(batch.shape[0]), desc="Timesteps"):
                    for i, layer in enumerate(self.layers):
                        if i == 0:
                            spk = layer.forward(batch[timestep])
                        else:
                            spk = layer.forward()

                        layer.train_synapses(spk, batch[timestep])

                # TODO: remove when network is stabilized
                return
