from typing import Deque, List, Optional, Self

from torch import nn
from torch.utils.data import DataLoader
import torch
import pandas as pd
import snntorch as snn
from snntorch import spikegen

from datasets.src.zenke_2a.constants import TEST_DATA_PATH, TRAIN_DATA_PATH
from datasets.src.zenke_2a.datagen import generate_sequential_dataset
from datasets.src.zenke_2a.dataset import SequentialDataset
from model.src.util import MovingAverageLIF, SpikeMovingAverage, TemporalFilter, VarianceMovingAverage

# Zenke's paper uses a theta_rest of -50mV
THETA_REST = 0

# Zenke's paper uses a beta of -1mV
BETA = 1

# Zenke's paper uses a lambda of 1
LAMBDA_HEBBIAN = 1

# Zenke's paper uses a xi of 1e-3
XI = 1e-3

# Zenke's paper uses a delta of 1e-5
DELTA = 1e-5

# Zenke's paper uses tau_rise and tau_fall of these values in units of ms
TAU_RISE_ALPHA = 2
TAU_FALL_ALPHA = 10
TAU_RISE_EPSILON = 5
TAU_FALL_EPSILON = 20

MAX_RETAINED_MEMS = 2

DATA_MEM_ASSUMPTION = 0.5


torch.set_printoptions(precision=10, sci_mode=False)


class Settings:
    def __init__(self,
                 layer_sizes: list[int],
                 beta: float,
                 learn_beta: bool,
                 num_steps: int,
                 data_size: int,
                 batch_size: int,
                 learning_rate: float,
                 epochs: int) -> None:
        self.layer_sizes = layer_sizes
        self.beta = beta
        self.learn_beta = learn_beta
        self.num_steps = num_steps
        self.data_size = data_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs


class LayerSettings:
    def __init__(self, prev_size: int, size: int, next_size: int, beta: float, learn_beta: bool,
                 batch_size: int, learning_rate: float, data_size: int) -> None:
        self.prev_size = prev_size
        self.size = size
        self.next_size = next_size
        self.beta = beta
        self.learn_beta = learn_beta
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.data_size = data_size


class Layer(nn.Module):
    def __init__(self, layer_settings: LayerSettings) -> None:
        super().__init__()

        self.layer_settings = layer_settings

        # weights from prev layer to this layer
        self.forward_weights = nn.Linear(
            layer_settings.prev_size, layer_settings.size)
        self.forward_lif = MovingAverageLIF(batch_size=layer_settings.batch_size, layer_size=layer_settings.size,
                                            beta=layer_settings.beta, learn_beta=layer_settings.learn_beta)

        self.mem_rec: Deque[torch.Tensor] = Deque(maxlen=MAX_RETAINED_MEMS)
        for _ in range(2):
            self.mem_rec.append(torch.zeros(
                layer_settings.batch_size, layer_settings.size))

        self.prev_layer: Optional[Layer] = None
        self.next_layer: Optional[Layer] = None

        self.mem = self.forward_lif.init_leaky()

        self.alpha_filter_first_term = TemporalFilter(
            tau_rise=TAU_RISE_ALPHA, tau_fall=TAU_FALL_ALPHA)
        self.epsilon_filter_second_term = TemporalFilter(
            tau_rise=TAU_RISE_EPSILON, tau_fall=TAU_FALL_EPSILON)
        self.alpha_filter_second_term = TemporalFilter(
            tau_rise=TAU_RISE_ALPHA, tau_fall=TAU_FALL_ALPHA)

        self.data_spike_moving_average = SpikeMovingAverage(
            batch_size=layer_settings.batch_size, data_size=self.layer_settings.data_size)
        self.variance_moving_average = VarianceMovingAverage()

    def set_next_layer(self, next_layer: Self) -> None:
        self.next_layer = next_layer

    def set_prev_layer(self, prev_layer: Self) -> None:
        self.prev_layer = prev_layer

    def forward(self, data: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor]:
        if data is None:
            assert self.prev_layer is not None
            current = self.forward_weights(
                self.prev_layer.spk_rec[-1].detach())
        else:
            data = data.detach()
            current = self.forward_weights(data)

        spk, mem = self.forward_lif(current, self.mem)
        self.mem = mem
        self.mem_rec.append(mem)

        return spk, mem

    def train_forward(self, data: Optional[torch.Tensor] = None) -> None:
        """
        The LPL learning rule is implemented here. It is defined as dw_ji/dt,
        for which we optimize the computation with matrices.

        This learning rule is broken up into three terms:

         1. The first term contains S_j(t) * f'(U_i(t)). We form a matrix via
            some unsqueezes to form a first term matrix of size (batch_size, j,
            i). This is mainly done by matrix multiplication.

         2. The second term contains S_i. We form a matrix via some unsqueezes
            to form a second term matrix of size (batch_size, i, j). This is
            mainly done by expanding the tensor to duplicate values for j
            dimension.

         3. The third term contains S_j. We form a matrix via some unsqueezes to
            form a third term matrix of size (batch_size, j, i). This is mainly
            done by expanding the tensor to duplicate values for i dimension.

        The final dw_ij/dt is formed by a Hadamard product of the first term and
        the second term, and then adding the third term. This is then summed
        across the batch dimension and divided by the batch size to form the
        final dw_ij/dt matrix. We then apply this to the weights.
        """
        # print(self.forward_weights.weight)

        with torch.no_grad():
            if data is not None:
                data_mean = self.data_spike_moving_average.apply(data)
                _variance = self.variance_moving_average.apply(
                    spike=data, spike_moving_average=data_mean)

            spk, _mem = self.forward(data)

            beta = self.layer_settings.beta

            # first term
            prev_layer_mem = torch.ones(
                self.layer_settings.batch_size, self.layer_settings.data_size) * DATA_MEM_ASSUMPTION \
                if self.prev_layer is None else self.prev_layer.mem_rec[-1]
            f_prime_u_i = beta * \
                (1 + beta * abs(prev_layer_mem - THETA_REST)) ** (-2)
            print("f prime u i: ", f_prime_u_i)
            f_prime_u_i = f_prime_u_i.unsqueeze(1)
            most_recent_spike = self.forward_lif.spike_moving_average.spike_rec[-1].unsqueeze(
                2)
            first_term_no_filter = most_recent_spike * f_prime_u_i

            first_term_epsilon = self.epsilon_filter_second_term.apply(
                first_term_no_filter)
            first_term_alpha = self.alpha_filter_first_term.apply(
                first_term_epsilon)
            first_term = self.layer_settings.learning_rate * \
                first_term_alpha * self.layer_settings.learning_rate

            # second term
            prev_layer_most_recent_spike = self.data_spike_moving_average.spike_rec[
                -1] if self.prev_layer is None else self.prev_layer.forward_lif.spike_moving_average.spike_rec[-1]
            prev_layer_two_spikes_ago = self.data_spike_moving_average.spike_rec[
                -2] if self.prev_layer is None else self.prev_layer.forward_lif.spike_moving_average.spike_rec[-2]
            prev_layer_spike_moving_average = self.data_spike_moving_average.tracked_value(
            ) if self.prev_layer is None else self.prev_layer.forward_lif.spike_moving_average.tracked_value()
            prev_layer_variance_moving_average = self.variance_moving_average.tracked_value(
            ) if self.prev_layer is None else self.prev_layer.forward_lif.variance_moving_average.tracked_value()

            second_term_no_filter = -1 * \
                (prev_layer_most_recent_spike -
                 prev_layer_two_spikes_ago) + LAMBDA_HEBBIAN / \
                (prev_layer_variance_moving_average + XI) * \
                (prev_layer_most_recent_spike - prev_layer_spike_moving_average)
            second_term_alpha = self.alpha_filter_second_term.apply(
                second_term_no_filter)
            second_term_alpha = second_term_alpha.unsqueeze(
                1).expand(-1, self.layer_settings.size, -1)

            # third term
            third_term = self.layer_settings.learning_rate * DELTA * \
                self.forward_lif.spike_moving_average.spike_rec[-1]
            prev_layer_size = self.layer_settings.data_size if self.prev_layer is None \
                else self.prev_layer.layer_settings.size
            third_term = third_term.unsqueeze(
                2).expand(-1, -1, prev_layer_size)

            # update weights
            dw_dt = first_term * second_term_alpha + third_term
            dw_dt = dw_dt.sum(0) / dw_dt.shape[0]
            self.forward_weights.weight += dw_dt


class Net(nn.Module):
    def __init__(self, settings: Settings) -> None:
        super().__init__()

        self.settings = settings

        # make settings for each layer
        network_layer_settings = []
        for i, size in enumerate(settings.layer_sizes):
            prev_size = settings.data_size if i == 0 else settings.layer_sizes[i-1]
            next_size = settings.layer_sizes[i+1] if i < len(
                settings.layer_sizes) - 1 else 0
            layer_settings = LayerSettings(
                prev_size, size, next_size, settings.beta, settings.learn_beta,
                settings.batch_size, settings.learning_rate, settings.data_size)
            network_layer_settings.append(layer_settings)

        # make layers
        self.layers = nn.ModuleList()
        for i, layer_spec in enumerate(network_layer_settings):
            layer = Layer(layer_spec)
            self.layers.append(layer)

        # connect layers
        for i, layer in enumerate(self.layers):
            if i > 0:
                layer.set_prev_layer(self.layers[i-1])
            if i < len(network_layer_settings) - 1:
                layer.set_next_layer(self.layers[i+1])

    # TODO: handle test data
    def process_data_online(self, train_loader: DataLoader, test_loader: DataLoader) -> None:
        for epoch in range(self.settings.epochs):
            for i, batch in enumerate(train_loader):
                # permute to (num_steps, batch_size, data_size)
                batch = batch.permute(1, 0, 2)

                # poisson encode
                spike_trains = spikegen.rate(batch, time_var_input=True)

                print(
                    f"Epoch {epoch} - Batch {i} - Sample data: {spike_trains.shape}")

                for timestep in range(spike_trains.shape[0]):
                    for i, layer in enumerate(self.layers):
                        if i == 0:
                            layer.train_forward(spike_trains[timestep])
                        else:
                            layer.train_forward(None)


if __name__ == "__main__":

    torch.manual_seed(1234)

    settings = Settings(
        layer_sizes=[1],
        beta=BETA,
        learn_beta=False,
        num_steps=25,
        data_size=2,
        batch_size=1,
        learning_rate=0.01,
        epochs=10
    )

    train_dataframe = pd.read_csv(TRAIN_DATA_PATH)
    train_sequential_dataset = SequentialDataset(
        train_dataframe, num_timesteps=settings.num_steps)
    train_data_loader = DataLoader(
        train_sequential_dataset, batch_size=settings.batch_size, shuffle=False)

    test_dataframe = pd.read_csv(TEST_DATA_PATH)
    test_sequential_dataset = SequentialDataset(test_dataframe)
    test_data_loader = DataLoader(
        test_sequential_dataset, batch_size=10, shuffle=False)

    net = Net(settings)
    net.process_data_online(train_data_loader, test_data_loader)
