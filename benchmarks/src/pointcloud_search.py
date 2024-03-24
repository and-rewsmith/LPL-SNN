import logging
import random
import time

import pandas as pd
import wandb
import torch
from torch.utils.data import DataLoader

from model.src import logging_util
from benchmarks.src.pointcloud import ENCODE_SPIKE_TRAINS
from datasets.src.zenke_2a.constants import TRAIN_DATA_PATH
from datasets.src.zenke_2a.dataset import DatasetType, SequentialDataset
from model.src.constants import DECAY_BETA, TAU_FALL_ALPHA, TAU_FALL_EPSILON, TAU_MEAN, TAU_RISE_ALPHA, TAU_RISE_EPSILON, TAU_STDP, TAU_VAR
from model.src.layer import Layer
from model.src.network import Net
from model.src.settings import Settings
from model.src.visualizer import NetworkVisualizer

THIS_TEST_NUM_SAMPLES = 5
THIS_TEST_NUM_DATAPOINTS = 8000


def objective():
    wandb.init(
        project="LPL-SNN-2",
        config={
            "architecture": "initial",
            "dataset": "point-cloud",
        },
        allow_val_change=True  # TODOPRE: review this as it is silencing a warning
    )

    layer_sizes = wandb.config.layer_sizes
    learning_rate = wandb.config.learning_rate
    dt = wandb.config.dt
    percentage_inhibitory = wandb.config.percentage_inhibitory
    exc_to_inhib_conn_c = wandb.config.exc_to_inhib_conn_c
    exc_to_inhib_conn_sigma_squared = wandb.config.exc_to_inhib_conn_sigma_squared
    layer_sparsity = wandb.config.layer_sparsity
    decay_beta = wandb.config.decay_beta
    tau_mean = wandb.config.tau_mean
    tau_var = wandb.config.tau_var
    tau_stdp = wandb.config.tau_stdp
    tau_rise_alpha = wandb.config.tau_rise_alpha
    tau_fall_alpha = wandb.config.tau_fall_alpha
    tau_rise_epsilon = wandb.config.tau_rise_epsilon
    tau_fall_epsilon = wandb.config.tau_fall_epsilon

    run_settings = f"""
    running with:
    layer_sizes: {layer_sizes}
    learning_rate: {learning_rate}
    dt: {dt}
    percentage_inhibitory: {percentage_inhibitory}
    exc_to_inhib_conn_c: {exc_to_inhib_conn_c}
    exc_to_inhib_conn_sigma_squared: {exc_to_inhib_conn_sigma_squared}
    layer_sparsity: {layer_sparsity}
    decay_beta: {decay_beta},
    tau_mean: {tau_mean},
    tau_var: {tau_var},
    tau_stdp: {tau_stdp},
    tau_rise_alpha: {tau_rise_alpha},
    tau_fall_alpha: {tau_fall_alpha},
    tau_rise_epsilon: {tau_rise_epsilon},
    tau_fall_epsilon: {tau_fall_epsilon},
    """
    logging.info(run_settings)

    with open("running_log.log", "a") as running_log:
        running_log.write(f"{run_settings}")
        running_log.flush()

        pass_count = 0
        total_count = 0
        for _ in range(10):
            is_pass = bench_specific_seed(
                running_log,
                layer_sizes, learning_rate, dt, percentage_inhibitory,
                exc_to_inhib_conn_c, exc_to_inhib_conn_sigma_squared, layer_sparsity,
                decay_beta, tau_mean, tau_var, tau_stdp, tau_rise_alpha, tau_fall_alpha,
                tau_rise_epsilon, tau_fall_epsilon
            )
            if is_pass:
                pass_count += 1
            total_count += 1

        running_log.write(
            run_settings + f"\npass_rate: {pass_count / total_count}\n\n===============================================================================")
        running_log.flush()

    wandb.log({"pass_rate": pass_count / total_count})


def bench_specific_seed(running_log, layer_sizes, learning_rate, dt, percentage_inhibitory, exc_to_inhib_conn_c, exc_to_inhib_conn_sigma_squared, layer_sparsity,
                        decay_beta, tau_mean, tau_var, tau_stdp, tau_rise_alpha, tau_fall_alpha, tau_rise_epsilon, tau_fall_epsilon):
    rand = random.randint(1000, 9999)
    torch.manual_seed(rand)

    settings = Settings(
        layer_sizes=layer_sizes,
        data_size=2,
        batch_size=THIS_TEST_NUM_SAMPLES,
        learning_rate=learning_rate,
        epochs=10,
        encode_spike_trains=ENCODE_SPIKE_TRAINS,
        dt=dt,
        percentage_inhibitory=percentage_inhibitory,
        exc_to_inhib_conn_c=exc_to_inhib_conn_c,
        exc_to_inhib_conn_sigma_squared=exc_to_inhib_conn_sigma_squared,
        layer_sparsity=layer_sparsity,
        decay_beta=decay_beta,
        tau_mean=tau_mean,
        tau_var=tau_var,
        tau_stdp=tau_stdp,
        tau_rise_alpha=tau_rise_alpha,
        tau_fall_alpha=tau_fall_alpha,
        tau_rise_epsilon=tau_rise_epsilon,
        tau_fall_epsilon=tau_fall_epsilon,
        device=torch.device("cpu")
    )

    try:
        train_dataframe = pd.read_csv(TRAIN_DATA_PATH)
    except FileNotFoundError:
        train_dataframe = None
    train_sequential_dataset = SequentialDataset(DatasetType.TRAIN,
                                                 train_dataframe, num_timesteps=THIS_TEST_NUM_DATAPOINTS,
                                                 planned_batch_size=settings.batch_size)
    train_data_loader = DataLoader(
        train_sequential_dataset, batch_size=settings.batch_size, shuffle=False)

    net = Net(settings).to(settings.device)

    layer: Layer = net.layers[0]
    weights = layer.forward_weights.weight()
    mask = layer.excitatory_mask_vec
    mask_expanded = mask.unsqueeze(
        1).expand(-1, layer.layer_settings.data_size)
    starting_weights_filtered_and_masked = weights[mask_expanded.bool()]

    net.process_data_online(train_data_loader)

    weights = layer.forward_weights.weight()
    weights_filtered_and_masked = weights[mask_expanded.bool()]

    is_pass = weights_filtered_and_masked[0] > 0.3 and weights_filtered_and_masked[1] < 0.05

    message = f"""---------------------------------
    starting weights: {starting_weights_filtered_and_masked}
    ending weights: {weights_filtered_and_masked}
    is_pass: {is_pass}
    ---------------------------------
    """
    running_log.write(message)
    running_log.flush()
    logging.info(message)

    return is_pass


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    torch.set_printoptions(precision=10, sci_mode=False)
    logging_util.set_logging()

    running_log = open("running_log.log", "w")
    message = f"Sweep logs. Current datetime: {time.ctime()}\n"
    running_log.write(message)
    running_log.close()
    logging.debug(message)

    sweep_configuration = {
        "method": "random",
        "metric": {"goal": "maximize", "name": "pass_rate"},
        "parameters": {
            "layer_sizes": {"values": [[2, 5], [2, 10, 10], [2, 10, 10, 10]]},
            "learning_rate": {"values": [0.01, 0.001, 0.0001]},
            "dt": {"values": [1, 0.1, 0.01, 0.001]},
            "percentage_inhibitory": {"values": [60, 50, 40, 30]},
            "exc_to_inhib_conn_c": {"values": [0.25, 0.5, 0.75]},
            "exc_to_inhib_conn_sigma_squared": {"values": [1, 5, 10, 20, 40, 60]},
            "layer_sparsity": {"values": [0.1, 0.3, 0.5, 0.7, 0.9]},
            "tau_mean": {"values": [30, 60, 300, 600, 1200, 1800]},
            "tau_var": {"values": [0.01, 0.02, 0.05, 0.1]},
            "tau_stdp": {"values": [0.01, 0.02, 0.05, 0.1]},
            "tau_rise_alpha": {"values": [0.001, 0.002, 0.005, 0.01]},
            "tau_fall_alpha": {"values": [0.005, 0.01, 0.02, 0.05]},
            "tau_rise_epsilon": {"values": [0.002, 0.005, 0.01, 0.02]},
            "tau_fall_epsilon": {"values": [0.01, 0.02, 0.05, 0.1]},
            "decay_beta": {"values": [0.8, 0.85, 0.9]},
        },
    }

    sweep_id = wandb.sweep(sweep=sweep_configuration, project="LPL-SNN-2")
    wandb.agent(sweep_id, function=objective)
