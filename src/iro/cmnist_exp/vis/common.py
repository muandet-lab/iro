import argparse
import copy
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from iro.cmnist_exp import algorithms, networks
from iro.cmnist_exp.datasets import get_cmnist_datasets
from iro.cmnist_exp.lib.fast_data_loader import FastDataLoader
from iro.cmnist_exp.lib import misc


def peek_loss_name(ckpt_path: str, default: str = "nll") -> str:
    payload = torch.load(ckpt_path, map_location="cpu")
    args_dict = payload.get("args", {})
    return args_dict.get("loss_fn", default)


def parse_probabilities(spec: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in spec.split(",") if x.strip())


def get_loss_setup(loss_name: str):
    if loss_name == "cross_ent":
        return F.cross_entropy, 2, True
    return F.binary_cross_entropy_with_logits, 1, False


def build_network_from_args(model_args, input_shape, n_targets):
    hidden_dim = getattr(model_args, "mlp_hidden_dim", 390)
    dropout = getattr(model_args, "dropout_p", 0.2)
    network_name = getattr(model_args, "network", "FiLMedMLP")
    if network_name == "MLP":
        return networks.MLP(np.prod(input_shape), hidden_dim, n_targets, dropout=dropout)
    if network_name == "FiLMedMLP":
        return networks.FiLMedMLP(np.prod(input_shape), hidden_dim, n_targets,
                                  dropout=dropout, film_dim=1)
    if network_name == "CNN":
        return networks.CNN(input_shape)
    raise NotImplementedError(f"Unknown network architecture {network_name}")


def load_algorithm_from_ckpt(ckpt_path: str, device: torch.device, input_shape):
    payload = torch.load(ckpt_path, map_location=device)
    args_dict = payload.get("args", {})
    model_args = argparse.Namespace(**args_dict)
    loss_name = getattr(model_args, "loss_fn", "nll")
    loss_fn, n_targets, _ = get_loss_setup(loss_name)
    alg_name = getattr(model_args, "algorithm", "iro").lower()
    alg_class = algorithms.get_algorithm_class(alg_name)
    network = build_network_from_args(model_args, input_shape, n_targets)
    hparams = copy.deepcopy(vars(model_args))
    algorithm = alg_class(network, hparams, loss_fn)
    state_dict = payload.get("model_dict", payload)
    algorithm.load_state_dict(state_dict, strict=False)
    algorithm.to(device)
    algorithm.eval()
    return algorithm, model_args, loss_fn, loss_name, n_targets


def get_env_loaders(data_dir: str,
                    env_ps: Sequence[float],
                    device: torch.device,
                    loss_name: str,
                    batch_size: int = 5000,
                    label_noise: float = 0.25,
                    full_resolution: bool = False,
                    use_test_set: bool = True):
    _, _, int_target = get_loss_setup(loss_name)
    envs = get_cmnist_datasets(
        data_dir,
        train_envs=[],
        test_envs=tuple(env_ps),
        label_noise_rate=label_noise,
        cuda=(device.type == "cuda"),
        int_target=int_target,
        subsample=not full_resolution,
        use_test_set=use_test_set
    )
    loaders = [
        FastDataLoader(dataset=env, batch_size=batch_size, num_workers=0)
        for env in envs
    ]
    input_shape = envs[0].tensors[0].size()[1:]
    return loaders, input_shape


def collect_metrics(algorithm, loaders, device, risk_param, loss_fn):
    accs, losses = [], []
    for loader in loaders:
        accs.append(misc.accuracy(algorithm, loader, device, alpha=risk_param))
        losses.append(misc.loss(algorithm, loader, loss_fn, device, alpha=risk_param))
    return np.array(accs), np.array(losses)


def compute_tail_accuracy(risks: torch.Tensor,
                          accs: torch.Tensor,
                          algorithm_name: str,
                          risk_param: float) -> float:
    if algorithm_name.lower() == "iro":
        var = torch.quantile(risks, risk_param, interpolation="linear")
        mask = risks >= var
        return accs[mask].mean().item()
    if algorithm_name.lower() == "esrm":
        gamma = torch.tensor(risk_param, dtype=risks.dtype, device=risks.device)
        n = risks.numel()
        if n == 0:
            return float("nan")
        sorted_risks, sorted_idx = torch.sort(risks)
        sorted_accs = accs[sorted_idx]
        u = (torch.arange(n, dtype=risks.dtype, device=risks.device) + 0.5) / n
        weights = gamma * torch.exp(-gamma * (1.0 - u))
        weights = weights / torch.clamp(1.0 - torch.exp(-gamma), min=1e-6)
        weights = weights / weights.sum()
        return torch.sum(sorted_accs * weights).item()
    return accs.mean().item()


def get_prediction_scores(algorithm, loader, device, loss_name, risk_param):
    probs, targets = [], []
    algorithm.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            t_param = torch.full((x.shape[0], 1), float(risk_param), device=device)
            logits = algorithm.predict(x, t_param)
            if loss_name == "cross_ent":
                prob = torch.softmax(logits, dim=1)[:, 1]
            else:
                prob = torch.sigmoid(logits.view(-1))
            probs.append(prob.detach().cpu())
            targets.append(y.detach().cpu().float())
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()
