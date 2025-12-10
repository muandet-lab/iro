import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from iro.cmnist_exp.vis.common import (
    get_env_loaders,
    load_algorithm_from_ckpt,
    parse_probabilities,
    peek_loss_name,
)
from iro.cmnist_exp.lib import misc


def main():
    parser = argparse.ArgumentParser(description="Bar chart of per-environment losses for both algorithms.")
    parser.add_argument("--iro_ckpt", required=True)
    parser.add_argument("--esrm_ckpt", required=True)
    parser.add_argument("--env_grid", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--iro_alpha", type=str, default="0.7",
                        help="Single value or comma-separated list matching env_grid.")
    parser.add_argument("--esrm_gamma", type=str, default="1.0",
                        help="Single value or comma-separated list matching env_grid.")
    parser.add_argument("--data_dir", type=str, default="../../data/")
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--label_noise", type=float, default=0.25)
    parser.add_argument("--full_resolution", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="vis_per_env_losses.png")
    args = parser.parse_args()

    env_ps = parse_probabilities(args.env_grid)
    if len(env_ps) == 0:
        raise ValueError("Provide at least one environment probability via --env_grid.")
    def _parse_params(spec):
        vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
        if len(vals) == 1:
            return vals * len(env_ps)
        if len(vals) != len(env_ps):
            raise ValueError("Risk parameter list must have length 1 or match env_grid length.")
        return vals
    iro_params = _parse_params(args.iro_alpha)
    esrm_params = _parse_params(args.esrm_gamma)

    device = torch.device(args.device)
    loss_name = peek_loss_name(args.iro_ckpt)
    if peek_loss_name(args.esrm_ckpt) != loss_name:
        raise ValueError("Checkpoints were trained with different loss functions.")
    loaders, input_shape = get_env_loaders(
        args.data_dir, env_ps, device,
        loss_name=loss_name,
        batch_size=args.batch_size,
        label_noise=args.label_noise,
        full_resolution=args.full_resolution,
        use_test_set=True
    )

    iro_algorithm, _, iro_loss_fn, _, _ = load_algorithm_from_ckpt(args.iro_ckpt, device, input_shape)
    esrm_algorithm, _, esrm_loss_fn, _, _ = load_algorithm_from_ckpt(args.esrm_ckpt, device, input_shape)

    iro_losses = []
    esrm_losses = []
    for loader, alpha, gamma in zip(loaders, iro_params, esrm_params):
        iro_losses.append(misc.loss(iro_algorithm, loader, iro_loss_fn, device, alpha=alpha))
        esrm_losses.append(misc.loss(esrm_algorithm, loader, esrm_loss_fn, device, alpha=gamma))

    indices = np.arange(len(env_ps))
    width = 0.35
    plt.figure(figsize=(12, 5))
    plt.bar(indices - width / 2, iro_losses, width=width, label="IRO")
    plt.bar(indices + width / 2, esrm_losses, width=width, label="ESRM")
    plt.xticks(indices, [f"{p:.1f}" for p in env_ps], rotation=45)
    plt.ylabel("Loss")
    plt.xlabel("Color-correlation probability")
    plt.title("Per-environment losses at fixed risk parameters")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved bar chart to {args.output}")


if __name__ == "__main__":
    main()
