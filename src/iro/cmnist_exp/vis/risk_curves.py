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
    parser = argparse.ArgumentParser(description="Plot accuracy/loss curves across environments for IRO vs ESRM.")
    parser.add_argument("--iro_ckpt", type=str, required=True, help="Checkpoint produced by train_sandbox.py for IRO.")
    parser.add_argument("--esrm_ckpt", type=str, required=True, help="Checkpoint produced by train_sandbox.py for ESRM.")
    parser.add_argument("--env_grid", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                        help="Comma-separated probabilities defining evaluation environments.")
    parser.add_argument("--iro_alpha", type=str, default="0.7",
                        help="Single value or comma-separated list matching env_grid.")
    parser.add_argument("--esrm_gamma", type=str, default="1.0",
                        help="Single value or comma-separated list matching env_grid.")
    parser.add_argument("--data_dir", type=str, default="../../data/", help="Root directory for CMNIST data.")
    parser.add_argument("--batch_size", type=int, default=5000, help="Batch size for evaluation loaders.")
    parser.add_argument("--label_noise", type=float, default=0.25)
    parser.add_argument("--full_resolution", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="vis_risk_curves.png")
    args = parser.parse_args()

    env_ps = parse_probabilities(args.env_grid)
    if len(env_ps) == 0:
        raise ValueError("Provide at least one environment probability via --env_grid.")

    device = torch.device(args.device)
    iro_loss_name = peek_loss_name(args.iro_ckpt)
    esrm_loss_name = peek_loss_name(args.esrm_ckpt)
    if esrm_loss_name != iro_loss_name:
        raise ValueError("IRO and ESRM checkpoints were trained with different loss functions.")

    loaders, input_shape = get_env_loaders(
        args.data_dir, env_ps, device,
        loss_name=iro_loss_name,
        batch_size=args.batch_size,
        label_noise=args.label_noise,
        full_resolution=args.full_resolution,
        use_test_set=True
    )
    def _parse_params(spec):
        vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
        if len(vals) == 1:
            return vals * len(env_ps)
        if len(vals) != len(env_ps):
            raise ValueError("Risk parameter list must have length 1 or match env_grid length.")
        return vals
    iro_params = _parse_params(args.iro_alpha)
    esrm_params = _parse_params(args.esrm_gamma)

    iro_algorithm, _, iro_loss_fn, _, _ = load_algorithm_from_ckpt(args.iro_ckpt, device, input_shape)
    esrm_algorithm, _, esrm_loss_fn, _, _ = load_algorithm_from_ckpt(args.esrm_ckpt, device, input_shape)

    iro_accs, iro_losses = [], []
    esrm_accs, esrm_losses = [], []
    for loader, alpha in zip(loaders, iro_params):
        iro_accs.append(misc.accuracy(iro_algorithm, loader, device, alpha=alpha))
        iro_losses.append(misc.loss(iro_algorithm, loader, iro_loss_fn, device, alpha=alpha))
    for loader, gamma in zip(loaders, esrm_params):
        esrm_accs.append(misc.accuracy(esrm_algorithm, loader, device, alpha=gamma))
        esrm_losses.append(misc.loss(esrm_algorithm, loader, esrm_loss_fn, device, alpha=gamma))
    iro_accs = np.array(iro_accs)
    iro_losses = np.array(iro_losses)
    esrm_accs = np.array(esrm_accs)
    esrm_losses = np.array(esrm_losses)

    env_ps_np = np.array(env_ps)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(env_ps_np, iro_accs, label=f"IRO (alpha={args.iro_alpha})", marker="o")
    axes[0].plot(env_ps_np, esrm_accs, label=f"ESRM (gamma={args.esrm_gamma})", marker="s")
    axes[0].set_xlabel("Color-correlation probability")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Accuracy vs environment shift")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(env_ps_np, iro_losses, label="IRO", marker="o")
    axes[1].plot(env_ps_np, esrm_losses, label="ESRM", marker="s")
    axes[1].set_xlabel("Color-correlation probability")
    axes[1].set_ylabel("Aggregated loss")
    axes[1].set_title("Risk vs environment shift")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("Risk vs environment correlation")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)
    print(f"Saved risk curves to {args.output}")


if __name__ == "__main__":
    main()
