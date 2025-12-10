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


def _parse_candidates(spec, flag_name):
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError(f"Provide at least one value for {flag_name}.")
    return vals


def evaluate_param(loader, algorithm, loss_fn, device, candidates, metric):
    best_val = None
    best_metric = None
    best_acc = None
    best_loss = None
    for candidate in candidates:
        acc = misc.accuracy(algorithm, loader, device, alpha=candidate)
        loss = misc.loss(algorithm, loader, loss_fn, device, alpha=candidate)
        value = -acc if metric == "accuracy" else loss
        if best_metric is None or value < best_metric:
            best_metric = value
            best_val = candidate
            best_acc = acc
            best_loss = loss
    return best_val, best_acc, best_loss


def main():
    parser = argparse.ArgumentParser(
        description="Per-environment losses with adaptive IRO/ESRM risk parameters."
    )
    parser.add_argument("--iro_ckpt", required=True)
    parser.add_argument("--esrm_ckpt", required=True)
    parser.add_argument("--env_grid", type=str,
                        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    parser.add_argument("--iro_alpha_candidates", type=str, required=True,
                        help="Comma-separated alpha candidates searched per environment.")
    parser.add_argument("--esrm_gamma_candidates", type=str, required=True,
                        help="Comma-separated gamma candidates searched per environment.")
    parser.add_argument("--selection_metric", type=str, default="loss",
                        choices=["loss", "accuracy"],
                        help="Metric used to pick the parameter per environment.")
    parser.add_argument("--data_dir", type=str, default="../../data/")
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--label_noise", type=float, default=0.25)
    parser.add_argument("--full_resolution", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="vis_per_env_losses_stable.png")
    args = parser.parse_args()

    env_ps = parse_probabilities(args.env_grid)
    if len(env_ps) == 0:
        raise ValueError("Provide at least one environment probability via --env_grid.")
    iro_candidates = _parse_candidates(args.iro_alpha_candidates, "--iro_alpha_candidates")
    esrm_candidates = _parse_candidates(args.esrm_gamma_candidates, "--esrm_gamma_candidates")

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

    iro_algorithm, _, iro_loss_fn, _, _ = load_algorithm_from_ckpt(args.iro_ckpt, device, input_shape)
    esrm_algorithm, _, esrm_loss_fn, _, _ = load_algorithm_from_ckpt(args.esrm_ckpt, device, input_shape)

    iro_losses, iro_accs, iro_selected = [], [], []
    esrm_losses, esrm_accs, esrm_selected = [], [], []

    for loader in loaders:
        alpha, acc, loss = evaluate_param(
            loader, iro_algorithm, iro_loss_fn, device,
            iro_candidates, args.selection_metric
        )
        iro_selected.append(alpha)
        iro_accs.append(acc)
        iro_losses.append(loss)

        gamma, acc_e, loss_e = evaluate_param(
            loader, esrm_algorithm, esrm_loss_fn, device,
            esrm_candidates, args.selection_metric
        )
        esrm_selected.append(gamma)
        esrm_accs.append(acc_e)
        esrm_losses.append(loss_e)

    indices = np.arange(len(env_ps))
    width = 0.35
    plt.figure(figsize=(12, 5))
    plt.bar(indices - width / 2, iro_losses, width=width, label="IRO (adaptive)")
    plt.bar(indices + width / 2, esrm_losses, width=width, label="ESRM (adaptive)")
    plt.xticks(indices, [f"{p:.1f}" for p in env_ps], rotation=45)
    plt.ylabel("Loss")
    plt.xlabel("Color-flip probability")
    plt.title(f"Per-environment losses with adaptive parameters ({args.selection_metric})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.output, dpi=200)
    print(f"Saved adaptive per-env losses to {args.output}")

    print("Selected IRO alphas per environment:")
    for p, alpha in zip(env_ps, iro_selected):
        print(f"  p={p:.2f}: alpha={alpha:.3f}")
    print("Selected ESRM gammas per environment:")
    for p, gamma in zip(env_ps, esrm_selected):
        print(f"  p={p:.2f}: gamma={gamma:.3f}")


if __name__ == "__main__":
    main()
