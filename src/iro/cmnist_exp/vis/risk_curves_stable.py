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


def _parse_params(spec, env_len):
    vals = [float(x.strip()) for x in spec.split(",") if x.strip()]
    if len(vals) == 1:
        return vals * env_len
    if len(vals) != env_len:
        raise ValueError("Parameter list must have length 1 or match env_grid length.")
    return vals


def _parse_candidates(spec):
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


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
        description="Risk curves with optional stabilization of IRO and ESRM risk parameters."
    )
    parser.add_argument("--iro_ckpt", type=str, required=True,
                        help="Checkpoint produced by train_sandbox.py for IRO.")
    parser.add_argument("--esrm_ckpt", type=str, required=True,
                        help="Checkpoint produced by train_sandbox.py for ESRM.")
    parser.add_argument("--env_grid", type=str,
                        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                        help="Comma-separated probabilities defining evaluation environments.")
    parser.add_argument("--iro_alpha", type=str, default="0.7",
                        help="Single value or comma-separated list matching env_grid.")
    parser.add_argument("--iro_alpha_candidates", type=str, default="",
                        help="(Optional) comma-separated alpha candidates to pick per environment.")
    parser.add_argument("--esrm_gamma_candidates", type=str,
                        default="0.1,0.2,0.4,0.7,1.0,1.5,2.0,3.0",
                        help="Comma-separated gamma candidates considered for each environment.")
    parser.add_argument("--selection_metric", type=str, default="loss",
                        choices=["loss", "accuracy"],
                        help="Metric used to pick the gamma per environment.")
    parser.add_argument("--data_dir", type=str, default="../../data/",
                        help="Root directory for CMNIST data.")
    parser.add_argument("--batch_size", type=int, default=5000,
                        help="Batch size for evaluation loaders.")
    parser.add_argument("--label_noise", type=float, default=0.25)
    parser.add_argument("--full_resolution", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default="vis_risk_curves_stable.png")
    args = parser.parse_args()

    env_ps = parse_probabilities(args.env_grid)
    if len(env_ps) == 0:
        raise ValueError("Provide at least one environment probability via --env_grid.")
    adaptive_iro = len(args.iro_alpha_candidates.strip()) > 0
    if adaptive_iro:
        iro_candidates = _parse_candidates(args.iro_alpha_candidates)
        if len(iro_candidates) == 0:
            raise ValueError("Provide at least one alpha candidate via --iro_alpha_candidates.")
        iro_params = None
    else:
        iro_params = _parse_params(args.iro_alpha, len(env_ps))
        iro_candidates = None
    gamma_candidates = _parse_candidates(args.esrm_gamma_candidates)
    if len(gamma_candidates) == 0:
        raise ValueError("Provide at least one gamma candidate via --esrm_gamma_candidates.")

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

    iro_accs, iro_losses, selected_alphas = [], [], []
    esrm_accs, esrm_losses, esrm_gammas = [], [], []

    for idx, loader in enumerate(loaders):
        if adaptive_iro:
            alpha, acc, loss = evaluate_param(
                loader, iro_algorithm, iro_loss_fn, device,
                iro_candidates, args.selection_metric
            )
        else:
            alpha = iro_params[idx]
            acc = misc.accuracy(iro_algorithm, loader, device, alpha=alpha)
            loss = misc.loss(iro_algorithm, loader, iro_loss_fn, device, alpha=alpha)
        selected_alphas.append(alpha)
        iro_accs.append(acc)
        iro_losses.append(loss)

        gamma, acc_e, loss_e = evaluate_param(
            loader, esrm_algorithm, esrm_loss_fn, device,
            gamma_candidates, args.selection_metric
        )
        esrm_gammas.append(gamma)
        esrm_accs.append(acc_e)
        esrm_losses.append(loss_e)

    env_ps_np = np.array(env_ps)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    axes[0, 0].plot(env_ps_np, iro_accs, label="IRO (adaptive)" if adaptive_iro else "IRO", marker="o")
    axes[0, 0].plot(env_ps_np, esrm_accs, label="ESRM (adaptive)", marker="s")
    axes[0, 0].set_xlabel("Color-flip probability")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_title("Accuracy vs environment shift")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].plot(env_ps_np, iro_losses, label="IRO (adaptive)" if adaptive_iro else "IRO", marker="o")
    axes[0, 1].plot(env_ps_np, esrm_losses, label="ESRM (adaptive)", marker="s")
    axes[0, 1].set_xlabel("Color-flip probability")
    axes[0, 1].set_ylabel("w loss")
    axes[0, 1].set_title("Risk vs environment shift")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].plot(env_ps_np, selected_alphas, marker="^", color="#1f77b4")
    axes[1, 0].set_xlabel("Color-flip probability")
    axes[1, 0].set_ylabel("Selected alpha")
    axes[1, 0].set_title(f"IRO alpha per env ({'adaptive' if adaptive_iro else 'fixed'})")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(env_ps_np, esrm_gammas, marker="^", color="#d62728")
    axes[1, 1].set_xlabel("Color-flip probability")
    axes[1, 1].set_ylabel("Selected gamma")
    axes[1, 1].set_title(f"ESRM gamma per env ({args.selection_metric})")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle("Risk vs environment with adaptive risk parameters")
    fig.tight_layout()
    fig.savefig(args.output, dpi=200)

    if adaptive_iro:
        print("Selected IRO alphas per environment:")
    else:
        print("IRO alphas per environment (fixed inputs):")
    for p, alpha in zip(env_ps, selected_alphas):
        print(f"  env p={p:.2f}: alpha={alpha:.3f}")

    print("Selected ESRM gammas per environment:")
    for p, gamma in zip(env_ps, esrm_gammas):
        print(f"  env p={p:.2f}: gamma={gamma:.3f}")
    print(f"Saved stabilized risk curves to {args.output}")


if __name__ == "__main__":
    main()
