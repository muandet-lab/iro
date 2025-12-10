import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

# Ensure the toolbox (repo/src) is on the path when running this script directly.
SRC_ROOT = Path(__file__).resolve().parents[2]  # .../src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from iro.utility.fast_data_loader import InfiniteDataLoader, FastDataLoader
from iro.utility import misc
from iro.utility import algorithms
import networks as networks
from datasets import get_cmnist_datasets

# +
parser = argparse.ArgumentParser(description='Colored MNIST')
# Datasets
parser.add_argument('--train_envs', type=str, default='0.01, 0.12, 0.0, 0.0, 0.99, 0.5, 0.7, 0.01, 0.0, 0.0, 0.14')
parser.add_argument('--test_envs', type=str, default='0.1,0.5,0.9')     # test envs to log/print
parser.add_argument('--test_env_ms', type=str, default='0.9')               # test env for selecting best model
parser.add_argument('--full_resolution', action='store_true')

# Network architecture
parser.add_argument('--network', type=str, default="FiLMedMLP")
parser.add_argument('--mlp_hidden_dim', type=int, default=390)

# Algorithms
parser.add_argument('--algorithm', type=str, default='iro')
parser.add_argument('--penalty_weight', type=float, default=1000)           # irm, vrex, etc.
parser.add_argument('--alpha', type=float, default=0.4)                     # qrm
parser.add_argument('--groupdro_eta', type=float, default=1.)               # group_dro

# General hparams
parser.add_argument('--steps', type=int, default=600)
parser.add_argument('--batch_size', type=int, default=25000)
parser.add_argument('--loss_fn', type=str, default='nll', choices=["nll", "cross_ent"])
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--lr_factor_reduction', type=float, default=1)
parser.add_argument('--lr_cos_sched', action='store_true')
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--dropout_p', type=float, default=0.2)
parser.add_argument('--erm_pretrain_iters', type=int, default=0)
parser.add_argument('--eval_freq', type=int, default=50)
parser.add_argument('--eval_risk_params', type=str, default='0.0,1.0,1.0',
                    help='Comma-separated risk parameters used when evaluating on --test_envs '
                         'during training for algorithms that require a risk input (IRO/ESRM/EVAR). '
                         'If a single value is provided it is broadcast to every env.')
parser.add_argument('--final_risk_params', type=str,
                    default='0.0,0.0,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0',
                    help='Risk parameters applied during the full post-training sweep over '
                         'all color correlations (used by IRO/ESRM/EVAR). Must match the number of '
                         'environments evaluated (11 by default).')
parser.add_argument('--user_risk_params', type=str, default='',
                    help='Optional extra risk parameters to log on the main selection env '
                         '(or --user_risk_env). Allows specifying risk aversion at test time.')
parser.add_argument('--user_risk_env', type=str, default=None,
                    help='Environment probability to pair with --user_risk_params. Defaults to '
                         '--test_env_ms if not set.')

# Directories and saving
parser.add_argument('--data_dir', type=str, default=None,
                    help="Root folder for datasets. Uses $IRO_DATA_DIR or ~/.cache/iro/data when omitted.")
parser.add_argument('--output_dir', type=str, default="../../cmnist_exp")
parser.add_argument('--exp_name', type=str, default="reproduce")
parser.add_argument('--save_ckpts', action='store_true')

# Reproducibility
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--deterministic', action='store_true')
parser.add_argument('--n_workers', type=int, default=0)

# --------  SETUP --------
default_args = argparse.Namespace(n_workers=0, other_arg='default')
args = parser.parse_args(namespace=default_args)

def resolve_data_dir(requested):
    """Pick a data directory without hard-coding large assets into the repo."""
    if requested and len(str(requested).strip()) > 0:
        return requested
    env_dir = os.getenv("IRO_DATA_DIR")
    if env_dir:
        return env_dir
    return os.path.expanduser("~/.cache/iro/data")

args.data_dir = resolve_data_dir(args.data_dir)
os.makedirs(args.data_dir, exist_ok=True)
md5_fname = hashlib.md5(str(args).encode('utf-8')).hexdigest()
def _sanitize(tag):
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(tag))
ckpt_prefix = f"{_sanitize(args.algorithm.lower())}_{_sanitize(args.exp_name)}_seed{args.seed}"

# +
alg_arg_keys = ["algorithm", "penalty_weight", "alpha", "groupdro_eta",
                "lr_factor_reduction", "lr_cos_sched", "steps", "save_ckpts"]
if args.loss_fn == "nll":
    n_targets = 1
    loss_fn = F.binary_cross_entropy_with_logits
    int_target = False
else:
    n_targets = 2
    loss_fn = F.cross_entropy
    int_target = True

test_env_ps = tuple(float(e) for e in args.test_envs.split(","))
if args.train_envs == 'default':
    train_env_ps = (0.1, 0.2)
elif args.train_envs == 'gray':
    train_env_ps = (0.5, 0.5)
else:
    train_env_ps = tuple(float(e) for e in args.train_envs.split(","))

args.train_env_ps = train_env_ps
train_env_names = [str(p) for p in train_env_ps]
test_env_names = [str(p) for p in test_env_ps]
risk_aware_algs = {'iro', 'inftask', 'esrm', 'evar'}

def parse_risk_params(spec, expected_len=None, flag_name="risk parameters"):
    if spec is None:
        return None
    spec = spec.strip()
    if len(spec) == 0:
        return None
    values = [float(s.strip()) for s in spec.split(',') if len(s.strip()) > 0]
    if expected_len is not None and len(values) not in (1, expected_len):
        raise ValueError(f"{flag_name} expects 1 or {expected_len} values, got {len(values)}.")
    if expected_len is not None and len(values) == 1:
        values = values * expected_len
    return values

if args.algorithm.lower() in risk_aware_algs:
    eval_risk_params = parse_risk_params(args.eval_risk_params, len(test_env_ps), "--eval_risk_params")
    user_risk_params = parse_risk_params(args.user_risk_params, None, "--user_risk_params")
else:
    eval_risk_params = None
    user_risk_params = None

# --------  LOGGING --------
logs_dir = os.path.join(args.output_dir, "logs", args.exp_name)
results_dir = os.path.join(args.output_dir, "results", args.exp_name)
ckpt_dir = os.path.join(args.output_dir, "ckpts")
os.makedirs(logs_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

sys.stdout = misc.Tee(os.path.join(logs_dir, 'out.txt'))
sys.stderr = misc.Tee(os.path.join(logs_dir, 'err.txt'))
print('Args:')
for k, v in sorted(vars(args).items()):
    print('\t{}: {}'.format(k, v))

# -------- REPRODUCIBILITY --------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
seed_all(args.seed)

if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------  DEVICE --------
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# --------  DATA LOADING --------
envs = get_cmnist_datasets(args.data_dir, train_envs=train_env_ps, test_envs=test_env_ps, label_noise_rate = 0.25, 
                           cuda=(device == "cuda"), int_target=int_target, subsample=not args.full_resolution)
train_envs, test_envs = envs[:len(train_env_ps)], envs[len(train_env_ps):]
input_shape = train_envs[0].tensors[0].size()[1:]
n_train_samples = train_envs[0].tensors[0].size()[0]
steps_per_epoch = n_train_samples / args.batch_size

train_loaders = [FastDataLoader(dataset=env, batch_size=args.batch_size, num_workers=args.n_workers)
                 for env in train_envs]
test_loaders = [FastDataLoader(dataset=env, batch_size=args.batch_size, num_workers=args.n_workers)
                for env in test_envs]
train_minibatches_iterator = zip(*train_loaders)

# --------  NETWORK --------
if args.network == "MLP":
    net = networks.MLP(np.prod(input_shape), 
                       args.mlp_hidden_dim, 
                       n_targets, 
                       dropout=args.dropout_p)
elif args.network == "FiLMedMLP":
    net = networks.FiLMedMLP(np.prod(input_shape), 
                             args.mlp_hidden_dim, 
                             n_targets, 
                             dropout=args.dropout_p,
                            film_dim=1)
elif args.network == "CNN":
    net = networks.CNN(input_shape)
else:
    raise NotImplementedError
# -

# -------- ALGORITHM --------
algorithm_class = algorithms.get_algorithm_class(args.algorithm)
algorithm = algorithm_class(net, vars(args), loss_fn)
algorithm.to(device)

# +
# -------- LOAD ERM CHECKPOINT --------
start_step = 1
if args.erm_pretrain_iters > 0:
    erm_args = vars(copy.deepcopy(args))
    for k in alg_arg_keys:
        del erm_args[k]
    erm_ckpt_name = hashlib.md5(str(erm_args).encode('utf-8')).hexdigest()
    erm_ckpt_pth = os.path.join(ckpt_dir, f"{ckpt_prefix}_erm_{erm_ckpt_name}.pkl")
    if os.path.exists(erm_ckpt_pth):
        algorithm.load_state_dict(torch.load(erm_ckpt_pth, map_location=device), strict=False)
        print(f"ERM-pretrained model loaded: {erm_ckpt_name}.")
        start_step = args.erm_pretrain_iters + 1

# -------- LR SCHEDULING --------
def adjust_learning_rate(optimizer, current_step, lr, total_steps):
    lr_adj = lr
    lr_adj *= 0.5 * (1. + math.cos(math.pi * current_step / total_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_adj


# +
# -------- UPDATES --------
h_alphas_train = [0.0,1.0,1.0]
results = {}
best_acc, best_weights = 0., copy.deepcopy(algorithm.state_dict())
start_time, step_since_eval = time.time(), 0

for step in range(start_step, args.steps + 1):
    step_start_time = time.time()

    # -------- ADJUST LR --------
    if args.lr_cos_sched and args.algorithm.lower() != "erm":
        if args.erm_pretrain_iters == 0:
            adjust_learning_rate(algorithm.optimizer, step, args.lr, args.steps)
        elif step > args.erm_pretrain_iters > 0:
            lr_ = args.lr / args.lr_factor_reduction
            steps_ = args.steps - args.erm_pretrain_iters
            step_ = step - args.erm_pretrain_iters
            adjust_learning_rate(algorithm.optimizer, step_, lr_, steps_)

    # -------- STEP --------
    try:
        minibatch_train = next(train_minibatches_iterator)
    except StopIteration:
        train_minibatches_iterator = zip(*train_loaders)
        minibatch_train = next(train_minibatches_iterator)
    step_values = algorithm.update(minibatch_train)
    # -------- EVALUATION --------
    if step % args.eval_freq == 0 or step == args.steps:
        results.update({
            'step': step,
            'epoch': step / steps_per_epoch,
            'avg_step_time': (time.time() - start_time) / (step - step_since_eval),
        })

        for key, val in step_values.items():
            results[key] = val
        for i, (env_name, env_loader) in enumerate(zip(test_env_names, test_loaders)):
            if args.algorithm.lower() not in risk_aware_algs:
                results[env_name + '_acc'] = misc.accuracy(algorithm, env_loader, device)
                results[env_name + '_loss'] = misc.loss(algorithm, env_loader, loss_fn, device)
            else:
                risk_param = eval_risk_params[i]
                results[env_name + '_acc'] = misc.accuracy(algorithm, env_loader, device, alpha=risk_param)
                results[env_name + '_loss'] = misc.loss(algorithm, env_loader, loss_fn, device, alpha=risk_param)
        
        results['mem_gb'] = torch.cuda.max_memory_allocated() / (1024. * 1024. * 1024.)
        results_keys = sorted(results.keys())

        misc.print_row(results_keys, colwidth=12)
        misc.print_row([results[key] for key in results_keys], colwidth=12)

        start_time, step_since_eval = time.time(), 0
        if results[args.test_env_ms + '_acc'] > best_acc:
            best_acc = results[args.test_env_ms + '_acc']
            best_weights = copy.deepcopy(algorithm.state_dict())

    # -------- SAVE ERM CHECKPOINT --------
    if step == args.erm_pretrain_iters > 0 and args.save_ckpts:
        torch.save(algorithm.state_dict(), erm_ckpt_pth)
        print("Saved ERM-pretrained model.")
# -

# -------- FINAL EVAL ON ALL ENVS AND HELD-OUT TEST SET --------
all_ps = [i / 10. for i in range(11)]
all_env_names = [str(p) for p in all_ps]
all_envs = get_cmnist_datasets(args.data_dir, train_envs=[], test_envs=all_ps, cuda=(device == "cuda"),
                               int_target=int_target, subsample=not args.full_resolution, use_test_set=True)
loaders = [FastDataLoader(dataset=env, batch_size=5000, num_workers=args.n_workers)
           for env in all_envs]
#since you know for ratio > 0.5 the color flips and you would be better off being invariant
if args.algorithm.lower() in risk_aware_algs:
    final_risk_params = parse_risk_params(args.final_risk_params, len(all_env_names), "--final_risk_params")
else:
    final_risk_params = None
results = {}
for ms_name in ["final", "best"]:
    if ms_name == "best":
        algorithm.load_state_dict(best_weights)

    # -------- EVAL --------
    for i, (env_name, env_loader) in enumerate(zip(all_env_names, loaders)):
        if args.algorithm.lower() not in risk_aware_algs:
            results[env_name+'_acc_'+ms_name] = misc.accuracy(algorithm, env_loader, device)
            results[env_name+'_loss_'+ms_name] = misc.loss(algorithm, env_loader, loss_fn, device)
        else:
            risk_param = final_risk_params[i]
            results[env_name+'_acc_'+ms_name] = misc.accuracy(algorithm,env_loader,device, alpha=risk_param)
            results[env_name+'_loss_'+ms_name] = misc.loss(algorithm,env_loader,loss_fn,device, alpha=risk_param)
    # -------- PRINT -------- 
    misc.cvar(algorithm, loaders, loss_fn, device, all_ps, args.algorithm.lower() not in risk_aware_algs)
    print(f"\n{ms_name} accuracies:")
    results_print_keys = [k for k in sorted(results.keys()) if f"_acc_{ms_name}" in k]
    misc.print_row([k.replace(f"_acc_{ms_name}", "") for k in results_print_keys], colwidth=5)
    misc.print_row([round(results[k], 3) for k in results_print_keys], colwidth=5)

    if user_risk_params:
        target_env = args.user_risk_env if args.user_risk_env is not None else args.test_env_ms
        if target_env not in all_env_names:
            raise ValueError(f"--user_risk_env ({target_env}) must be one of {all_env_names}.")
        target_idx = all_env_names.index(target_env)
        target_loader = loaders[target_idx]
        print(f"\nUser-specified risk metrics on env {target_env} ({ms_name} weights):")
        for risk_param in user_risk_params:
            acc = misc.accuracy(algorithm, target_loader, device, alpha=risk_param)
            loss_val = misc.loss(algorithm, target_loader, loss_fn, device, alpha=risk_param)
            print(f"  risk={risk_param:.4f} -> acc={acc:.4f}, loss={loss_val:.4f}")

    # -------- SAVE CHECKPOINT --------
    if args.save_ckpts:
        ckpt_save_dict = {"args": vars(args), "model_dict": algorithm.state_dict()}
        ckpt_name = f"{ckpt_prefix}_{ms_name}.pkl"
        torch.save(ckpt_save_dict, os.path.join(ckpt_dir, ckpt_name))
        print(f"Saved checkpoint: {ckpt_name}")


# +
# -------- SAVE ALL RESULTS --------
# Create args_id without seed to allow the mean over seeds to be easily computed in collect_results.py
args_no_seed = copy.deepcopy(args)
delattr(args_no_seed, "seed")
args_id = hashlib.md5(str(args_no_seed).encode('utf-8')).hexdigest()

if (args.train_envs == 'gray') and (args.algorithm.lower() == "erm"):
    results["algorithm"] = "oracle"
else:
    results["algorithm"] = args.algorithm.lower()
results["seed"] = args.seed
results["args_id"] = args_id
results["args"] = vars(args_no_seed)

with open(os.path.join(results_dir, f"{md5_fname}.jsonl"), 'a') as f:
    f.write(json.dumps(results, sort_keys=True) + "\n")
# -
