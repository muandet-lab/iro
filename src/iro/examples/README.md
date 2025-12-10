# Examples: how to run

Set the repo `src/` on your `PYTHONPATH` (or `pip install -e .`), then run the examples directly.

```bash
# Option 1: temporary path
PYTHONPATH=src python -m iro.examples.smoke_tests

# Option 2: editable install
pip install -e .
python -m iro.examples.smoke_tests
```

## Smoke tests
- `python -m iro.examples.smoke_tests`  
  Checks aggregator outputs (CVaR/EVaR/ESRM) and runs a tiny ERM loop on synthetic data.

## Iris aggregation demo
- `python src/iro/examples/iris_aggregation_generalization.py`  
  Trains per-domain logistic regressors on synthetic Iris shifts, summarizes CVaR/ESRM/EVaR aggregations, and plots trade-offs.

## Credit card IRO demos
- `python src/iro/examples/creditcard_iro.py`  
  Downloads the UCI credit default dataset, trains ERM + stochastic CVaR/ESRM/EVaR variants, prints aggregation tables, and plots CVaR curves.

## CMNIST experiments
Set a data cache (avoid storing data in the repo):
```bash
export IRO_DATA_DIR=/path/to/cache
```

- Training:  
  `PYTHONPATH=src python src/iro/cmnist_exp/train_sandbox.py --algorithm iro --data_dir "$IRO_DATA_DIR"`  
  Swap `--algorithm esrm` or `--algorithm evar` for other risk measures.

- Visualizations (after training; point to your checkpoints):  
  ```bash
  PYTHONPATH=src python src/iro/cmnist_exp/vis/risk_distribution_hist.py --iro_ckpt <ckpt> --esrm_ckpt <ckpt> --data_dir "$IRO_DATA_DIR"
  PYTHONPATH=src python src/iro/cmnist_exp/vis/pareto_front.py --iro_ckpt <ckpt> --esrm_ckpt <ckpt> --data_dir "$IRO_DATA_DIR"
  PYTHONPATH=src python src/iro/cmnist_exp/vis/per_env_loss_fixed.py --iro_ckpt <ckpt> --esrm_ckpt <ckpt> --data_dir "$IRO_DATA_DIR"
  PYTHONPATH=src python src/iro/cmnist_exp/vis/per_env_loss_stable.py --iro_ckpt <ckpt> --esrm_ckpt <ckpt> --data_dir "$IRO_DATA_DIR"
  PYTHONPATH=src python src/iro/cmnist_exp/vis/risk_curves.py --iro_ckpt <ckpt> --esrm_ckpt <ckpt> --data_dir "$IRO_DATA_DIR"
  PYTHONPATH=src python src/iro/cmnist_exp/vis/risk_curves_stable.py --iro_ckpt <ckpt> --esrm_ckpt <ckpt> --data_dir "$IRO_DATA_DIR"
  ```

If you omit `--data_dir`, scripts fall back to `IRO_DATA_DIR` or `~/.cache/iro/data`.
