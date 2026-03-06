# Installation

## From Source

```bash
git clone https://github.com/muandet-lab/iro.git
cd iro
pip install -e .
```

## Verify Installation

```bash
python -m iro --help
```

You should see commands:
- `train`
- `eval`

## Artifact Defaults

After `python -m iro train ...`, artifacts are written under `./iro_exp` by default:
- `logs/reproduce/out.txt`
- `logs/reproduce/err.txt`
- `results/reproduce/<run_id>.jsonl`
- `ckpts/<run_id>_final.pkl`
- `ckpts/<run_id>_best.pkl`

Evaluate a checkpoint:

```bash
python -m iro eval --experiment cmnist_iro \
  -o eval.checkpoint_path=./iro_exp/ckpts/<run_id>_final.pkl \
  -o eval.split=test \
  -o eval.alpha=1.0
```

iWildCam-WILDS evaluation:

```bash
python -m iro eval --experiment iwildcam_iro \
  -o eval.checkpoint_path=./iro_exp/ckpts/<run_id>_best.pkl \
  -o eval.split=all \
  -o eval.alpha=0.8
```
