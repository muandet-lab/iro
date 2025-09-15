# iro — Imprecise Risk Optimization in Python

![PyPI version](https://img.shields.io/badge/PyPI-not%20published-yellow)
[![Documentation Status](https://readthedocs.org/projects/iro/badge/?version=latest)](https://iro.readthedocs.io/en/latest/)

**iro** is a Python library for Imprecise Risk Optimization (IRO), providing tools for risk aggregation and optimization under uncertainty. It implements ideas from the domain generalization literature, enabling robust learning across multiple environments.

* Free software: MIT License

#### Reference
Our IRO framework is, at its core, based on the ICML24 paper "Domain Generalisation via Imprecise Learning".

> Muandet, K., et al. *"Domain Generalisation via Imprecise Learning."* 2025. [arXiv link](https://arxiv.org/abs/2404.04669)

## Features

Currently implemented aggregation functions for risk include:

- **Conditional Value-at-Risk (CVaR)**  
- **Value-at-Risk (VaR)**  
- **Entropic risk measure**  
- **Mean (expected risk)**  
- **Worst-case risk**  
- **Median (robust alternative to mean)**  
- **Variance (penalizing spread)**  

Other planned features:

- Full support for **imprecise/domain-adaptive (risk) optimization algorithms**  
- Integration with **multiple dataset environments** for domain generalization  
- Extensive documentation and tutorials  

## Installation

The package is **not yet published on PyPI**. We plan to reserve the PyPI name `iropy` for future releases.  

For now, you can install directly from this GitHub repository:

```bash
git clone https://github.com/muandet-lab/iro.git
cd iro
pip install -e .